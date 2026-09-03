#!/bin/bash
# @image      harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
# @name  dsv4-dspark
# @port  8120
# @gpus  0,1,2,3,4,5,6,7
# @expect-gpu 8
# @requires /data1/models/dsv4-0731-w8a8-dspark $CFGDIR/inner/run_prodfix.sh
# @attest 投机解码DSPARK:speculative
# ============================================================================
# DSv4-01 · 0731-w8a8 + DSpark · TP8 —— 贪心主线（2026-08-14 定稿生产线固化）
#
# 实测：单流 decode 33.2 tok/s（编程类），DSpark accept 4.38/0.68；
#       8 并发 8/8、10 并发 10/10 稳定；KV 池 1.03M token。
#
# 硬边界（均已实证，详见线内 README）：
#   ⚠ 只在 temperature=0（贪心）稳定；temp>0 且并发≥8 触发 GPU 硬件异常
#     HSA_STATUS_ERROR_EXCEPTION 0x1016，服务挂死需重启（~14 分钟）。
#     需要采样+高并发 → 用 02 线（无投机，任意温度稳定）。
#   ⚠ 勿改 --kv-cache-dtype：解码内核按 fp8 布局读 KV，bf16 会静默乱码
#     且 accept rate 恒 1.00（这是告警信号不是好消息）。
#   ⚠ Think 需请求显式带 chat_template_kwargs={"thinking": true}。
#   ⚠ 就绪 ~14 分钟（triton JIT + graph 捕获）；就绪后首次生成必偏慢（JIT 污染，
#     12.3 tok/s），第二次起才是真实值——压测前先热身两次。
#
# 权重：自量化产物（本包 quant/ 内含完整配方：bf16 原模型 → 0731 W8A8 + DSpark 张量）。
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; PORT="${PORT:-8120}"; NAME="${NAME:-dsv4-dspark}"
CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DSV4_MODELS_ROOT="${DSV4_MODELS_ROOT:-/data1/models}"
MODEL="$DSV4_MODELS_ROOT/dsv4-0731-w8a8-dspark"
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

[ -f "$MODEL/config.json" ] || {
  echo "✗ 缺权重 $MODEL —— 本线权重为自量化产物，无法自动下载。"
  echo "  配方在包内 quant/：从公开 bf16 原模型量化生成（约数小时），见 quant/README.md"
  exit 1
}

echo "DSv4-01 贪心主线 $NAME: GPU=$GPUS PORT=$PORT (TP8)"
$DOCKER run -d --name "$NAME" \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal \
  -v "$MODEL":/models:ro \
  -v "$CFGDIR/inner":/patches:ro \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e NCCL_P2P_DISABLE=1 -e PORT=$PORT -e SPEC_ALGO=dspark \
  -e MEM_FRACTION_STATIC="${MEM_FRAC:-0.85}" -e CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}" \
  -e PREFILL_CHUNK="${PREFILL_CHUNK:-4096}" -e MAX_RUNNING="${MAX_RUNNING:-}" \
  -w /patches --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  run_prodfix.sh >/dev/null

echo "$NAME started on $PORT（就绪 ~14 分钟；探测 /health；就绪后先热身两次再测）"
