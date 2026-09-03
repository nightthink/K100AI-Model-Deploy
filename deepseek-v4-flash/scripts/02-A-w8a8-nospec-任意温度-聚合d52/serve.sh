#!/bin/bash
# @image      harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
# @name  dsv4-nospec
# @port  8121
# @gpus  0,1,2,3,4,5,6,7
# @expect-gpu 8
# @requires /data1/models/dsv4-0731-w8a8 $CFGDIR/inner/run_prodfix.sh
# 
# ============================================================================
# DSv4-02 · 0731-w8a8 无投机 · TP8 —— 任意温度稳线（2026-08-14 定稿生产线固化）
#
# 实测：temperature=0.7 × 8 并发 8/8 全通，聚合 52.2 tok/s；单流 ~12.3 tok/s。
#   定位：需要采样（temperature>0）+ 高并发的场景；01 线（DSpark 贪心）的安全后备。
#   权重为 01 线权重剔除 DSpark 张量的版本（quant/ 配方同时产出两者）。
#
# 边界：Think 需显式 chat_template_kwargs；就绪 ~14 分钟；勿改 --kv-cache-dtype。
#
# 权重：自量化产物（本包 quant/ 内含完整配方：bf16 原模型 → 0731 W8A8 + DSpark 张量）。
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; PORT="${PORT:-8121}"; NAME="${NAME:-dsv4-nospec}"
CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DSV4_MODELS_ROOT="${DSV4_MODELS_ROOT:-/data1/models}"
MODEL="$DSV4_MODELS_ROOT/dsv4-0731-w8a8"
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

[ -f "$MODEL/config.json" ] || {
  echo "✗ 缺权重 $MODEL —— 本线权重为自量化产物，无法自动下载。"
  echo "  配方在包内 quant/：从公开 bf16 原模型量化生成（约数小时），见 quant/README.md"
  exit 1
}

echo "DSv4-02 任意温度稳线 $NAME: GPU=$GPUS PORT=$PORT (TP8)"
$DOCKER run -d --name "$NAME" \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal \
  -v "$MODEL":/models:ro \
  -v "$CFGDIR/inner":/patches:ro \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e NCCL_P2P_DISABLE=1 -e PORT=$PORT -e SPEC_ALGO=none \
  -e MEM_FRACTION_STATIC="${MEM_FRAC:-0.85}" -e CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}" \
  -e PREFILL_CHUNK="${PREFILL_CHUNK:-4096}" -e MAX_RUNNING="${MAX_RUNNING:-}" \
  -w /patches --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  run_prodfix.sh >/dev/null

echo "$NAME started on $PORT（就绪 ~14 分钟；探测 /health；就绪后先热身两次再测）"
