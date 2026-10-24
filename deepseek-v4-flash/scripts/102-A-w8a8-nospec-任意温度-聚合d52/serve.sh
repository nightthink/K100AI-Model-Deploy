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
# ★ 2026-09-12 改版：P2P 解禁（NCCL_P2P_DISABLE=1 → NCCL_P2P_LEVEL=SYS）
#
#   依据来自 **101 线**的单变量 A/B（各 2 小时、同 harness、命中率 91.2% vs 91.7%）：
#   整机吞吐 +26.8%、并发 16 路 decode +28.7%、冷启 prefill +47.9%、起服 −24.7%，
#   全面胜出无一项回退。详见 docs/对照-101线P2P解禁-2026-09-12.md。
#
#   ⚠ **本线未单独做 P2P A/B**——它与 101 线同镜像、同权重谱系（仅剔除 DSpark
#     张量）、同 inner 补丁，差别只在不开投机，故一并采用。若日后要引用本线的
#     P2P 收益数字，须自己跑一轮，不能直接套 101 的。
#   ⚠ **硬前提**：驱动必须已打 v4 补丁（hycu.ko 的 gpu_dma_buf_attach 一字节改）。
#     未打而开 P2P，跨 socket 会触发驱逐活锁。
#   ⚠ custom AR 仍开不了，与 P2P 无关：本镜像是上游原版，TP8 且无 XGMI 时直接
#     return（实测 is_full_nvlink([0..7],8)=False，8 卡全 PCIe 2~3 跳）；
#     QuickAllReduce 白名单也没有 gfx928。故 --disable-custom-all-reduce 保留。
#
#   2026-09-12 本线复验（P2P 开启）：temp=0.7 top_p=0.95 × 8 并发 **8/8 成功**、
#   聚合 50.1 tok/s，与定稿时的 52.2 同量级——「任意温度稳定」这一卖点仍成立。
#
# 实测（2026-08-14 定稿时，P2P 禁用）：temperature=0.7 × 8 并发 8/8 全通，聚合 52.2 tok/s；单流 ~12.3 tok/s。
#   定位：需要采样（temperature>0）+ 高并发的场景；101 线（DSpark 贪心）的安全后备。
#   权重为 101 线权重剔除 DSpark 张量的版本（quant/ 配方同时产出两者）。
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
  ${EXTRA_DOCKER_ENV:-} -e NCCL_P2P_LEVEL=SYS -e PORT=$PORT -e SPEC_ALGO=none \
  -e MEM_FRACTION_STATIC="${MEM_FRAC:-0.85}" -e CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}" \
  -e PREFILL_CHUNK="${PREFILL_CHUNK:-4096}" -e MAX_RUNNING="${MAX_RUNNING:-}" \
  -w /patches --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  run_prodfix.sh >/dev/null

echo "$NAME started on $PORT（就绪 ~14 分钟；探测 /health；就绪后先热身两次再测）"
