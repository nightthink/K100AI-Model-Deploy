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
# ★ 2026-09-12 改版：P2P 解禁（NCCL_P2P_DISABLE=1 → NCCL_P2P_LEVEL=SYS）
#
#   原先禁用 P2P 的依据是「不禁就在权重加载后挂死」，但追溯发现那是 **vLLM-0706
#   镜像**的观察（Bug 报告 A-3），sglang 侧只有一句「官方启动器也这么设」的旁证。
#   v4 驱动补丁修掉跨 socket 驱逐活锁后重测，单变量 A/B（各 2 小时、同 harness、
#   命中率 91.2% vs 91.7%）结果是全面胜出、无一项回退：
#
#     整机吞吐      231.7 → 293.7 请求/h   +26.8%
#     并发 16 路    10.1  → 13.0  tok/s    +28.7%
#     并发 8 路     14.2  → 18.1  tok/s    +27.5%
#     单流 decode   32.1  → 35.4  tok/s    +10.3%
#     冷启 prefill  524   → 775   tok/s    +47.9%
#     起服就绪      1041s → 784s           −24.7%
#
#   规律是「并发越高收益越大」——并行度不变、唯一变的是传输路径，
#   并发越高 all-reduce 越密集，主机内存两跳与 P2P 直连的差距就越显。
#   详见 docs/对照-101线P2P解禁-2026-09-12.md。
#
#   ⚠ **硬前提**：驱动必须已打 v4 补丁（hycu.ko 的 gpu_dma_buf_attach 一字节改）。
#     未打补丁而开 P2P，跨 socket 会触发驱逐活锁。
#   ⚠ custom AR 仍然开不了，与 P2P 无关：本镜像是上游原版，TP8 且无 XGMI 时
#     直接 return（实测 is_full_nvlink([0..7],8)=False，8 卡全 PCIe 2~3 跳）；
#     QuickAllReduce 的白名单也没有 gfx928。故 --disable-custom-all-reduce 保留。
#
# 实测（2026-08-14 定稿时，P2P 禁用）：单流 decode 33.2 tok/s（编程类），
#       DSpark accept 4.38/0.68；8 并发 8/8、10 并发 10/10 稳定；KV 池 1.03M token。
#       上面 P2P 开启后的数取自标准 harness（bench/），口径与此处不同，勿混用。
#
# 硬边界（均已实证，详见线内 README）：
#   ⚠ 只在 temperature=0（贪心）稳定；temp>0 且并发≥8 触发 GPU 硬件异常
#     HSA_STATUS_ERROR_EXCEPTION 0x1016，服务挂死需重启（~14 分钟）。
#     需要采样+高并发 → 用 102 线（无投机，任意温度稳定）。
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
  ${EXTRA_DOCKER_ENV:-} -e NCCL_P2P_LEVEL=SYS -e PORT=$PORT -e SPEC_ALGO=dspark \
  -e MEM_FRACTION_STATIC="${MEM_FRAC:-0.85}" -e CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}" \
  -e PREFILL_CHUNK="${PREFILL_CHUNK:-4096}" -e MAX_RUNNING="${MAX_RUNNING:-}" \
  -w /patches --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  run_prodfix.sh >/dev/null

echo "$NAME started on $PORT（就绪 ~14 分钟；探测 /health；就绪后先热身两次再测）"
