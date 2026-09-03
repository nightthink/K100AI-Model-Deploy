#!/bin/bash
# @image   harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @farm    Qwen3.8-27B Qwen3.8-27B-1M
# @name  q38-sgA
# @port  8101
# @gpus  0,1,2,3
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-1M $WORK/dlhook2-sg.so
# @attest 上下文长度:context_length
# @attest 投机解码NEXTN:speculative
# ============================================================================
# A 线 TP4 调优版 —— 面向「单请求 prefill 快 + decode 快」，聚合吞吐让位
#
# 用法： GPUS=0,1,2,3 PORT=8101 NAME=q38-sgA bash serve_sg_tp4_tuned.sh
#        GPUS=4,5,6,7 PORT=8102 NAME=q38-sgB bash serve_sg_tp4_tuned.sh
#        再 bash serve_router.sh 起 8100 单一入口（DP2×TP4）
#
# 相对 serve_sg1m.sh 的每一处改动与依据：
#   prefill
#     --chunked-prefill-size 32768   默认按显存自动取 2048/4096/8192；块大→内核启动少
#     --max-prefill-tokens   45000   海光官方 K100-AI 配方（待办第 7 项）
#   两项都吃
#     ★ --disable-custom-all-reduce 必须保留（2026-08-26 实测）：去掉它 rank0 立刻 SIGSEGV，
#       调用栈 aiter::CustomAllreduce::allreduce → cross_device_reduce_2stage。
#       护栏 1 说的「ACS 清后 custom AR 可用」是 vLLM 的实现；A 线 sglang 走 aiter 的实现，
#       在 K100-AI 上直接段错误。DocPang 的 CUSTOM_AR=1 同样是 vLLM 侧，不可迁移到 sglang。
#     SGLANG_USE_CUDA_IPC_TRANSPORT=1    官方开着。★ 仅同 socket 安全，跨 socket 必触发
#                                        52ms 活锁 —— 本脚本只用于 TP4 单 socket，切勿用于 TP8
#   decode
#     SGLANG_USE_AITER_LINEAR_ATTN=1     GDN 占 48/64 层，decode 大头；官方开着，我们从未测过
#     SGLANG_USE_FUSED_TOPK_SOFTMAX=1    官方开着
#
# 未开（有争议、无人验证，留作下一轮旋钮）：
#   SGLANG_USE_LIGHTOP / SGLANG_USE_CAUSAL_CONV1D  —— 海光官方说 1，DocPang 说 0
#
# ★ 起来后必须先跑一次 measure.py 冒烟：若单流 ≈0.05 tok/s，说明 custom all-reduce
#   在 ACS 没清干净的情况下被启用了，立刻加回 --disable-custom-all-reduce
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8101}"; NAME="${NAME:-q38-sgA}"
# MEM_FRAC：DFlash 等带草稿模型的算法需调低（3.85GB 草稿模型 + 其 CUDA graph 会挤占 KV 池）
MEM_FRAC="${MEM_FRAC:-0.90}"
# MODEL_PATH：换模型用（如 W8A8 INT8）。默认仍是 1M 软链农场。
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.8-27B-1M}"
CTX="${CTX:-1000000}"; CHUNK="${CHUNK:-32768}"; MAXPRE="${MAXPRE:-45000}"
# 投机解码旋钮（实测基线 topk1/steps3/draft4 → 每步产 2.49 token，接受率 62%）
SPEC_STEPS="${SPEC_STEPS:-3}"; SPEC_TOPK="${SPEC_TOPK:-1}"; SPEC_DRAFT="${SPEC_DRAFT:-4}"
# SPEC_ALGO=NONE 关掉投机解码 —— 用于验证投机是否无损：temperature=0 下
# 投机解码应当逐 token 等价于不投机，输出不一致就说明验证环节有问题。
SPEC_ALGO="${SPEC_ALGO:-NEXTN}"
DFLASH_DRAFT="${DFLASH_DRAFT:-/data/models/Qwen3.8-27B-DFlash2}"
DFLASH_BLOCK="${DFLASH_BLOCK:-8}"          # DocPang 验证过的取值
if [ "$SPEC_ALGO" = "NONE" ]; then
  SPEC_ARGS=""
elif [ "$SPEC_ALGO" = "DFLASH" ]; then
  # DFlash2：参数与 EAGLE/NEXTN 系不同，用独立草稿模型
  SPEC_ARGS="--speculative-algorithm DFLASH --speculative-draft-model $DFLASH_DRAFT"
  SPEC_ARGS="$SPEC_ARGS --speculative-dflash-block-size $DFLASH_BLOCK"
  EXTRA_MOUNT="$EXTRA_MOUNT -v $DFLASH_DRAFT:$DFLASH_DRAFT:ro"
else
  SPEC_ARGS="--speculative-algorithm $SPEC_ALGO --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk $SPEC_TOPK --speculative-num-draft-tokens $SPEC_DRAFT"
fi
LIGHTOP="${LIGHTOP:-}"; CAUSAL_CONV1D="${CAUSAL_CONV1D:-}"   # 海光官方说 1，DocPang 说 0，未验证
EXTRA_ENV="${EXTRA_ENV_IN:-}"   # 透传任意 -e K=V（探索用）
# EXTRA_MOUNT：额外挂载，形如 "-v /host/a.json:/container/b.json:ro"。
# 用途之一：把我们自造的 aiter gfx928 内核配置挂进去（海光只提供 gfx936/gfx938，
# 且其形状表不覆盖本模型的 H/HV，实测每次都回落到 BV=32,num_warps=1）。
EXTRA_MOUNT="${EXTRA_MOUNT:-}"
# EXTRA_ARGS：追加任意 sglang 启动参数（探索用），如
#   EXTRA_ARGS="--attention-backend triton --cuda-graph-max-bs 8"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ENTRY_WRAPPER：把容器入口换成包装脚本（宿主路径），用于确定性注入运行时补丁。
# 本镜像里 sitecustomize 的自动导入不生效，且 site.py 会静默吞异常，故不用它。
ENTRY_WRAPPER="${ENTRY_WRAPPER:-}"
if [ -n "$ENTRY_WRAPPER" ]; then
  EXTRA_MOUNT="$EXTRA_MOUNT -v $(dirname "$ENTRY_WRAPPER"):/patches:ro"
  ENTRY_TARGET="/patches/$(basename "$ENTRY_WRAPPER")"
else
  ENTRY_TARGET=""
fi
[ -n "$LIGHTOP" ]       && EXTRA_ENV="$EXTRA_ENV -e SGLANG_USE_LIGHTOP=$LIGHTOP"
[ -n "$CAUSAL_CONV1D" ] && EXTRA_ENV="$EXTRA_ENV -e SGLANG_USE_CAUSAL_CONV1D=$CAUSAL_CONV1D"
D="${WORK:-/data/q38-work}"   # 目录中立：launch.sh 导出 WORK
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
mkdir -p "$D/tritoncache" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

case "$GPUS" in
  0,1,2,3|4,5,6,7) ;;
  *) echo "✗ 本脚本只允许同 socket 的四卡组（0,1,2,3 或 4,5,6,7）"; echo "  跨 socket 会让 SGLANG_USE_CUDA_IPC_TRANSPORT=1 触发 52ms 活锁"; exit 1;;
esac
eval "$(bash ${COMMON:-$D}/numa_bind.sh "$GPUS" 2>/dev/null)"   # 推导 --cpuset-cpus / --cpuset-mems
# NUMA_OVERRIDE：探索用，覆盖上面推导出的绑定（设为 "none" 表示完全不绑）
if [ -n "${NUMA_OVERRIDE+x}" ]; then
  [ "$NUMA_OVERRIDE" = "none" ] && NUMA_ARGS="" || NUMA_ARGS="$NUMA_OVERRIDE"
fi
bash "${COMMON:-$D}/preflight_acs.sh" "$GPUS" || exit 1          # custom AR 依赖 ACS 已清

echo "副本 $NAME: GPU=$GPUS PORT=$PORT ctx=$CTX chunk=$CHUNK maxpre=$MAXPRE  NUMA[$NUMA_ARGS]"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" \
  --network host --privileged $NUMA_ARGS --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  $EXTRA_MOUNT \
  -v /opt/hyhal:/opt/hyhal:ro -v $MODELS_ROOT:/data/models:ro \
  -v $D:/w -v $D/tritoncache:/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e NCCL_P2P_LEVEL=PHB -e NCCL_DEBUG=WARN -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
  -e SGLANG_USE_AITER_LINEAR_ATTN=1 \
  -e SGLANG_USE_FUSED_TOPK_SOFTMAX=1 $EXTRA_ENV \
  --entrypoint python3 \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  ${ENTRY_TARGET:--m sglang.launch_server} \
  --model-path $MODEL_PATH --served-model-name qwen38 \
  --trust-remote-code --tp 4 \
  --context-length $CTX --mem-fraction-static $MEM_FRAC --page-size 64 \
  --disable-custom-all-reduce \
  --chunked-prefill-size $CHUNK --max-prefill-tokens $MAXPRE \
  $SPEC_ARGS \
  $EXTRA_ARGS \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port $PORT
echo "$NAME started on $PORT"
