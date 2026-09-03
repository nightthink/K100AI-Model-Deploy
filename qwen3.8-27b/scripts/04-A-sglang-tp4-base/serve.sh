#!/bin/bash
# @image   harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @name  q38-sg1m
# @port  8100
# @gpus  0,1,2,3
# @farm    Qwen3.8-27B Qwen3.8-27B-1M
set -e
# 本账号可能不在 docker 组但有 sudo NOPASSWD
# 目录中立：WORK 由 launch.sh 导出（脚本树根）；单独运行时回退旧约定路径
W="${WORK:-/data/q38-work}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
mkdir -p "$W/tritoncache" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
# 卡组可传，默认用快卡组 0-3（卡 4/5/6 是限流的 0x6211，见 docs/硬件异质-*.md）
GPUS="${GPUS:-0,1,2,3}"; NAME="${NAME:-q38-sg1m}"; PORT="${PORT:-8100}"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" \
  --network host --privileged --shm-size 256g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v $MODELS_ROOT:/data/models:ro \
  -v $W/tritoncache:/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e NCCL_P2P_LEVEL=PHB -e NCCL_DEBUG=WARN -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  --entrypoint python3 \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  -m sglang.launch_server \
  --model-path /data/models/Qwen3.8-27B-1M --served-model-name qwen38 \
  --trust-remote-code --tp 4 \
  --context-length 1000000 --mem-fraction-static 0.90 \
  --disable-custom-all-reduce --page-size 64 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port $PORT
echo "q38-sg1m started"
