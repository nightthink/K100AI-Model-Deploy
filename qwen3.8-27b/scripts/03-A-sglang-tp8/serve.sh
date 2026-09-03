#!/bin/bash
# @image   harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @farm    Qwen3.8-27B Qwen3.8-27B-1M
# @name  q38-sg8h
# @port  8100
# @gpus  0,1,2,3,4,5,6,7
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-1M $WORK/dlhook2-sg.so
# @attest agent过滤（活锁绕法）:→保留4个
# @attest 投机解码NEXTN:speculative
# ★ TP8 混合传输版（2026-08-23）：同 socket 走 VRAM P2P、跨 socket 走 SHM
# 与 serve_sg8.sh 的差别：NCCL_P2P_DISABLE=1 → P2P_LEVEL=PXB + ALGO=Ring，并挂 dlhook2-sg.so（AUTO）
# --disable-custom-all-reduce 必须保留。详见 docs/复盘三-52ms活锁根因与TP8混合传输.md
set -e
NAME="${NAME:-q38-sg8h}"
# 本账号有 sudo NOPASSWD；不在 docker 组就走 sudo
# 目录中立：WORK 由 launch.sh 导出（脚本树根）；单独运行时回退旧约定路径
W="${WORK:-/data/q38-work}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
mkdir -p "$W/tritoncache" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" \
  --network host --privileged --shm-size 256g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v $MODELS_ROOT:/data/models:ro \
  -v $W:/w -v $W/tritoncache:/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_P2P_LEVEL=PXB -e NCCL_ALGO=Ring -e LD_PRELOAD=/w/dlhook2-sg.so -e DLHOOK2_AUTO=1 -e DLHOOK2_LOG=1 -e NCCL_DEBUG=WARN -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  --entrypoint python3 \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  -m sglang.launch_server \
  --model-path /data/models/Qwen3.8-27B-1M --served-model-name qwen38 \
  --trust-remote-code --tp 8 \
  --context-length 1000000 --mem-fraction-static 0.90 \
  --disable-custom-all-reduce --page-size 64 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 8100
echo "q38-sg8h (TP8 混合传输) started, port 8100"
