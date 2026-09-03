#!/bin/bash
# @image   harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @name  q38-tp8tree
# @port  8108
# @gpus  0,1,2,3,4,5,6,7
# ★ TP8 混合传输版（2026-08-23）：同 socket 走 VRAM P2P、跨 socket 走 SHM
# 与 serve_tp8.sh 的差别只有三处：
#   1. NCCL_P2P_DISABLE=1  →  NCCL_P2P_LEVEL=PXB + NCCL_ALGO=Ring
#   2. 加 LD_PRELOAD=/w/dlhook2.so + DLHOOK2_AUTO=1（按 socket 过滤 agent 列表）
#   3. 容器名 q38-tp8 → q38-tp8h
# --disable-custom-all-reduce 必须保留：自研 all-reduce 会自己跨全部 8 rank 做 IPC，
# 那条路绕不过活锁，跟 NCCL 怎么设无关。
# 实测集合带宽 1.5 → 5.7 GB/s（3.75×），正确性 27/27。详见 docs/复盘三-52ms活锁根因与TP8混合传输.md
set -e
NAME="${NAME:-q38-tp8tree}"
# 本账号可能不在 docker 组但有 sudo NOPASSWD
# 目录中立：WORK 由 launch.sh 导出（脚本树根）；单独运行时回退旧约定路径
W="${WORK:-/data/q38-work}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
mkdir -p "$W/tritoncache" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
YARN='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
SPEC='{"method":"mtp","num_speculative_tokens":3}'
CC='{"cudagraph_mode":"PIECEWISE"}'
V=/usr/local/lib/python3.10/dist-packages/vllm
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" \
  --network host --privileged --shm-size 256g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v $MODELS_ROOT/Qwen3.8-27B:/data/models/Qwen3.8-27B:ro \
  -v $W:/w \
  -v $W/patches/gdn_prefix_cache.py:$V/model_executor/layers/mamba/gdn_prefix_cache.py:ro \
  -v $W/patches/kv_cache_utils.py:$V/v1/core/kv_cache_utils.py:ro \
  -v $W/patches/gpu_model_runner.py:$V/v1/worker/gpu_model_runner.py:ro \
  -v $W/patches/layers_mamba_utils.py:$V/model_executor/layers/mamba/mamba_utils.py:ro \
  -v $W/patches/gdn_attn.py:$V/v1/attention/backends/gdn_attn.py:ro \
  -v $W/patches/gdn_linear_attn.py:$V/model_executor/layers/mamba/gdn_linear_attn.py:ro \
  -v $W/tritoncache:/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e VLLM_PLUGINS=hcu \
  -e VLLM_NCCL_SO_PATH=/opt/dtk/lib/librccl.so.1 -e VLLM_DISABLE_PYNCCL=0 \
  -e NCCL_P2P_LEVEL=PXB -e NCCL_ALGO=Tree -e LD_PRELOAD=/w/dlhook2.so -e DLHOOK2_AUTO=1 -e NCCL_DEBUG=WARN -e ALLREDUCE_STREAM_WITH_COMPUTE=1 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706 \
  -c "python3 /w/patches/patch_gdn_cache.py && exec python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.8-27B --served-model-name qwen38 \
  --trust-remote-code --dtype bfloat16 --tensor-parallel-size 8 \
  --max-model-len 524288 --max-num-seqs 16 --gpu-memory-utilization 0.90 \
--max-num-batched-tokens 8192 \
  --compilation-config '$CC' \
  --speculative-config '$SPEC' \
  --reasoning-parser qwen3 \
  --hf-overrides '$YARN' \
  --disable-custom-all-reduce --enable-prefix-caching --mamba-cache-mode all --mamba-block-size 2048 --block-size 1024 \
  --host 0.0.0.0 --port 8108"
echo "q38-tp8tree (TP8 混合传输 + Tree，长上下文冷启优先) started, port 8108"
