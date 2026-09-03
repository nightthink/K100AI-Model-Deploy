#!/bin/bash
# @image   harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @name  q38-rep
# @port  8100
# @gpus  4,5,6,7
# ============================================================================
# 参数化副本启动：任意 GPU 组 → 自动 NUMA 绑定 + 自动选对应的修正拓扑
#   GPUS=0,1,2,3 PORT=8101 NAME=q38-r0 bash serve_replica.sh
#   GPUS=4,5,6,7 PORT=8102 NAME=q38-r1 bash serve_replica.sh
# 依赖：numa_bind.sh（cpuset 推导）、rccl-topo-socket{0,1} 修正拓扑
# ============================================================================
set -e
# 本账号可能不在 docker 组但有 sudo NOPASSWD
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
GPUS="${GPUS:-4,5,6,7}"; PORT="${PORT:-8100}"; NAME="${NAME:-q38-rep}"
MAXLEN="${MAXLEN:-524288}"; MAXSEQ="${MAXSEQ:-16}"; GPUUTIL="${GPUUTIL:-0.90}"
# 1M 用 YARN_FACTOR=4.0；TP4 上 1M 与 all 模式缓存显存互斥，需 MAMBA_ALL=0
YARN_FACTOR="${YARN_FACTOR:-2.0}"; MAMBA_ALL="${MAMBA_ALL:-1}"
MAMBA_ARG=""; [ "$MAMBA_ALL" = "1" ] && MAMBA_ARG="--mamba-cache-mode all --mamba-block-size 2048"
D="${WORK:-/data/q38-work}"   # 目录中立
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
mkdir -p "$D/tritoncache" 2>/dev/null || true

TP=$(awk -F, '{print NF}' <<< "$GPUS")
eval "$(bash ${COMMON:-$D}/numa_bind.sh "$GPUS" 2>/dev/null)"

case "$GPUS" in
  0,1,2,3) TOPO_ENV="-e NCCL_TOPO_FILE=/w/rccl-topo-socket0-fixed.xml -e NCCL_P2P_LEVEL=PHB" ;;
  4,5,6,7) TOPO_ENV="-e NCCL_TOPO_FILE=/w/rccl-topo-4card-fixed.xml -e NCCL_P2P_LEVEL=PHB" ;;
  *)       TOPO_ENV="-e NCCL_P2P_DISABLE=1"   # 跨 socket：P2P 只能全关（见 ACS 文档 §13）
           echo "⚠ 跨 socket GPU 组：P2P 全关 + 无拓扑文件；通信慢 19×，不建议" >&2 ;;
esac

YARN='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":'"$YARN_FACTOR"',"original_max_position_embeddings":262144}}}'
SPEC='{"method":"mtp","num_speculative_tokens":3}'
CC='{"cudagraph_mode":"PIECEWISE"}'
V=/usr/local/lib/python3.10/dist-packages/vllm

# ★ 起服务前先体检 ACS（同 serve_final.sh 的理由）
bash "${COMMON:-$D}/preflight_acs.sh" "$GPUS" || exit 1

echo "副本 $NAME: GPU=$GPUS TP=$TP PORT=$PORT  NUMA[$NUMA_ARGS]"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --restart unless-stopped --name "$NAME" \
  --network host --privileged $NUMA_ARGS --shm-size 64g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v $MODELS_ROOT/Qwen3.8-27B:/data/models/Qwen3.8-27B:ro \
  -v $D:/w \
  -v $D/patches/gdn_prefix_cache.py:$V/model_executor/layers/mamba/gdn_prefix_cache.py:ro \
  -v $D/patches/kv_cache_utils.py:$V/v1/core/kv_cache_utils.py:ro \
  -v $D/patches/gpu_model_runner.py:$V/v1/worker/gpu_model_runner.py:ro \
  -v $D/patches/layers_mamba_utils.py:$V/model_executor/layers/mamba/mamba_utils.py:ro \
  -v $D/patches/gdn_attn.py:$V/v1/attention/backends/gdn_attn.py:ro \
  -v $D/patches/gdn_linear_attn.py:$V/model_executor/layers/mamba/gdn_linear_attn.py:ro \
  -v $D/patches/triton_unified_attention.py:/usr/local/lib/python3.10/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py:ro \
  -v $D/tritoncache:/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=$GPUS -e VLLM_PLUGINS=hcu \
  -e VLLM_NCCL_SO_PATH=/opt/dtk/lib/librccl.so.1 -e VLLM_DISABLE_PYNCCL=0 \
  $TOPO_ENV -e NCCL_DEBUG=WARN -e ALLREDUCE_STREAM_WITH_COMPUTE=1 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706 \
  -c "python3 /w/patches/patch_gdn_cache.py && exec python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.8-27B --served-model-name qwen38 \
  --trust-remote-code --dtype bfloat16 --tensor-parallel-size $TP \
  --max-model-len $MAXLEN --max-num-seqs $MAXSEQ --gpu-memory-utilization $GPUUTIL \
  --max-num-batched-tokens 8192 \
  --compilation-config '$CC' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --hf-overrides '$YARN' \
  --enable-prefix-caching $MAMBA_ARG --block-size 1024 \
  --host 0.0.0.0 --port $PORT"
echo "$NAME started on port $PORT"
