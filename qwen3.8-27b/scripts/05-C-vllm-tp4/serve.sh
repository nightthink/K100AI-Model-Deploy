#!/bin/bash
# @image   harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @name  q38-final
# @port  8100
# @gpus  4,5,6,7
# GDN all 模式 Phase1 验证：无投机、无SP、enforce-eager、256K、mamba-block-size 8192
set -e
NAME="${NAME:-q38-final}"
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
# ★ 起服务前先体检 ACS：本配置启用了 custom all-reduce，ACS 若没清干净
#   会以约 0.05 tok/s 起来且无任何报错（详见 docs/ACS-P2P解锁.md）
bash "${COMMON:-$W/common}/preflight_acs.sh" 4,5,6,7 || exit 1

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --restart unless-stopped --name "$NAME" \
  --network host --privileged --cpuset-cpus=48-59 --cpuset-mems=4-7 --shm-size 256g \
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
  -v $W/patches/triton_unified_attention.py:/usr/local/lib/python3.10/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py:ro \
  -v $W/tritoncache:/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 -e VLLM_PLUGINS=hcu \
  -e VLLM_NCCL_SO_PATH=/opt/dtk/lib/librccl.so.1 -e VLLM_DISABLE_PYNCCL=0 \
  -e NCCL_P2P_LEVEL=PHB -e NCCL_TOPO_FILE=/w/rccl-topo-4card-fixed.xml -e NCCL_DEBUG=WARN -e ALLREDUCE_STREAM_WITH_COMPUTE=1 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e VLLM_HCU_USE_FLASH_ATTN_UNIFIED=${FA_UNIFIED:-0} \
  -e VLLM_HCU_MAMBA_SSM_CACHE_DTYPE=${SSM_DTYPE:-0} \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706 \
  -c "python3 /w/patches/patch_gdn_cache.py && exec python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.8-27B --served-model-name qwen38 \
  --trust-remote-code --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 524288 --max-num-seqs 16 --gpu-memory-utilization 0.90 \
--max-num-batched-tokens 8192 \
  --compilation-config '$CC' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --hf-overrides '$YARN' \
  --enable-prefix-caching --mamba-cache-mode all --mamba-block-size 2048 --block-size 1024 \
  --host 0.0.0.0 --port 8100"
echo "q38-vgdn (phase1) started"
