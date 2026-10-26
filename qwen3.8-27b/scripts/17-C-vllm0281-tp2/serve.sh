#!/bin/bash
# @image harbor.sourcefind.cn:5443/hcu/admin/base/custom@sha256:2eff683dd747db3484a1b523c05e49d75bb45ed6724b4d1238344385c7e88c9d
# @weights Qwen/Qwen3.8-27B Qwen3.8-27B
# @name q38-vllm0281
# @port 8117
# @gpus 0,1
# @expect-gpu 2
# @requires $MODELS_ROOT/Qwen3.8-27B $CFGDIR/runtime_check.py
# @attest 发行版本自证:Q38_RUNTIME_VERSION=
# @attest HCU平台自证:Q38_HCU_PLATFORM=
# @attest Triton注意力:TRITON_ATTN
set -euo pipefail

CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK="${WORK:-$(dirname "$CFGDIR")}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
M="$MODELS_ROOT/Qwen3.8-27B"
NAME="${NAME:-q38-vllm0281}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8117}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
MEM_FRAC="${MEM_FRAC:-0.90}"
ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-/opt/dtk/lib/rocblas/library_gpu4}"
IMAGE="harbor.sourcefind.cn:5443/hcu/admin/base/custom@sha256:2eff683dd747db3484a1b523c05e49d75bb45ed6724b4d1238344385c7e88c9d"

if [[ ! "$GPUS" =~ ^([0-9]+),([0-9]+)$ ]] || [[ "${BASH_REMATCH[1]:-}" = "${BASH_REMATCH[2]:-}" ]]; then
  echo "✗ 17线仅支持两张不同GPU的双卡列表，例如 GPUS=0,1" >&2
  exit 1
fi
: "${LAUNCH_ID:?请通过打包后的 up.sh start 启动17线}"
PKGV="$(awk '{print $2}' "$WORK/VERSION")"
[ -n "$PKGV" ] || { echo "✗ 缺少拉起包版本身份" >&2; exit 1; }
if timeout 15 docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo -n docker); fi
CACHE="$WORK/cache-$NAME"
mkdir -p "$CACHE/triton" "$CACHE/inductor" "$CACHE/vllm"

# 标签沿用公共Worker合同；不在配置脚本里按名称删除或重建其它实例。
"${DOCKER[@]}" run -d --pull=never --restart=no --name "$NAME" \
  --label line=17 --label publisher=01ma --label pkg="$PKGV" --label package_id="$PKGV" \
  --label launch_id="$LAUNCH_ID" --label launch_mode=docker --label gpus="$GPUS" --label port="$PORT" \
  --label model_id="$(printf '%s' "$(realpath -m "$M")" | sha256sum | cut -c1-16)" \
  --network host --shm-size 32g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$M":/models/target:ro -v "$CFGDIR":/line:ro -v "$CACHE":/cache \
  -e HIP_VISIBLE_DEVICES="$GPUS" -e PYTHONNOUSERSITE=1 \
  -e ROCBLAS_TENSILE_LIBPATH="$ROCBLAS_TENSILE_LIBPATH" \
  -e VLLM_HCU_USE_CUSTOM_OPS=0 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e VLLM_CACHE_ROOT=/cache/vllm \
  --entrypoint python3 "$IMAGE" /line/runtime_check.py /models/target \
  --served-model-name qwen38 --trust-remote-code --language-model-only \
  --tensor-parallel-size 2 --dtype bfloat16 --safetensors-load-strategy prefetch \
  --attention-backend TRITON_ATTN --block-size 64 --enable-prefix-caching --mamba-cache-mode align \
  --prefix-match-unit 64 --enable-mamba-fine-grained-prefix-cache \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len "$CTX" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$MEM_FRAC" --host "$HOST" --port "$PORT"

echo "17线已派发：$NAME GPU=$GPUS PORT=$PORT；就绪由公共Worker核验"
