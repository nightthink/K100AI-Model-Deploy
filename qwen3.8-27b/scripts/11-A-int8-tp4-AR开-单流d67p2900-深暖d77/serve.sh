#!/bin/bash
# @image      harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828
# @weights    z-lab/Qwen3.8-27B-DFlash2 Qwen3.8-27B-DFlash2
# @weights-ms hygon/Qwen3.8-27B-Channel-INT8-w8a8 Qwen3.8-27B-Channel-INT8-w8a8-hygon
# @name  q38-int8C
# @port  8111
# @gpus  0,1,2,3
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon $MODELS_ROOT/Qwen3.8-27B-DFlash2 $CFGDIR/minichain5/sitecustomize.py
# @attest 上下文长度:context_length
# @attest 投机解码DFLASH:speculative
# ============================================================================
# 11 · INT8-W8A8 + DFlash2 + custom-AR —— 深上下文冠军档（2026-09-02 定案）
#
# 与 09 的本质差异（DocPang v30 参数体系吸收，dp30 系列 A/B 定界）：
#   ★ custom all-reduce 开启（同 socket 四卡）：主引擎，短 +40% / 120K 暖 +64%。
#     当年 E1 禁 AR 是毒链时代的冤案——干净链上 AR 是纯增益。
#   ★ DocPang chat 模板 + pack min-q 2048 + mem 0.95 + graphs 1-8 + mamba 32
#     + max-total-tokens 1M + mamba-track-interval 16384。
#   实测（验证机B 卡0-3 弱卡组，LRU 代码类探针）：
#     单流 decode 短 58-67 / 16K 62 / 64K 63 / 120K 暖 53-77（瞬时 108，accept 波动 0.26-0.81）
#     对照 09 同类口径：128K 暖 52.3 → +47%。
#   链 = minichain5（repair + raw-q8 + q16k）。v30 完整链与 minichain5 持平（已 A/B），
#   取简者。v122 三件套照 09 惯例挂载（temperature>0 防打死）。
#
# 边界（承 09 + 新增）：
#   - 卡组必须同 socket（0,1,2,3 或 4,5,6,7）——AR/P2P 跨 socket 会踩 52ms 活锁
#   - 就绪探测只能用 /model_info 或 /v1/models（/health 会注入生成请求）
#   - bs16×投机=崩 边界仍在；graphs 到 8 已验证
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8111}"; NAME="${NAME:-q38-int8C}"
MEM_FRAC="${MEM_FRAC:-0.95}"
D="${WORK:-/data/q38-work}"
CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
TARGET="$MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon"
DRAFT_SRC="$MODELS_ROOT/Qwen3.8-27B-DFlash2"
DRAFT="$MODELS_ROOT/Qwen3.8-27B-DFlash2-v2"
mkdir -p "$D/tritoncache-int8" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

case "$GPUS" in
  0,1,2,3|4,5,6,7) ;;
  *) echo "✗ 只允许同 socket 四卡组（0,1,2,3 或 4,5,6,7）——本线 AR/P2P 开启"; exit 1;;
esac

# ── draft 派生目录（同 09）──
if [ ! -f "$DRAFT/config.json" ]; then
  mkdir -p "$DRAFT" 2>/dev/null || { sudo mkdir -p "$DRAFT" && sudo chown "$(id -u):$(id -g)" "$DRAFT"; }
  SRC_CFG="$DRAFT_SRC/config.json"
  [ -f "$DRAFT_SRC/config.json.orig_hf" ] && SRC_CFG="$DRAFT_SRC/config.json.orig_hf"
  cp "$SRC_CFG" "$DRAFT/config.json"
  ln -f "$DRAFT_SRC/model.safetensors" "$DRAFT/model.safetensors" 2>/dev/null \
    || cp "$DRAFT_SRC/model.safetensors" "$DRAFT/model.safetensors"
fi
grep -q "DFlash2DraftModel" "$DRAFT/config.json" \
  || { echo "✗ draft config 不是 DFlash2DraftModel"; exit 1; }

echo "11-INT8深上下文冠军档 $NAME: GPU=$GPUS PORT=$PORT mem=$MEM_FRAC (AR开)"
$DOCKER run -d --name "$NAME" \
  --network host --ipc host --privileged --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$D/tritoncache-int8":/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET":/models/target:ro \
  -v "$DRAFT":/models/draft:ro \
  -v "$CFGDIR/minichain5":/minichain4:ro \
  -v "$CFGDIR/qwen38_chat_template.jinja":/qwen38_chat_template.jinja:ro \
  -v "$CFGDIR/v122/dflash_worker.py":/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_worker.py:ro \
  -v "$CFGDIR/v122/dflash_info.py":/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_info.py:ro \
  -v "$CFGDIR/v122/dflash_utils.py":/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_utils.py:ro \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e W8A8_SUPPORT_METHODS=1 \
  -e TRITON_JSON_DIR=/data/qwen38-27b-k100ai-int8-opt/cache/tp4 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_KV_LAYOUT_DCU_FA=true -e SGLANG_USE_LIGHTOP=0 \
  -e SGLANG_USE_CAUSAL_CONV1D=0 -e SGLANG_USE_TRITON_VLLM_FA=0 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e SGLANG_Q38_TP4_Q16_KV_LENGTHS=16384,32768,49152,65536,81920,98304,114688,131072,147456,163840,180224,196608,212992,229376,245760 \
  -e SGLANG_Q38_TP4_Q16_SPLIT_KV=4 -e SGLANG_Q38_TP4_Q16_QSPLIT2=1 \
  -e SGLANG_Q38_TP4_Q16_QSPLIT_KV_EXACT=131072 \
  -e SGLANG_Q38_TP4_TAIL_SPLIT_8K=1 -e SGLANG_Q38_TP4_TAIL_SPLIT_MIN_PREFIX=131072 \
  -e SGLANG_Q38_TP4_LONG_CHUNK_8K_PREFIX=131072 \
  -e SGLANG_Q38_TP4_QTAIL_KV_LENGTHS=257900 -e SGLANG_Q38_TP4_QTAIL_SPLIT_KV=8 \
  -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  -e PYTHONPATH=/minichain4 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828 -c "
  export PYTHONPATH=/minichain4
  python3 -m sglang.launch_server \
    --model-path /models/target --served-model-name qwen38 \
    --chat-template /qwen38_chat_template.jinja \
    --trust-remote-code --tp 4 --page-size 64 \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --attention-backend fa3 --mm-attention-backend fa3 \
    --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 32 \
    --mamba-track-interval 16384 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 --disable-piecewise-cuda-graph \
    --mem-fraction-static $MEM_FRAC \
    --context-length 1048576 \
    --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 2048 --pack-paged-kv-to-varlen-min-kv-tokens 2048 \
    --max-total-tokens 1048576 --max-running-requests 8 --pp-max-micro-batch-size 8 \
    --speculative-algorithm DFLASH --speculative-draft-model-path /models/draft \
    --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton \
    --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
    --watchdog-timeout 7200 --dist-timeout 7200 --skip-server-warmup \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port $PORT" >/dev/null

echo "$NAME started on $PORT（就绪看 /v1/models；首个请求兼暖场）"
