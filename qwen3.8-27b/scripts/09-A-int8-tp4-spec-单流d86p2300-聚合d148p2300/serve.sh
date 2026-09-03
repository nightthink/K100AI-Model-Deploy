#!/bin/bash
# @image      harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828
# @weights    z-lab/Qwen3.8-27B-DFlash2 Qwen3.8-27B-DFlash2
# @weights-ms hygon/Qwen3.8-27B-Channel-INT8-w8a8 Qwen3.8-27B-Channel-INT8-w8a8-hygon
# @name  q38-int8A
# @port  8109
# @gpus  0,1,2,3
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon $MODELS_ROOT/Qwen3.8-27B-DFlash2 $CFGDIR/minichain5/sitecustomize.py
# @attest 上下文长度:context_length
# @attest 投机解码DFLASH:speculative
# ============================================================================
# 09 · INT8-W8A8 + DFlash2 投机 · TP4 —— 低延迟档（2026-08-29 定案）
#
# 实测（验证机B 卡0-3，scan3 口径）：
#   单流 decode 85.9 tok/s（代码，瞬时 105-113）/ 47.0（essay）
#   prefill ~2300 tok/s（96K 输入 TTFT 43.1s，暖服务器）
#   聚合 4路 133.6 / 8路 148.3 tok/s；1M/请求可达（decode@996K = 10.3）
#
# 原理：纯 0828 树 + 官方 INT8 + 原生 DFLASH + minichain5 迷你补丁链
#   （gfx928 varlen 修复 + raw-q8 verifier + q16k 长 prefill 桶）+ 厂商 GEMM 调优 JSON。
#   ★ 2026-09-01 minichain4→5：q=8 verify 由 2×q4 改为单次 raw paged_attention
#     （DocPang v1.2.1 吸收，flash-attn 260602 legacy ABI 版本锁，几何不符自动回退）。
#     A/B 实测（卡4-7）：SHA 八位点全等；decode 16K +2.4% / 64K +7.8% / 128K +12.2% / 245K +26.5%。
#   ★ v122/ 三文件挂载（DocPang v1.2.2 吸收）：修复 temperature>0 / 默认采样
#     一发即打死 API 服务的缺陷（实测 3A 复现、3B 修复、greedy SHA 回归全等）。
#   ★ 切勿挂厂商完整 runtime_patch 链（TP-gather 钩子会让 tp>1 每步分钟级，
#     见 docs/2026-08-29-TP4攻克-病理定罪与外科配方.md）。
#
# 边界（均已实证）：
#   - cuda-graph 只到 bs8：bs16×投机=调度器崩溃
#   - 并发甜点 8 路（mamba extra_buffer 每请求约 5 槽）
#   - 就绪探测只能用 /model_info 或 /v1/models，禁用 /health（会注入生成请求）
#   - 测 TTFT 前先发小请求暖场（慢速 tokenizer 首载几分钟）
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8109}"; NAME="${NAME:-q38-int8A}"
MEM_FRAC="${MEM_FRAC:-0.85}"
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
  *) echo "✗ 只允许同 socket 四卡组（0,1,2,3 或 4,5,6,7）"; exit 1;;
esac

# ── draft 派生目录：本镜像的树要求原版架构名 DFlash2DraftModel ──
# （历史遗留：旧实验曾把 config.json 改名为 DFlashDraftModel 以适配 0811 v1 树）
if [ ! -f "$DRAFT/config.json" ]; then
  mkdir -p "$DRAFT" 2>/dev/null || { sudo mkdir -p "$DRAFT" && sudo chown "$(id -u):$(id -g)" "$DRAFT"; }
  SRC_CFG="$DRAFT_SRC/config.json"
  [ -f "$DRAFT_SRC/config.json.orig_hf" ] && SRC_CFG="$DRAFT_SRC/config.json.orig_hf"
  cp "$SRC_CFG" "$DRAFT/config.json"
  ln -f "$DRAFT_SRC/model.safetensors" "$DRAFT/model.safetensors" 2>/dev/null \
    || cp "$DRAFT_SRC/model.safetensors" "$DRAFT/model.safetensors"
fi
grep -q "DFlash2DraftModel" "$DRAFT/config.json" \
  || { echo "✗ draft config 不是 DFlash2DraftModel（本树必须用原版架构名）"; exit 1; }

echo "09-INT8低延迟档 $NAME: GPU=$GPUS PORT=$PORT mem=$MEM_FRAC"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" \
  --network host --privileged --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET":/models/target:ro \
  -v "$DRAFT":/models/draft:ro \
  -v "$CFGDIR/minichain5":/minichain4:ro \
  -v "$CFGDIR/v122/dflash_worker.py":/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_worker.py:ro \
  -v "$CFGDIR/v122/dflash_info.py":/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_info.py:ro \
  -v "$CFGDIR/v122/dflash_utils.py":/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_utils.py:ro \
  -v "$D/tritoncache-int8":/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e W8A8_SUPPORT_METHODS=1 \
  -e TRITON_JSON_DIR=/data/qwen38-27b-k100ai-int8-opt/cache/tp4 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_KV_LAYOUT_DCU_FA=true -e SGLANG_USE_LIGHTOP=0 \
  -e SGLANG_USE_CAUSAL_CONV1D=0 -e SGLANG_USE_TRITON_VLLM_FA=0 \
  -e SGLANG_Q38_TP4_Q16_KV_LENGTHS=16384,32768,49152,65536,81920,98304,114688,131072,147456,163840,180224,196608,212992,229376,245760 \
  -e SGLANG_Q38_TP4_Q16_SPLIT_KV=4 -e SGLANG_Q38_TP4_Q16_QSPLIT2=1 \
  -e SGLANG_Q38_TP4_Q16_QSPLIT_KV_EXACT=131072 \
  -e SGLANG_Q38_TP4_TAIL_SPLIT_8K=0 -e SGLANG_Q38_TP4_LONG_CHUNK_8K_PREFIX=0 \
  -e SGLANG_Q38_TP4_QTAIL_SPLIT_KV=0 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828 -c "
source /opt/dtk/env.sh
export PYTHONPATH=/minichain4
exec python3 -m sglang.launch_server \
  --model-path /models/target --served-model-name qwen38 \
  --trust-remote-code --tp 4 --page-size 64 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-backend fa3 --mm-attention-backend fa3 \
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
  --cuda-graph-bs 1 2 4 8 --disable-piecewise-cuda-graph \
  --mem-fraction-static $MEM_FRAC --disable-custom-all-reduce \
  --context-length 1048576 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 4096 --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /models/draft \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
  --watchdog-timeout 7200 --dist-timeout 7200 --skip-server-warmup \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port $PORT"
echo "$NAME started on $PORT（就绪看 /v1/models；首个请求兼暖场）"
