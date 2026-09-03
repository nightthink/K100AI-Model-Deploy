#!/bin/bash
# @image      qwen38-k100ai-int8:v1.3.1
# @weights    z-lab/Qwen3.8-27B-DFlash2 Qwen3.8-27B-DFlash2
# @weights-ms hygon/Qwen3.8-27B-Channel-INT8-w8a8 Qwen3.8-27B-Channel-INT8-w8a8-hygon
# @name  q38-int8D
# @port  8112
# @gpus  0,1,2,3
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon $MODELS_ROOT/Qwen3.8-27B-DFlash2
# @attest 上下文长度:context_length
# @attest 投机解码DFLASH:speculative
# ============================================================================
# 12 · DocPang v1.3.1 成品镜像整包线 —— 深前缀 prefill 冠军档（2026-09-03 立编）
#
# 与 11 的分工：decode 各段与 11 打平；本线独有两点（源自其 v1.3.1 私有 sglang 源码层，
# 不可拆件移植——wheel 单移/wheel+v30链 已实验证伪）：
#   ★ 40-254K 段冷 prefill 快 1.6×（120K TTFT 59s vs 11 的 85-102s）→ 冷重建段收益
#   ★ 350K 深段 decode 34.5 vs 11 的 ~25（+40%）；350K@ctx458752 实测连贯（TTFT 662s）
#
# ⚠ 镜像是社区成品（DocPang，网盘分发，33GB，不可逐层审计）：
#   S2 无法自动拉取。离线兜底：夸克网盘 Qwen3.8-K100AI-v1.3.1-final-image.tar.zst
#   （SHA256 4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f）
#   → zstd -dc | docker load。101 上已 load。
# ⚠ 权重用 hygon 官方 Channel-INT8（勿用 Freaksterz——我方拷贝已证损坏，accept 归零）。
# ⚠ 其验证边界 262144；ctx 458752 为我方扩测（350K 实测过，带树警告属预期）。
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8112}"; NAME="${NAME:-q38-int8D}"
CTX="${CTX:-458752}"
CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
TARGET="$MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon"
DRAFT_SRC="$MODELS_ROOT/Qwen3.8-27B-DFlash2"
DRAFT="$MODELS_ROOT/Qwen3.8-27B-DFlash2-v2"
IMG="qwen38-k100ai-int8:v1.3.1"
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

case "$GPUS" in
  0,1,2,3|4,5,6,7) ;;
  *) echo "✗ 只允许同 socket 四卡组（本线 AR/P2P 开启）"; exit 1;;
esac
$DOCKER image inspect "$IMG" >/dev/null 2>&1 || {
  echo "✗ 本地无镜像 $IMG —— 本线为离线整包线，S2 不能自动拉取。"
  echo "  从夸克网盘取 Qwen3.8-K100AI-v1.3.1-final-image.tar.zst（校验 SHA256 见头注）"
  echo "  然后: zstd -dc <包> | docker load"
  exit 1
}

# ── draft 派生目录（同 09/11）──
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

# render 节点按 GPUS 推导（本机 PCI 序：卡0-7 ↔ renderD128-135）
RD=()
IFS=, read -ra GG <<<"$GPUS"
for g in "${GG[@]}"; do RD+=(--device "/dev/dri/renderD$((128+g)):/dev/dri/renderD$((128+g))"); done

echo "12-v1.3.1整包线 $NAME: GPU=$GPUS PORT=$PORT ctx=$CTX"
$DOCKER run -d --name "$NAME" \
  --network host --ipc host --security-opt label=disable \
  --device /dev/kfd:/dev/kfd "${RD[@]}" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET":/models/target:ro \
  -v "$DRAFT":/models/draft:ro \
  -e PROFILE=tp4 -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PORT="$PORT" -e SERVED_MODEL_NAME=qwen38 \
  -e CONTEXT_LENGTH="$CTX" -e MEM_FRACTION_STATIC="${MEM_FRAC:-0.95}" \
  -e MAX_RUNNING_REQUESTS=8 -e PP_MAX_MICRO_BATCH_SIZE=8 \
  -e MAX_TOTAL_TOKENS=1048576 -e CHUNKED_PREFILL_SIZE=16384 -e MAX_PREFILL_TOKENS=16384 \
  -e MAMBA_TRACK_INTERVAL=16384 -e MAX_MAMBA_CACHE_SIZE=32 \
  -e SPECULATIVE_NUM_DRAFT_TOKENS=8 -e SGLANG_EMPTY_CACHE_INTERVAL=60 \
  -e TOOL_CALL_PARSER=qwen3_coder -e REASONING_PARSER=qwen3 \
  "$IMG" >/dev/null

echo "$NAME started on $PORT（就绪看 /model_info；含预监听 warmup，就绪较慢）"
