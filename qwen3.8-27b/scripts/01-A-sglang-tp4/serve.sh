#!/bin/bash
# @image      harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828
# @name  q38-sgA
# @port  8101
# @gpus  0,1,2,3
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-1M $MODELS_ROOT/Qwen3.8-27B $CFGDIR/minichain5n/sitecustomize.py
# @attest 上下文长度:context_length
# @attest 投机解码NEXTN:speculative
# ============================================================================
# 01 · bf16 + NEXTN · TP4 —— 全精度主线 v2（2026-09-03 迁树升级）
#
# v2 变化（11 号线优化迁移实验定稿，docs/2026-09-02-dp30攻坚 续章）：
#   ★ 底座 0811 树 → 0828 树；custom-AR 开启（0811 树 aiter AR 会 SIGSEGV，
#     0828 树实现已修——11 号线验证）；不再需要 dlhook2-sg 预载。
#   ★ minichain5n：gfx928 varlen 修复 + q<=4 直走原生（NEXTN verify 必经；
#     若走 triton 修复路径，120K decode 塌到 1.8——本次定界）。
#   ★ NEXTN steps 2 / draft 3（draft4 在本树深上下文塌方，勿回调）。
#   实测（卡0-3）：短代码 31.5-32.8（原 20.7-25.5）/ 长文 26.2（原 19.2）
#               / 120K 26.5 冷 29.5 暖（原深段 ~15-16）—— 全面 +30~60%。
#
# 边界：
#   - 1M 农场为符号链接目录：必须同时挂 $MODELS_ROOT 根（链接目标在其下）
#   - 0828 树按 VL 家族读模型：挂 overrides/ 两个 preprocessor json
#   - 1M 农场 tokenizer 较旧缺 think 结束符：挂 hygon INT8 目录的 tokenizer 两件（同词表）
#   - 就绪探测用 /model_info（/health 注入生成请求）
#   - ⚠ 02 号线内嵌本配置：本次升级后 02 需重新验收方可发布
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8101}"; NAME="${NAME:-q38-sgA}"
MEM_FRAC="${MEM_FRAC:-0.90}"
D="${WORK:-/data/q38-work}"
CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
M="$MODELS_ROOT/Qwen3.8-27B-1M"
TOK="$MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon"
mkdir -p "$D/tritoncache-int8" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

case "$GPUS" in
  0,1,2,3|4,5,6,7) ;;
  *) echo "✗ 只允许同 socket 四卡组（本线 AR/IPC 开启）"; exit 1;;
esac
TOKMOUNTS=()
if [ -f "$TOK/tokenizer.json" ]; then
  TOKMOUNTS=(-v "$TOK/tokenizer.json":/models/target/tokenizer.json:ro
             -v "$TOK/tokenizer_config.json":/models/target/tokenizer_config.json:ro)
else
  echo "⚠ 未找到 hygon tokenizer（$TOK），沿用农场自带——若 think_end 报错请补齐该目录"
fi

echo "01v2-bf16全精度主线 $NAME: GPU=$GPUS PORT=$PORT mem=$MEM_FRAC (AR开·0828树)"
$DOCKER run -d --name "$NAME" \
  --network host --ipc host --privileged --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$D/tritoncache-int8":/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELS_ROOT":"$MODELS_ROOT":ro \
  -v "$M":/models/target:ro \
  -v "$CFGDIR/overrides/preprocessor_config.json":/models/target/preprocessor_config.json:ro \
  -v "$CFGDIR/overrides/video_preprocessor_config.json":/models/target/video_preprocessor_config.json:ro \
  "${TOKMOUNTS[@]}" \
  -v "$CFGDIR/minichain5n":/minichain4:ro \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_KV_LAYOUT_DCU_FA=true -e SGLANG_USE_LIGHTOP=0 \
  -e SGLANG_USE_CAUSAL_CONV1D=0 -e SGLANG_USE_TRITON_VLLM_FA=0 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  -e PYTHONPATH=/minichain4 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828 -c "
  export PYTHONPATH=/minichain4
  python3 -m sglang.launch_server \
    --model-path /models/target --served-model-name qwen38 \
    --trust-remote-code --tp 4 --page-size 64 \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --attention-backend fa3 --mm-attention-backend fa3 \
    --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 --disable-piecewise-cuda-graph \
    --mem-fraction-static $MEM_FRAC \
    --context-length 1000000 \
    --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 2048 --pack-paged-kv-to-varlen-min-kv-tokens 2048 \
    --speculative-algorithm NEXTN --speculative-num-steps 2 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
    --watchdog-timeout 7200 --dist-timeout 7200 --skip-server-warmup \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port $PORT" >/dev/null

echo "$NAME started on $PORT（就绪看 /model_info；首个请求兼暖场）"
