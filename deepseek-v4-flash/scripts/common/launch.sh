#!/bin/bash
# ============================================================================
# 统一启动器 —— 按 docs/方案脚本规范-设计文档.md 走完 S3→S8（可选 S9）
#
#   bash common/launch.sh A-sglang-tp8-hybrid
#   bash common/launch.sh A-sglang-tp4-tuned GPUS=0,1,2,3 PORT=8101 NAME=q38-sgA
#   TEARDOWN=1 bash common/launch.sh <配置名>     # 冒烟：跑通即收尾
#   SKIP_GATE=1 / IGNORE_ATTEST=1                 # 仅排障用，不可用于测量
#
# 目录约定（每条线的每个配置一个子目录，自给自足）：
#   lib/           载体适配 + 阶段契约      —— 共用
#   common/        跨配置工具（本文件、路由、NUMA、ACS 体检）—— 共用
#   <编号>-<线路>-<引擎>-<拓扑>/     （编号目录直接在 scripts/ 下）
#       serve.sh   ★ 该配置的 S4+S6，**只管这两件**
#       README.md  该配置的依据、参数理由、实测数据、已知限制
#
# serve.sh 顶部用 `# @xxx` 声明元数据：
#   @name @port @gpus @expect-gpu @requires @attest
#
# 传给 serve.sh 的标准变量：WORK（工作目录根）COMMON CFGDIR
# ============================================================================
set -u
COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(dirname "$COMMON")"          # scripts/ 或远端的 /data/q38-work
. "$WORK/lib/stages.sh"

CFG="${1:?用法: launch.sh <配置名> [KEY=VAL ...]}"; shift
CFGDIR="$WORK/$CFG"        # 编号目录直接在 scripts/ 下
SERVE="$CFGDIR/serve.sh"
[ -f "$SERVE" ] || { echo "找不到 $SERVE"; echo "可用配置："; ls -1 "$WORK/configs" | grep -v '^_' | sed 's/^/  /'; exit 1; }
for kv in "$@"; do export "$kv"; done
export WORK COMMON CFGDIR

meta(){ grep -E "^# @$1[ \t]" "$SERVE" | sed -E "s/^# @$1[ \t]+//"; }
NAME="${NAME:-$(meta name | head -1)}"
PORT="${PORT:-$(meta port | head -1)}"
GPUS="${GPUS:-$(meta gpus | head -1)}"
EXPECT_GPU="$(meta expect-gpu | head -1)"; EXPECT_GPU="${EXPECT_GPU:-8}"
export MODELS_ROOT="${MODELS_ROOT:-/data/models}"
# ── S2 · 宿主环境自举：镜像/权重/1M 农场，缺则补、有则跳过（SKIP_S2=1 跳过）──
if [ "${SKIP_S2:-0}" != 1 ]; then
  bash "$COMMON/ensure_ready.sh" "$SERVE" || { echo "S2 未通过，拒绝启动（SKIP_S2=1 可跳过自举）"; exit 1; }
fi
# @requires 里的 $WORK / $MODELS_ROOT 在此展开（serve.sh 用变量写，包才目录中立）
mapfile -t _REQ_RAW < <(meta requires | tr ' ' '\n' | grep -v '^$')
REQ=(); for _r in "${_REQ_RAW[@]}"; do REQ+=("$(eval echo "$_r")"); done
mapfile -t ATTEST < <(meta attest)

echo "════ 配置 $CFG ════"
echo "  实例=$NAME 端口=$PORT 卡组=${GPUS:-未声明} 期望卡数=$EXPECT_GPU 载体=$CARRIER"

if [ "${SKIP_GATE:-0}" = "1" ]; then _say "S3 已跳过（SKIP_GATE=1，数据不可用于结论）"
else stage3_gate "$GPUS" "$EXPECT_GPU" "${REQ[@]}" || exit 1; fi

stage5_cleanup "$NAME" || exit 1

_say "S4+S6 拉起：$CFG/serve.sh"
bash "$SERVE" || { _say "✗ 启动命令返回非零"; exit 1; }

stage7_wait_ready "$NAME" "http://127.0.0.1:$PORT/v1/models" \
                  "${READY_TIMEOUT:-3900}" "${MAX_RESTARTS:-2}" \
  || { _say "S9 失败收尾"; stage9_teardown "$NAME"; exit 1; }

if ! stage8_attest "$NAME" "${ATTEST[@]}"; then
  _say "★ S8 未通过 —— 服务起来了，但跑的不是声明的配置。"
  _say "  此时测出的数据**不可用于结论**（LD_PRELOAD 失败是静默的）。"
  [ "${IGNORE_ATTEST:-0}" = "1" ] || { stage9_teardown "$NAME"; exit 2; }
  _say "  IGNORE_ATTEST=1，继续。"
fi

echo "════ $NAME 就绪且自证通过，端口 $PORT ════"
[ "${TEARDOWN:-0}" = "1" ] && stage9_teardown "$NAME"
exit 0
