#!/bin/bash
# ============================================================================
# ★ 拉起包统一入口（固定名称 up.sh）—— 放在包根目录，目录中立
#
#   ⚠ `bash up.sh`（无参数）的行为**取决于包形态**，两者不同：
#     · 一线一包（包根有 .primary，即现场拿到的所有 q38-kit-NN 包）
#         → **直接拉起该线**，等同 `bash up.sh NN`。不是预览！
#     · 多配置树（仓库侧 deploy-kit/scripts/，无 .primary）
#         → 只打印自检 + 配置一览 + 推荐，不拉起。
#     要在一线一包里"只看不拉"，用 `bash up.sh status`。
#
#   bash up.sh                        # 见上：一线一包=拉起；多配置树=一览
#   bash up.sh 09                     # 拉起 09 号线（体检→S1→S2..S8→真实请求冒烟）
#   bash up.sh 10 GPUS=4,5,6,7        # 带参数拉起（KEY=VAL 透传给 launch.sh）
#   bash up.sh 09 status              # 观察该线状态
#   bash up.sh status                 # 观察全部
#   bash up.sh 09 stop                # 停止该线全部实例（S9：容器清场）
#   bash up.sh stop                   # 停止全部并报告 KFD 残留
#   bash up.sh stop PORT=8102         # ★ 定点停一个实例，不触碰同线其它实例
#   bash up.sh restart PORT=8102      # ★ 定点重启（参数从容器实况还原，无需重报卡与端口）
#   bash up.sh status PORT=8102       # ★ 只看该实例（含其拉起阶段与容器日志尾）
#   bash up.sh logs PORT=8102         # ★ 只看该实例的容器日志
#
# ★ 标准性能测试（全部线同一口径，换线只改端口；产出四指标矩阵）：
#   bash up.sh benchmark              # 测在跑实例；未指定端口则自动取
#   bash up.sh benchmark PORT=8102 CYCLES=2 OUT=/tmp/perf01
#   负载：输入 45K–350K（均值160K）／输出 500–4K（均值800）／命中率均值95%最小85%；
#         350K 后切 ~70K 全新前缀。产出 TTFT / prefill速率 / decode速率 / E2E 矩阵。
#
# ★ 同机多实例（同一个包，只改参数，不改任何文件）：
#     bash up.sh GPUS=0,1,2,3 PORT=8101      # 实例一（沿用 @name）
#     bash up.sh GPUS=4,5,6,7 PORT=8102      # 实例二（名字自动派生为 …-8102）
#   若模型只占 2 卡，同法可在 8 卡机上起 4 个。未给 NAME 时按端口派生实例名；
#   Triton 缓存亦按实例隔离（$D/tritoncache-int8-$NAME），互不干扰。
#
#   bash up.sh logs [-f]              # 看本线容器的 docker 日志（-f 跟随）
#
#   环境变量：MODELS_ROOT=/data/models（权重根）  S1_SKIP=1（跳过机器准备）
#
# ★ 标记体系：本流程创建/设置的一切运行时产物统一带前缀 01ma.cli-model-svc-
#   （容器名、systemd 单元、/usr/local/sbin 脚本、sysctl 片段）。
#   stop 按该前缀清理 —— 不管前序 up 跑到哪一步、成功还是失败，都能完整停止。
# ★ 幂等：up 可任意重复执行（S1 幂等安装；拉起前先按标记清掉本线旧实例再起新）。
# ★ 可观察：全程输出落 logs/<线>-up-<时刻>.log；status 显示阶段进展 + 实例表 +
#   容器日志尾；实时跟随用 `tail -f logs/latest-up.log` 或 `bash up.sh logs -f`。
#
# 自包含契约：除 镜像 与 模型权重（S2 自动拉取）外，包内包含运行所需的一切。
# 十步契约：S1 本脚本；S2-S8 由 common/launch.sh 执行；stop=S9 清场。
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export WORK="$HERE"
export MODELS_ROOT="${MODELS_ROOT:-/data/models}"
say(){ echo "══ $* ══"; }
die(){ echo; echo "✗✗ $1"; echo "   下一步：$2"; exit 1; }
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
PFX="01ma.cli-model-svc-"          # ★ 本流程所有运行时产物的识别前缀

# ── 生命周期合同（R24–R34）的纯逻辑模块 ──────────────────────────
#   身份规范化、冲突分类、Resume/端口迁移决策全在 common/contract.sh，
#   它不碰 docker、不读设备前提，故可被 tests/pkg/verify_launch_contract.sh
#   在无 GPU 机器上直接 source 验证。本处只负责加载与调用，不复制其逻辑。
#   ★ 不提供任何「跳过驱动/设备检查」的开关——模块可测是因为它本身不依赖那些前提。
CONTRACT_LOADED=0
# 新四动作只属于 01/13 的一线一包；共用骨架不能改变其它线的旧入口。
_contract_line=$(cut -c1-2 "$HERE/.primary" 2>/dev/null || true)
if [[ "$_contract_line" = 01 || "$_contract_line" = 13 ]] && [ -f "$HERE/common/contract.sh" ]; then
  . "$HERE/common/contract.sh"
  [ -f "$HERE/common/lifecycle.sh" ] && . "$HERE/common/lifecycle.sh"
  CONTRACT_LOADED=1
fi
LAUNCH_MODE="docker"              # R25.4-R25.6：只表达执行机制，01/13 均为 docker

# ★ 真实宿主模型路径（R25.1）——必须是宿主 realpath，不得用 MODELS_ROOT 根，
#   也不得用容器内 /models/target 冒充（那是挂载点，不是身份）。
#   01/13 线的 serve.sh 均以 `M="$MODELS_ROOT/Qwen3.8-27B-1M"` 定义，从中解析。
ct_line_model(){
  local sv="$HERE/$CFGNAME/serve.sh" mr sub
  [ -f "$sv" ] || { echo ""; return 1; }
  mr="${MODELS_ROOT:-/data/models}"
  sub=$(sed -n 's/^M="\$MODELS_ROOT\/\([^"]*\)".*/\1/p' "$sv" | head -1)
  [ -z "$sub" ] && { echo ""; return 1; }
  echo "$mr/$sub"
}
# ★ 退出码语义（供上层调用者判断）：
#     0  服务已就绪          1  失败
#    10  补丁已部署，需重启机器后重跑（内核模块补丁改的是磁盘 .ko，必须重启才生效）
EXIT_REBOOT=10
REBOOT_FLAG="$HERE/.reboot-required"
mkdir -p "$HERE/logs" 2>/dev/null || true

cfg_dirs(){ ls -d "$HERE"/[0-9][0-9]*-* 2>/dev/null; }   # 两位/三位编号通吃
meta(){ grep -E "^# @$2[ \t]" "$1/serve.sh" 2>/dev/null | sed -E "s/^# @$2[ \t]+//" | head -1; }
cfg_name(){  # 编号/目录名 → 目录名
  local d; d=$(ls -d "$HERE/$1"-* 2>/dev/null | head -1)
  [ -z "$d" ] && d=$(ls -d "$HERE/$1" 2>/dev/null | head -1)
  [ -n "$d" ] && basename "$d"
}
cfg_port(){ local d="$HERE/$1"
  if [ "$(echo "$1" | cut -c1-2)" = 02 ]; then echo 8100; else local p; p=$(meta "$d" port); echo "${p:-8101}"; fi; }
cfg_names(){ local d="$HERE/$1"   # 该线管理的容器名列表（带标记前缀）
  if [ "$(echo "$1" | cut -c1-2)" = 02 ]; then echo "${PFX}q38-router ${PFX}q38-a ${PFX}q38-b"
  else local n; n=$(meta "$d" name); [ -n "$n" ] && echo "$PFX$n"; fi; }
cfg_legacy(){ local d="$HERE/$1"  # 迁移期：无前缀旧名（老包/手工拉起的同线实例）
  if [ "$(echo "$1" | cut -c1-2)" = 02 ]; then echo "q38-router q38-a q38-b"; else meta "$d" name; fi; }

probe(){ # port → OK/―
  curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q 200 && echo "在线" || echo "―"; }

stage_progress(){  # 从最近一次 up 日志解析 S1-S10 进展
  local L; L=$(ls -t "$HERE"/logs/*-up-*.log 2>/dev/null | head -1)
  [ -n "$L" ] || { echo "  （尚无拉起日志）"; return; }
  echo "  最近拉起: $(basename "$L")"
  local mk=""
  chk(){ grep -qE "$2" "$L" && mk="$mk $1✓" || mk="$mk $1·"; }
  chk 体检 "✓ docker / 驱动"
  chk S1  "S1 机器准备|S1 已跳过"
  chk S2  "S2 通过"
  chk S3  "S3 通过"
  chk S5  "S5 (清理|无残留)"
  chk S4S6 "S4\+S6 拉起"
  chk S7  "S7 就绪"
  chk S8  "S8 通过"
  chk 冒烟 "模型回复"
  echo "  阶段:$mk"
  grep -qE "✗✗|✗ " "$L" && { echo "  ⚠ 该次运行有失败步骤，末尾摘录："; tail -4 "$L" | sed "s/^/    /"; }
  echo "  （实时跟随: tail -f $L）"
}

# 端口 → 实例名（多实例的统一句柄解析）
_by_port(){
  local want="$1" n p
  for n in $($DOCKER ps -a --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null); do
    p=$($DOCKER inspect "$n" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
    [ "$p" = "$want" ] && { echo "$n"; return 0; }
  done
  return 1
}

do_status(){  # 句柄统一为端口：起服时给的是卡+端口，停/看状态用同一个端口即可。
  local _t=""
  for _a in "$@"; do
    case "$_a" in
      NAME=*) _t="${_a#NAME=}" ;;
      PORT=*) _t=$(_by_port "${_a#PORT=}") ;;
    esac
  done
  if [ -n "$_t" ]; then
    local st rc p c lg
    st=$($DOCKER inspect -f '{{.State.Status}}' "$_t" 2>/dev/null); st=${st:-无}
    rc=$($DOCKER inspect -f '{{.RestartCount}}' "$_t" 2>/dev/null); rc=${rc:--}
    p=$($DOCKER inspect "$_t" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
    [ "$st" = running ] && c=$(probe "${p:-0}") || c="―"
    echo "── 实例 $_t ──"
    printf "  端口 %s   接口 %s   容器 %s   重启 %s\n" "${p:-?}" "$c" "$st" "$rc"
    lg=$(ls -t "$HERE"/logs/*-${p}-up-*.log 2>/dev/null | head -1)
    if [ -n "$lg" ]; then
      echo "  拉起日志 $(basename "$lg")"
      echo "  ── 拉起阶段（S1-S8）──"
      grep -E "S[0-9] |✓ |✗ " "$lg" | tail -14 | sed 's/^/    /'
    fi
    echo "  ── 容器日志尾 ──"
    $DOCKER logs --tail 8 "$_t" 2>&1 | grep -vE "GET /|vllm\._C" | tail -6 | cut -c1-150 | sed 's/^/    /'
    # ★ 重启待办属于**机器级**事实，定点查询同样必须看得到。
    #   2026-09-17 实机验收：它原先只写在下方全量列表路径里，而定点分支在此 return，
    #   于是 `status PORT=` 永远不显示这一栏。
    if [ -f "$REBOOT_FLAG" ]; then
      echo "  ⚠ 重启待办: 是 —— 补丁已部署但机器未重启，服务起不来或补丁未生效"
      sed 's/^/      /' "$REBOOT_FLAG"
    else
      echo "  重启待办: 无"
    fi
    return 0
  fi

  echo "── 阶段进展（S1-S10）──"
  stage_progress
  echo "── 实例表 ──"
  printf "  %-4s %-30s %-6s %-8s %-12s %s\n" 编号 实例 端口 接口 容器 重启
  local d n c st rc p
  for d in $(cfg_dirs); do
    local base num; base=$(basename "$d"); num=$(echo "$base" | cut -c1-2)
    [ -n "${1:-}" ] && [ "$num" != "$1" ] && continue
    p=$(cfg_port "$base")
    local names; names=$(cfg_names "$base")
    [ -z "$names" ] && { printf "  %-4s %-14s %-6s %-8s %-12s %s\n" "$num" "(无@name)" "$p" "―" "―" "-"; continue; }
    for n in $names; do
      st=$($DOCKER inspect -f '{{.State.Status}}' "$n" 2>/dev/null); st=${st:-无}
      rc=$($DOCKER inspect -f '{{.RestartCount}}' "$n" 2>/dev/null); rc=${rc:--}
      [ "$st" = running ] && c=$(probe "$p") || c="―"
      printf "  %-4s %-30s %-6s %-8s %-12s %s\n" "$num" "$n" "$p" "$c" "$st" "$rc"
    done
  done
  # ★ 派生实例（按端口自动命名的第 2..N 个）不在 @name 元数据里，
  #   只按 cfg_names 列会完全看不见它们。这里按标记前缀补列实际存在的实例。
  local listed extra
  listed=" $(for d in $(cfg_dirs); do cfg_names "$(basename "$d")"; done | tr '\n' ' ') "
  extra=0
  for n in $($DOCKER ps -a --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null); do
    case "$listed" in *" $n "*) continue;; esac
    [ "$extra" = 0 ] && echo "  ── 派生实例（按端口命名）──" && extra=1
    st=$($DOCKER inspect -f '{{.State.Status}}' "$n" 2>/dev/null); st=${st:-无}
    rc=$($DOCKER inspect -f '{{.RestartCount}}' "$n" 2>/dev/null); rc=${rc:--}
    p=$($DOCKER inspect "$n" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
    [ "$st" = running ] && c=$(probe "${p:-0}") || c="―"
    printf "  %-4s %-30s %-6s %-8s %-12s %s\n" "―" "$n" "${p:-?}" "$c" "$st" "$rc"
  done
  echo "  KFD 进程: $(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)    权重根: $MODELS_ROOT    标记前缀: $PFX"
  if [ -f "$REBOOT_FLAG" ]; then
    echo "  ⚠ 重启待办: 是 —— 补丁已部署但机器未重启，服务起不来或补丁未生效"
    sed 's/^/      /' "$REBOOT_FLAG"
  else
    echo "  重启待办: 无"
  fi
  echo "── 容器日志尾（各在跑实例最近 8 行）──"
  local any=0
  for n in $($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null); do
    any=1
    echo "  ▸ $n:"
    $DOCKER logs --tail 8 "$n" 2>&1 | grep -vE "GET /|vllm\._C" | tail -6 | cut -c1-150 | sed "s/^/    /"
  done
  [ "$any" = 0 ] && echo "  （无在跑实例）"
  echo "  （完整日志: bash up.sh logs [-f]；或 $DOCKER logs -f <实例名>）"
}

do_logs(){  # 容器 docker 日志；多实例用 PORT=/NAME= 指定，否则取第一个在跑的
  local n=""
  for _a in "$@"; do
    case "$_a" in
      NAME=*) n="${_a#NAME=}" ;;
      PORT=*) n=$(_by_port "${_a#PORT=}") ;;
    esac
  done
  [ -n "$n" ] || n=$($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null | head -1)
  [ -n "$n" ] || { echo "无在跑实例"; exit 1; }
  if [ "${1:-}" = "-f" ]; then exec $DOCKER logs -f --tail 100 "$n"
  else $DOCKER logs --tail 200 "$n" 2>&1 | cut -c1-200; fi
}

do_benchmark(){
  # ★ 标准性能测试 —— 全部线的性能数字都出自这一个脚本，换线只改端口。
  #   口径（用户确认，唯一权威）：输入 45K–350K 均值 160K；输出 500–4K 均值 800；
  #   输入 cache 命中均值 95% 最小 85%；涨到 350K 后切 ~70K 全新前缀。
  #   产出 TTFT / prefill 速率 / decode 速率 / E2E 四个指标的「输入 × 输出」矩阵。
  local p="" cyc=2 hrs=0 out="" S="$HERE/bench/standard_load.py"
  for _a in "$@"; do
    case "$_a" in
      PORT=*)   p="${_a#PORT=}" ;;
      CYCLES=*) cyc="${_a#CYCLES=}" ;;
      HOURS=*)  hrs="${_a#HOURS=}" ;;   # 给了就按时长跑，CYCLES 忽略
      OUT=*)    out="${_a#OUT=}" ;;
    esac
  done
  [ -f "$S" ] || { echo "✗ 包内缺 bench/standard_load.py"; return 1; }
  # 未给端口时：优先取在跑实例的端口，否则用本线默认端口
  if [ -z "$p" ]; then
    local n; n=$($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null | head -1)
    [ -n "$n" ] && p=$($DOCKER inspect "$n" --format '{{join .Config.Cmd " "}}' 2>/dev/null \
                        | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
    p="${p:-$(cfg_port "$CFGNAME")}"
  fi
  [ "$(probe "$p")" = "在线" ] || {
    echo "✗ 端口 $p 上没有在线服务。先 bash up.sh 拉起，再 bash up.sh benchmark"; return 1; }
  out="${out:-$HERE/logs/bench-$p-$(date +%m%d-%H%M%S)}"
  # 抬头必须与实际跑法一致：给了 HOURS 就按时长跑，此时印「N 个周期」是误导。
  if [ "$hrs" != 0 ]; then
    echo "══ 标准性能测试：端口 $p，按时长跑 $hrs 小时 → $out ══"
  else
    echo "══ 标准性能测试：端口 $p，$cyc 个周期 → $out ══"
  fi
  echo "   （输入将从 45K 逐轮涨到 350K，单周期约数十分钟；口径见 docs/标准性能测试.md）"
  STATE_PORT="$p" emit $ST_BENCH_RUNNING BENCH_RUNNING "正在 benchmark，仍未结束" \
    "PORT=$p" "$([ "$hrs" != 0 ] && echo "HOURS=$hrs" || echo "CYCLES=$cyc")" "OUT=$out"
  python3 "$S" --port "$p" --cycles "$cyc" --hours "$hrs" --out "$out"; local rc=$?
  if [ "$rc" != 0 ]; then
    STATE_PORT="$p" emit $ST_FAILED FAILED "benchmark 异常终止" "PORT=$p" "RC=$rc" "OUT=$out"
    return "$rc"
  fi
  # ★ 自证未达标 = 跑成了别的负载，性能数字一律不作数，必须与「已结束」区分开
  if grep -q '未达标' "$out/report.md" 2>/dev/null; then
    STATE_PORT="$p" emit $ST_BENCH_INVALID BENCH_INVALID \
      "benchmark 已结束，但自证表未达标——本次性能数字不作数" \
      "PORT=$p" "REPORT=$out/report.md" \
      "FAILED_ITEMS=$(grep -c '未达标' "$out/report.md")"
    return $ST_BENCH_INVALID
  fi
  STATE_PORT="$p" emit $ST_BENCH_DONE BENCH_DONE "benchmark 已结束" \
    "PORT=$p" "REPORT=$out/report.md" "ROUNDS=$(wc -l < "$out/rounds.jsonl" 2>/dev/null || echo 0)"
}

do_restart(){
  # restart = 停掉该实例 → 用**它原本的参数**重走 S1..S8。
  # 参数从容器实况还原（卡组取 HIP_VISIBLE_DEVICES、端口取命令行），
  # 不要求使用者重报——否则 restart 就退化成「自己 stop 再自己 start」。
  local _t="" g p
  for _a in "$@"; do
    case "$_a" in
      NAME=*) _t="${_a#NAME=}" ;;
      PORT=*) _t=$(_by_port "${_a#PORT=}") ;;
    esac
  done
  if [ -z "$_t" ]; then
    local all cnt
    # ★ 只数**在跑**的实例。用 ps -a 会把早已退出的旧容器算进来：
    #   2026-09-17 实机验收时，9 天前退出的 q38-sgA-int8B 被计入，
    #   于是误判「同机有 2 个实例」而拒绝 restart。
    all=$($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null)
    cnt=$(printf '%s\n' "$all" | grep -c . || true)
    if [ "$cnt" = 1 ]; then _t="$all"
    else
      echo "✗ 同机有 $cnt 个实例，restart 必须指定 PORT= 或 NAME="
      printf '%s\n' "$all" | sed 's/^/    /'
      return 1
    fi
  fi
  $DOCKER inspect "$_t" >/dev/null 2>&1 || { echo "✗ 未找到实例 $_t"; return 1; }
  g=$($DOCKER inspect "$_t" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -oP 'HIP_VISIBLE_DEVICES=\K.*' | head -1)
  p=$($DOCKER inspect "$_t" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
  [ -n "$g" ] && [ -n "$p" ] || { echo "✗ 还原不出卡组/端口（g=[$g] p=[$p]），拒绝重启"; return 1; }
  echo "  重启 $_t —— 卡组 $g、端口 $p（取自容器实况，无需重报）"
  $DOCKER rm -f "$_t" >/dev/null 2>&1 && echo "  已停 $_t"
  sleep 3
  echo "  ── 重新拉起（走完整 S1..S8）──"
  exec bash "$0" GPUS="$g" PORT="$p" NAME="$_t"
}

do_stop(){
  # ★ 支持定点停：bash up.sh stop PORT=8102 / NAME=<实例名>
  #   不给定点参数时行为与旧版一致（按线号或全停）。
  emit $ST_STOPPING STOPPING "正在停止，仍未结束" \
    "TARGETS=$($DOCKER ps -q --filter "name=$PFX" 2>/dev/null | wc -l)"
  local _tgt=""
  for _a in "$@"; do
    case "$_a" in
      NAME=*) _tgt="${_a#NAME=}" ;;
      PORT=*) _tgt=$($DOCKER ps -a --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null | while read -r _n; do
                 _p=$($DOCKER inspect "$_n" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
                 [ "$_p" = "${_a#PORT=}" ] && echo "$_n"; done | head -1) ;;
    esac
  done
  if [ -n "$_tgt" ]; then
    if $DOCKER inspect "$_tgt" >/dev/null 2>&1; then
      $DOCKER rm -f "$_tgt" >/dev/null 2>&1 && echo "  已停 $_tgt（定点，未触碰其它实例）"
    else echo "  未找到实例 $_tgt"; fi
    sleep 2
    echo "  S9 校验：KFD 进程 $(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)（其余属仍在跑的实例）"
    return 0
  fi
  local d base num n
  for d in $(cfg_dirs); do
    base=$(basename "$d"); num=$(echo "$base" | cut -c1-2)
    [ -n "${1:-}" ] && [ "$num" != "$1" ] && continue
    for n in $(cfg_names "$base") $(cfg_legacy "$base"); do
      [ -z "$n" ] && continue
      if $DOCKER inspect "$n" >/dev/null 2>&1; then
        $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  已停 $n"
      fi
      # ★ 派生实例（第 2..N 个，名字是 <@name>-<端口>）不在 @name 元数据里，
      #   只按 cfg_names 停会把它们漏掉。必须按「该线的名字前缀」再扫一遍。
      #   2026-09-17 实机：全停只停掉 q38-sgA，派生的 q38-sgA-8102 活了下来，
      #   而下方的全局前缀兜底被 `[ -z "$1" ]` 挡住（一线一包分发恒传线号），
      #   于是「停的下」不成立。这里按线扫，既补上漏网，又不会跨线误杀。
      local _dv
      for _dv in $($DOCKER ps -a --format '{{.Names}}' 2>/dev/null | grep -E "^${n}-[0-9]+$"); do
        $DOCKER rm -f "$_dv" >/dev/null 2>&1 && echo "  已停 $_dv（派生实例）"
      done
    done
  done
  # ★ 兜底：按标记前缀清扫（覆盖任何中间状态遗留，与上面按名清理互补）
  if [ -z "${1:-}" ]; then
    for n in $($DOCKER ps -aq --filter "name=$PFX" 2>/dev/null); do
      $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  已清残留(按标记) $n"
    done
  fi
  sleep 3
  local _kfd; _kfd=$(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)
  echo "  S9 校验：KFD 进程 $_kfd（全停后应为 0；单线停止时余数属其它在跑配置）"
  # 全停后仍有 KFD 进程 = 孤儿残留，必须显式回报（否则下次起服会撞车）
  if [ -z "${1:-}" ] && [ "$_kfd" -gt 0 ] && [ "$($DOCKER ps -q --filter "name=$PFX" | wc -l)" = 0 ]; then
    emit $ST_KFD_RESIDUE KFD_RESIDUE "已停止，但 KFD 仍有 $_kfd 个孤儿进程" \
      "KFD_PROCS=$_kfd" "FIX=确认无其它业务后重启机器；绝不 rmmod/insmod 重载驱动"
    return 0
  fi
  emit $ST_STOPPED STOPPED "已完成停止指令，指定的服务已停止" "KFD_PROCS=$_kfd"
}

# ── 状态契约：上层调用者按 STATE= 行与退出码判断，不必解析人类可读文本 ──────
#   每次动作都会打印一行  STATE=<名> CODE=<码> MSG=<说明> [附加 k=v]
#   并落盘到 $HERE/.state（多实例时为 .state-<端口>），供异步轮询。
ST_READY=0;          ST_FAILED=1
ST_NEED_REBOOT=10                      # 补丁已部署，等待主机重启
ST_NO_DOCKER=11;     ST_NO_DRIVER=12   # 启动程序/驱动未安装
ST_NO_MODEL=13;      ST_NO_IMAGE=14    # 模型/镜像缺失
ST_STARTING=20;      ST_STARTING_BAD=21; ST_START_FAILED=22
ST_BENCH_RUNNING=30; ST_BENCH_DONE=31;  ST_BENCH_INVALID=32
ST_SERVICE_ERROR=40
ST_STOPPING=50;      ST_STOPPED=51
ST_NOT_FOUND=60;     ST_AMBIGUOUS=61
ST_PORT_BUSY=62;     ST_GPU_BUSY=63;    ST_PATCH_UNKNOWN=64
ST_ATTEST_FAILED=65; ST_MODEL_INCOMPLETE=66; ST_KFD_RESIDUE=67

emit(){   # 码 名 说明 [附加 k=v ...]
  local code="$1" name="$2" msg="$3"; shift 3
  local f="$HERE/.state${STATE_PORT:+-$STATE_PORT}"
  echo "STATE=$name CODE=$code MSG=$msg${*:+ $*}"
  { echo "STATE=$name"; echo "CODE=$code"; echo "MSG=$msg"
    echo "TS=$(date -Iseconds)"; for kv in "$@"; do echo "$kv"; done
  } > "$f" 2>/dev/null || true
}
# die 的状态版：打印状态行后退出，码即状态码
die_st(){ local c="$1" n="$2" m="$3"; shift 3; echo; emit "$c" "$n" "$m" "$@"; exit "$c"; }

# ★ #7：只打印、不落盘的状态输出（emit 会写 .state，state 动作不得使用）
emit_ro(){ local code="$1" name="$2" msg="$3"; shift 3
  echo "STATE=$name CODE=$code MSG=$msg${*:+ $*}"; }

do_state(){
  # ★ #7 纯无副作用：先读快照（Worker 写的权威事实），再核验容器/HTTP。
  #   不调用 emit（那会写 .state），不新增/修改/删除任何文件或资源。
  local want="" n="" p=""
  for _a in "$@"; do case "$_a" in PORT=*) want="${_a#PORT=}";; NAME=*) n="${_a#NAME=}";; esac; done

  # ① ★ #7 快照优先（须先于主机前提）：STARTING 且尚无容器时仍须返回 STARTING；
  #      2026-09-22 真实行为测试查实：原先放在驱动检查之后，无卡机上 state 一律回 NO_DRIVER，
  #      快照里的 STARTING/FAILED/NEED_REBOOT 全被环境盖掉——快照是 Worker 写的权威事实，
  #      与本机是否装驱动无关，故必须先读快照。
  #      FAILED / NEED_REBOOT 直接返回对应终态；READY 才继续核验容器与 HTTP。
  if [ -n "$want" ] && [ "${CONTRACT_LOADED:-0}" = 1 ]; then
    local _ss _sl; _ss=$(lc_read_snapshot_field "$want" state 2>/dev/null)
    _sl=$(lc_read_snapshot_field "$want" launch_id 2>/dev/null)
    case "$_ss" in
      STARTING)
        if ! _by_port "$want" >/dev/null 2>&1; then
          emit_ro $ST_STARTING STARTING "后台启动中，容器尚未创建（快照事实）" \
            "PORT=$want" "LAUNCH_ID=$_sl"; return $ST_STARTING
        fi
        local _h; _h=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$want/v1/models" 2>/dev/null)
        [ "$_h" = 200 ] || { emit_ro $ST_STARTING STARTING "后台启动中，接口未就绪" \
            "PORT=$want" "LAUNCH_ID=$_sl" "HTTP=$_h"; return $ST_STARTING; };;
      FAILED)
        emit_ro $ST_START_FAILED START_FAILED "启动失败（快照事实）" \
          "PORT=$want" "LAUNCH_ID=$_sl"; return $ST_START_FAILED;;
      NEED_REBOOT)
        emit_ro $ST_NEED_REBOOT NEED_REBOOT "补丁已部署，等待主机重启（快照事实）" \
          "PORT=$want" "LAUNCH_ID=$_sl"; return $ST_NEED_REBOOT;;
      STOPPED)
        emit_ro $ST_STOPPED STOPPED "该生命周期已停止" \
          "PORT=$want" "LAUNCH_ID=$_sl"; return $ST_STOPPED;;
      READY)
        # READY 快照不自动采信：须容器标签与快照 launch_id 互证（R36），再核 HTTP。
        # 核验只读标签与接口，不依赖本机驱动，故同样先于主机前提。
        local _rn _rl _rline _rg _rmid _rh
        _rn=$(_by_port "$want" 2>/dev/null) || {
          emit_ro $ST_NOT_FOUND NOT_FOUND "快照为 READY 但容器已不在（R36 互证失败）" \
            "PORT=$want" "LAUNCH_ID=$_sl"; return $ST_NOT_FOUND; }
        ct_owned_proof "$_rn" "$want" || {
          emit_ro $ST_AMBIGUOUS AMBIGUOUS "端口 $want 上的容器归属无法证明，需人工干预（R32）" \
            "PORT=$want" "HOLDER=$_rn" "REASON=state_unknown_ownership" \
            "DETAIL=${CT_OWN_WHY:-unknown}"; return $ST_AMBIGUOUS; }
        _rl="$CT_OWN_LID"
        _rh=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$want/v1/models" 2>/dev/null)
        [ "$_rh" = 200 ] || { emit_ro $ST_SERVICE_ERROR SERVICE_ERROR "容器在但接口不可用" \
            "PORT=$want" "HOLDER=$_rn" "LAUNCH_ID=$_sl" "HTTP=$_rh"; return $ST_SERVICE_ERROR; }
        _rg=$(ct_ctn_label "$_rn" gpus); _rmid=$(ct_ctn_label "$_rn" model_id)
        emit_ro $ST_READY READY "服务就绪（快照 + 标签互证 + 接口 200）" \
          "PORT=$want" "NAME=$_rn" "LAUNCH_ID=$_sl" "GPUS=$_rg" "MODEL_ID=$_rmid" \
          "LAUNCH_MODE=$(ct_ctn_label "$_rn" launch_mode)" "ENDPOINT=http://127.0.0.1:$want/v1"
        return $ST_READY;;
    esac
  fi

  # ① 主机前提（无快照可依时才回落到环境判定）
  command -v docker >/dev/null || { emit_ro $ST_NO_DOCKER NO_DOCKER "没有 docker 命令"; return $ST_NO_DOCKER; }
  $DOCKER info >/dev/null 2>&1 || { emit_ro $ST_NO_DOCKER NO_DOCKER "docker 不可用（权限或服务未起）"; return $ST_NO_DOCKER; }
  { [ -e /dev/kfd ] && [ -d /opt/hyhal ]; } || {
    emit_ro $ST_NO_DRIVER NO_DRIVER "驱动未安装（缺 /dev/kfd 或 /opt/hyhal）——本包不负责装驱动"
    return $ST_NO_DRIVER; }

  # ② 等待重启优先于一切服务状态：未重启时补丁没生效，起来了也不算数
  [ -f "$REBOOT_FLAG" ] && {
    emit_ro $ST_NEED_REBOOT NEED_REBOOT "补丁已部署，等待主机重启" \
      "$(grep -h '^REBOOT_REASON=' "$REBOOT_FLAG" 2>/dev/null | head -1)"
    return $ST_NEED_REBOOT; }

  # ③ 压测判定移到「定位实例之后」，见下方 ⑥·5。
  #   ★ R13.6：原先在此处用全局 `pgrep -f standard_load.py` 判断，与端口无关，
  #   于是任一实例在压测就把两侧都盖成 BENCH_RUNNING。2026-09-17 实测：A 的
  #   压测已结束（进程由 2 降到 1），两个端口却仍都报 BENCH_RUNNING。
  #   按实例查的状态必须按该实例判定，不得用全局信号代答。


  # ④ 定位实例：给了端口就按端口；没给且有多个实例则判为歧义
  if [ -z "$n" ]; then
    if [ -n "$want" ]; then
      n=$(_by_port "$want") || { emit_ro $ST_NOT_FOUND NOT_FOUND "端口 $want 上没有本流程的实例" "PORT=$want"
                                 return $ST_NOT_FOUND; }
    else
      # ★ 歧义判定只数**在跑**的实例。用 ps -a 会把早已退出的旧容器算进来，
      #   机上若留有两具「尸体」，无实例时会被误报成 AMBIGUOUS 而非 STOPPED。
      #   （按端口查单个实例仍用 _by_port/ps -a，那里需要能查到已停的实例。）
      local all cnt; all=$($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null)
      cnt=$(printf '%s\n' "$all" | grep -c . || true)
      if [ "$cnt" = 0 ]; then
        emit_ro $ST_STOPPED STOPPED "无实例，服务已停止"
        return $ST_STOPPED
      elif [ "$cnt" -gt 1 ]; then
        emit_ro $ST_AMBIGUOUS AMBIGUOUS "同机有 $cnt 个实例，须用 PORT= 指定要看哪一个" \
          "INSTANCES=$(printf '%s' "$all" | tr '\n' ',')"
        return $ST_AMBIGUOUS
      fi
      n="$all"
    fi
  fi
  p=$($DOCKER inspect "$n" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
  STATE_PORT="${p:-$want}"

  # ⑤ 容器在不在、跑没跑
  local run; run=$($DOCKER inspect "$n" --format '{{.State.Running}}' 2>/dev/null)
  if [ "$run" != true ]; then
    local ec; ec=$($DOCKER inspect "$n" --format '{{.State.ExitCode}}' 2>/dev/null)
    emit_ro $ST_STOPPED STOPPED "实例已停止" "NAME=$n" "PORT=${p:-?}" "EXIT_CODE=${ec:-?}"
    return $ST_STOPPED
  fi

  # ⑥ 接口通不通 → 区分「仍在拉起」与「已就绪」与「服务异常」
  local http; http=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/v1/models" 2>/dev/null)
  local lg; lg=$($DOCKER logs --tail 200 "$n" 2>&1)
  local bad=0 sym=""
  echo "$lg" | grep -qi 'out of memory'      && { bad=1; sym="$sym OOM"; }
  echo "$lg" | grep -q  'Traceback'          && { bad=1; sym="$sym Traceback"; }
  echo "$lg" | grep -qE '" 5[0-9]{2}'        && { bad=1; sym="$sym 5xx"; }

  if [ "$http" = 200 ]; then
    [ "$bad" = 1 ] && { emit_ro $ST_SERVICE_ERROR SERVICE_ERROR "服务异常，表现为:$sym" \
                          "NAME=$n" "PORT=$p" "HTTP=200"; return $ST_SERVICE_ERROR; }
    # ⑥·5 压测判定：只认**打在本实例端口上**的 benchmark 进程（R13.6）
    local _bp _bpid=""
    for _bp in $(pgrep -f 'standard_load\.py' 2>/dev/null); do
      tr '\0' ' ' < "/proc/$_bp/cmdline" 2>/dev/null \
        | grep -qE -- "--port[= ]$p( |$)" && { _bpid="$_bp"; break; }
    done
    if [ -n "$_bpid" ]; then
      emit_ro $ST_BENCH_RUNNING BENCH_RUNNING "正在 benchmark，仍未结束" \
        "NAME=$n" "PORT=$p" "BENCH_PID=$_bpid"
      return $ST_BENCH_RUNNING
    fi
    local g; g=$($DOCKER inspect "$n" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
                 | grep -oP 'HIP_VISIBLE_DEVICES=\K.*' | head -1)
    emit_ro $ST_READY READY "服务正常，等待命令中" "NAME=$n" "PORT=$p" "GPUS=${g:-?}" \
      "UPTIME=$($DOCKER inspect "$n" --format '{{.State.StartedAt}}' 2>/dev/null)"
    return $ST_READY
  fi

  if [ "$bad" = 1 ]; then
    emit_ro $ST_STARTING_BAD STARTING_BAD "正在拉起但状态异常:$sym" "NAME=$n" "PORT=$p" "HTTP=$http"
    return $ST_STARTING_BAD
  fi
  emit_ro $ST_STARTING STARTING "正在拉起，状态正常，仍在等待就绪" "NAME=$n" "PORT=$p" "HTTP=$http" \
    "LAST_LOG=$(echo "$lg" | tail -1 | cut -c1-80)"
  return $ST_STARTING
}

precheck(){
  say "① 环境体检（硬性前提）"
  # 每条前提失败都回报**具体**状态码，而不是笼统的 exit 1——上层才能分别处置
  command -v docker >/dev/null \
    || die_st $ST_NO_DOCKER NO_DOCKER "没有 docker 命令" "FIX=安装 docker 后重试"
  $DOCKER info >/dev/null 2>&1 \
    || die_st $ST_NO_DOCKER NO_DOCKER "docker 不可用" "FIX=把用户加入 docker 组，或配置 sudo NOPASSWD"
  [ -e /dev/kfd ] \
    || die_st $ST_NO_DRIVER NO_DRIVER "没有 /dev/kfd —— DCU 驱动未装载" "FIX=先装 DTK/驱动；本包不负责装驱动"
  NGPU=$(ls -d /sys/bus/pci/drivers/hycu/0000:* 2>/dev/null | wc -l)
  [ "$NGPU" -ge 4 ] \
    || die_st $ST_NO_DRIVER NO_DRIVER "hycu 只绑定 $NGPU 张卡（需 ≥4）" "NGPU=$NGPU" "FIX=检查驱动与硬件"
  [ -d /opt/hyhal ] \
    || die_st $ST_NO_DRIVER NO_DRIVER "缺 /opt/hyhal（容器必挂）" "FIX=从 DTK 安装或从已装机器拷贝"
  sudo -n true 2>/dev/null || [ "$(id -u)" = 0 ] \
    || die_st $ST_FAILED FAILED "S1 需要 root" "FIX=以 root 运行或配置 sudo NOPASSWD"
  # 模型权重：@requires 里的路径必须在位（缺了就不必往下走，省掉六分钟的等待）
  # 「不在」与「在但不完整」是两种状态：前者要去取，后者要去修，上层处置不同。
  local _m
  for _m in $(meta "$HERE/$CFGNAME" requires 2>/dev/null); do
    case "$_m" in
      *MODELS_ROOT*) eval "_m=\"$_m\"" 2>/dev/null
        [ -e "$_m" ] || die_st $ST_NO_MODEL NO_MODEL "指定位置没有对应模型: $_m" \
            "MODELS_ROOT=$MODELS_ROOT" "MISSING=$_m" "FIX=放置权重，或设 MODELS_ROOT= 指向正确目录"
        # 是目录就要像个权重目录：有 config.json，且有权重分片
        if [ -d "$_m" ]; then
          [ -f "$_m/config.json" ] \
            || die_st $ST_MODEL_INCOMPLETE MODEL_INCOMPLETE "权重目录缺 config.json: $_m" \
                 "PATH=$_m" "FIX=重新获取权重（可能是上次下载中断）"
          [ "$(ls "$_m"/*.safetensors "$_m"/*.bin 2>/dev/null | wc -l)" -gt 0 ] \
            || die_st $ST_MODEL_INCOMPLETE MODEL_INCOMPLETE "权重目录里没有任何分片文件: $_m" \
                 "PATH=$_m" "FIX=重新获取权重（可能是上次下载中断）"
        fi;;
    esac
  done
  # 镜像：离线现场（SKIP_S2=1）不会去拉，此时镜像不在就得当场说清楚，
  # 而不是等到 S4 起容器时报一句难懂的 docker 错误。
  local _img; _img=$(meta "$HERE/$CFGNAME" image 2>/dev/null)
  if [ -n "$_img" ] && [ "${SKIP_S2:-${SKIP_FETCH:-0}}" = 1 ]; then
    $DOCKER image inspect "$_img" >/dev/null 2>&1 \
      || die_st $ST_NO_IMAGE NO_IMAGE "镜像不在本机，且已设 SKIP_S2=1 不去获取" \
           "IMAGE=$_img" "FIX=先 docker pull 该镜像，或去掉 SKIP_S2=1 让 S2 自动获取"
  fi
  echo "  ✓ docker / 驱动($NGPU 卡) / hyhal / 提权 / 权重 / 镜像"
}

deploy_patches(){
  # ── S1.5 随包补丁自动部署 ──────────────────────────────────────────────
  # 边界：**驱动本身不由本包安装**（宿主机前提，precheck 只检查不安装）；
  #       但对「已装驱动」的补丁由本包负责部署，含内核模块补丁。
  # 内核模块补丁改的是磁盘上的 .ko，运行中的驱动仍是旧的——**必须重启才生效**，
  # 且一律「改文件 → 重启机器」，绝不 rmmod/insmod 重载（会破坏驱动）。
  # 故部署完若需重启，写 .reboot-required 并以 exit 10 回报上层调用者。
  say "②·5 随包补丁部署（驱动本身不装，只打补丁）"
  local SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  local need=0
  rm -f "$REBOOT_FLAG" 2>/dev/null || true

  # ① 内核模块补丁：hycu.ko v4（TP8 全 P2P 的硬前提，13/14/16 线必需）
  if [ -f "$HERE/patches/patch_v4.sh" ]; then
    local ko="${HYCU_KO:-/usr/local/hyhal/dkms/hycu.ko}"
    local off=$((0xa0 + 0x2b3dd))        # 0x2B47D，gpu_dma_buf_attach 内的 jns
    if [ -f "$ko" ]; then
      local b; b=$($SUDO xxd -p -s "$off" -l 1 "$ko" 2>/dev/null || echo "")
      case "$b" in
        eb) echo "  ✓ hycu.ko v4 补丁已在位（无需重启）";;
        79) echo "  · hycu.ko 未打 v4 补丁 —— 自动部署中…"
            if $SUDO bash "$HERE/patches/patch_v4.sh" apply 2>&1 | sed 's/^/    /'; then
              need=1
              echo "REBOOT_REASON=hycu.ko v4 补丁已写入磁盘，需重启后驱动才生效" >> "$REBOOT_FLAG"
            else
              die "hycu.ko v4 补丁部署失败" "手工执行 sudo bash patches/patch_v4.sh status 查看"
            fi;;
        "") echo "  ⚠ 读不到 $ko 的补丁字节（权限或缺 xxd），跳过 v4 自动部署";;
        *)  die_st $ST_PATCH_UNKNOWN PATCH_UNKNOWN \
              "v4 补丁字节与预期不符（读到 '$b'，期望 eb/79），拒绝自动改写" \
              "KO=$ko" "OFFSET=$off" "FIX=偏移可能随驱动版本变化，人工确认后再投产";;
      esac
    else
      echo "  ⚠ 未找到 $ko，跳过 v4（若驱动路径不同，用 HYCU_KO= 指定）"
    fi
  fi

  # ② 运行时补丁（免重启）：1M 农场、repair 提取件等，由各线 serve.sh 在容器内挂载生效
  [ -d "$HERE/patches/model-1m-farm" ] && echo "  ✓ 1M 农场随包（运行时挂载，免重启）"
  [ -d "$HERE/patches/repair-from-0512" ] && echo "  ✓ repair 提取件随包（运行时挂载，免重启）"

  if [ "$need" = 1 ]; then
    echo "NEED_REBOOT=1" >> "$REBOOT_FLAG"
    echo
    echo "════════════════════════════════════════════════════════"
    echo "  ⚠ 补丁已部署，但**需要重启机器**后才生效"
    cat "$REBOOT_FLAG" | sed 's/^/    /'
    echo "    重启后重跑本命令即可继续拉起。"
    echo "════════════════════════════════════════════════════════"
    die_st $ST_NEED_REBOOT NEED_REBOOT "补丁已部署，等待主机重启" \
      "REBOOT_FLAG=$REBOOT_FLAG" "FIX=重启机器后重跑同一条命令"
  fi
}

s1(){
  [ "${S1_SKIP:-0}" = 1 ] && { say "② S1 已跳过（S1_SKIP=1）"; return; }
  say "② S1 机器准备（内核参数 + ACS 清除 + 开机自愈；幂等；产物带 $PFX 标记）"
  local SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  local ACS_SRC="$HERE/patches/acs_clear_all.sh"; [ -f "$ACS_SRC" ] || ACS_SRC="$HERE/patches/rccl-acs-topo/acs_clear_all.sh"
  local SVC_SRC="$HERE/patches/acs-clear.service"; [ -f "$SVC_SRC" ] || SVC_SRC="$HERE/patches/rccl-acs-topo/acs-clear.service"
  # 脚本内部引用一并改写到带标记路径
  sed "s|/usr/local/sbin/acs_clear_all.sh|/usr/local/sbin/${PFX}acs_clear_all.sh|" \
    "$HERE/common/machine_prep.sh" | $SUDO tee "/usr/local/sbin/${PFX}machine_prep.sh" >/dev/null
  $SUDO chmod 0755 "/usr/local/sbin/${PFX}machine_prep.sh"
  $SUDO install -m 0755 "$ACS_SRC" "/usr/local/sbin/${PFX}acs_clear_all.sh"
  echo "kernel.numa_balancing=0" | $SUDO tee "/etc/sysctl.d/99-${PFX}dcu.conf" >/dev/null
  sed "s|ExecStart=.*|ExecStart=/usr/local/sbin/${PFX}machine_prep.sh|" "$SVC_SRC" \
    | $SUDO tee "/etc/systemd/system/${PFX}acs-clear.service" >/dev/null
  # 迁移：停用旧的无标记单元/文件（若存在）
  if systemctl is-enabled acs-clear.service >/dev/null 2>&1; then
    $SUDO systemctl disable acs-clear.service >/dev/null 2>&1
    echo "  · 已停用旧单元 acs-clear.service（由 ${PFX}acs-clear.service 接管）"
  fi
  $SUDO systemctl daemon-reload && $SUDO systemctl enable "${PFX}acs-clear.service" >/dev/null 2>&1
  if [ -n "$SUDO" ]; then
    EXPECT_GPUS="$NGPU" "$SUDO" -E bash "/usr/local/sbin/${PFX}machine_prep.sh" --now | sed 's/^/  /'
  else
    EXPECT_GPUS="$NGPU" bash "/usr/local/sbin/${PFX}machine_prep.sh" --now | sed 's/^/  /'
  fi
}

smoke(){
  local port="$1"
  say "⑤ 实测请求（端口 $port；首请求含 tokenizer 装载与内核暖场，可能需数分钟）"
  python3 - "$port" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
p = {"model": "qwen38", "messages": [{"role": "user", "content": "用一句话自我介绍。"}],
     "max_tokens": 60, "temperature": 0}
r = urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % port,
    json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=900)
d = json.loads(r.read())
m = d["choices"][0]["message"]
txt = (m.get("content") or m.get("reasoning") or m.get("reasoning_content") or "").strip()
assert txt, "服务返回了空内容"
print("  模型回复:", txt[:120].replace("\n", " "))
PY
}

menu(){
  echo "════ Qwen3.8-27B on 海光 K100-AI · 拉起包 ════"
  echo "包位置: $HERE"
  echo
  echo "可用配置（详见各目录 README.md；目录名内含实测 单流/聚合 decode-prefill）："
  local d base
  for d in $(cfg_dirs); do
    base=$(basename "$d")
    printf "  %s\n" "$base"
  done
  echo
  echo "推荐：低延迟交互→09；高QPS批量→10（GPUS=4,5,6,7 可与 09 并存）；"
  echo "      1M 上下文 bf16 保守线→01；8卡双副本+网关→02。"
  echo
  echo "用法： bash up.sh 09            拉起"
  echo "       bash up.sh 09 status    看状态      bash up.sh status   看全部"
  echo "       bash up.sh 09 stop      停止        bash up.sh stop     全停"
  echo
  do_status || true
}

# ════════════════════════════════════════════════════════════════
# ★ 位置要紧（2026-09-22 真实行为测试查实）：以下函数**必须定义在参数分发之前**。
#   原先它们写在分发块之下的同步主流程里，而一线一包分发在命中分支后即 exit ——
#   `up.sh start` 跑到调用点时函数尚未定义，报 `contract_resolve: command not found`(127)。
#   源码 grep 看不出这个缺陷，只有真跑 start 才暴露。
# ════════════════════════════════════════════════════════════════
# ════ #9 归属安全的 stop ════
#   按 PORT= 或 LAUNCH_ID= 解析快照，只停「标签 + 快照互证」属本包的容器。
#   遇同名/同端口未知资源一律失败关闭，绝不按名称前缀全删。
do_stop_safe(){
  local want="" wlid=""
  for _a in "$@"; do case "$_a" in PORT=*) want="${_a#PORT=}";; LAUNCH_ID=*) wlid="${_a#LAUNCH_ID=}";; esac; done
  [ "${CONTRACT_LOADED:-0}" = 1 ] || { do_stop "$NUM" "$@"; return 0; }
  local line pkg; line="$(echo "$CFGNAME" | cut -c1-2)"; pkg="$(awk '{print $2}' "$HERE/VERSION" 2>/dev/null)"
  local ports="" p n lid iid stopped=0
  if [ -n "$want" ]; then ports="$want"
  elif [ -n "$wlid" ]; then
    for f in "$(lc_dir)"/snapshot-*.json; do [ -f "$f" ] || continue
      [ "$(sed -n 's/.*"launch_id":"\([^"]*\)".*/\1/p' "$f"|head -1)" = "$wlid" ] && \
        ports="$ports $(sed -n 's/.*"port":"\([^"]*\)".*/\1/p' "$f"|head -1)"; done
  else
    # 全停：只处理本包 state 目录中可证明的 launch，不碰其它资源
    for f in "$(lc_dir)"/snapshot-*.json; do [ -f "$f" ] || continue
      ports="$ports $(sed -n 's/.*"port":"\([^"]*\)".*/\1/p' "$f"|head -1)"; done
  fi
  for p in $ports; do
    [ -z "$p" ] && continue
    n=$(_by_port "$p" 2>/dev/null || true)
    # ★ 与复用/核验/清理同一把尺子：publisher+package_id+line+快照 launch_id 全等才停。
    if [ -n "$n" ] && ! ct_owned_proof "$n" "$p"; then
      emit $ST_AMBIGUOUS AMBIGUOUS "端口 $p 上的资源归属无法证明，拒绝停止，需人工干预（R32）" \
        "HOLDER=$n" "PORT=$p" "REASON=stop_unknown_ownership" "DETAIL=${CT_OWN_WHY:-unknown}"
      return $ST_AMBIGUOUS
    fi
    lid=$(lc_read_snapshot_field "$p" launch_id 2>/dev/null)
    if [ -n "$lid" ] && ! lc_stop_worker "$lid" "$(lc_read_snapshot_field "$p" state 2>/dev/null)"; then
      emit $ST_AMBIGUOUS AMBIGUOUS "无法证明后台启动链已安全停止，需人工干预" \
        "PORT=$p" "LAUNCH_ID=$lid" "REASON=${LC_STOP_REASON:-worker_stop_unproved}"
      return $ST_AMBIGUOUS
    fi
    # Worker 停止之前可能刚创建容器；终止启动链后重新发现、互证，再定点删除。
    n=$(_by_port "$p" 2>/dev/null || true)
    if [ -n "$n" ]; then
      if ! ct_owned_proof "$n" "$p"; then
        emit $ST_AMBIGUOUS AMBIGUOUS "停止期间出现归属不明的资源，拒绝删除" \
          "HOLDER=$n" "PORT=$p" "DETAIL=${CT_OWN_WHY:-unknown}"
        return $ST_AMBIGUOUS
      fi
      if ! $DOCKER rm -f "$n" >/dev/null 2>&1; then
        emit $ST_FAILED FAILED "容器停止失败，不能回报 STOPPED" "HOLDER=$n" "PORT=$p"
        return $ST_FAILED
      fi
      echo "  已停（归属已互证）: $n"; stopped=$((stopped+1))
    fi
    if [ -n "$lid" ]; then
      iid=$(lc_new_id)
      LC_GPUS=$(lc_read_snapshot_field "$p" GPUS 2>/dev/null)
      LC_MODE=docker LC_LINE="$line"
      lc_write_snapshot "$p" "$lid" "$iid" STOPPED "$ST_STOPPED" stopped "GPUS=$LC_GPUS"
      lc_emit_event "$lid" "$iid" "$p" STOPPED "$ST_STOPPED" stopped
    fi
  done
  emit $ST_STOPPED STOPPED "已停止经互证的实例" "STOPPED=$stopped"
  return $ST_STOPPED
}

# ════ 合同判定（R26–R28）——进程内函数，start 直接调用，不递归起子进程 ════
#   返回 0 = 可继续新建生命周期；非 0 = 已回报状态并应由调用方退出。
#   ★ 修（递归 __decide）：原实现 start 调 `bash up.sh __decide`，子进程走完整同步
#     主流程（precheck/S1/launch/S7/S8/smoke），start 因此仍然阻塞。改为函数后，
#     同步段只做「发现实例 + 归属核验 + 决策」，绝不触碰任何同步链步骤。
ct_model_id(){ # 宿主 realpath → 稳定非敏感摘要（R25.1 / 修 #5：路径原文不得外泄）
  local rp; rp=$(ct_norm_model "${1:-}")
  printf '%s' "$rp" | sha256sum | cut -c1-16
}
ct_ctn_label(){ $DOCKER inspect "$1" --format "{{index .Config.Labels \"$2\"}}" 2>/dev/null | grep -v '^<no value>$'; }

# ════ 唯一的归属证明（R36）——复用 / 核验 / 停止 / 清理 四类入口统一走它 ════
#   $1=容器名 $2=期望端口（可空）→ 0 = 已证明属本包；非 0 = 未知归属（调用方失败关闭）
#   四项**全部**精确相等才算证明，任何一项缺失或不符都判未知：
#     publisher   == 01ma                     （发布方）
#     package_id  == 本包 VERSION 的版本       （必须存在；**缺失不视为允许**）
#     line        == 本包线号                 （线级隔离）
#     launch_id   == **该容器自己端口**快照的 launch_id（本次生命周期；无快照即未知）
#
#   ★ 事实源必须是「容器自己的端口」，不是本次要起的目标端口（2026-09-22 修）：
#     端口迁移时旧实例在旧端口上，它的快照自然写在 snapshot-<旧端口>.json；
#     若拿尚未创建的新目标端口快照去证明旧实例，必然 no_snapshot ⇒ 旧实例被误判未知，
#     R28 的自动定点停旧就退化成人工处置。旧端口从**稳定标签 port** 读，不从命令行猜。
#     $2 非空时额外要求「标签 port == $2」，供 state/stop 这类按端口定位的调用方自校验。
#   失败原因写入 CT_OWN_WHY，证明通过时 CT_OWN_LID/CT_OWN_PORT 给出 launch_id 与其端口。
CT_OWN_PUBLISHER=01ma
ct_owned_proof(){
  local c="$1" want="${2:-}"
  local pub pkgid cline clid slid port
  local line pkg; line="$(echo "$CFGNAME" | cut -c1-2)"; pkg="$(awk '{print $2}' "$HERE/VERSION" 2>/dev/null)"
  CT_OWN_WHY=""; CT_OWN_LID=""; CT_OWN_PORT=""
  pub=$(ct_ctn_label "$c" publisher)
  [ -n "$pub" ]   || { CT_OWN_WHY=missing_publisher;     return 1; }
  [ "$pub" = "$CT_OWN_PUBLISHER" ] || { CT_OWN_WHY=publisher_mismatch; return 1; }
  pkgid=$(ct_ctn_label "$c" package_id)
  [ -n "$pkgid" ] || { CT_OWN_WHY=missing_package_id;    return 1; }
  [ -n "$pkg" ]   || { CT_OWN_WHY=missing_local_version; return 1; }
  [ "$pkgid" = "$pkg" ] || { CT_OWN_WHY=package_id_mismatch; return 1; }
  cline=$(ct_ctn_label "$c" line)
  [ -n "$cline" ] || { CT_OWN_WHY=missing_line;          return 1; }
  [ "$cline" = "$line" ] || { CT_OWN_WHY=line_mismatch;  return 1; }
  clid=$(ct_ctn_label "$c" launch_id)
  [ -n "$clid" ] || { CT_OWN_WHY=missing_launch_id;      return 1; }
  port=$(ct_ctn_label "$c" port)                        # 稳定标签，不从 Cmd 猜
  [ -n "$port" ] || { CT_OWN_WHY=missing_port_label;     return 1; }
  [ -z "$want" ] || [ "$port" = "$want" ] || { CT_OWN_WHY=port_mismatch; return 1; }
  slid=$(lc_read_snapshot_field "$port" launch_id 2>/dev/null)
  [ -n "$slid" ] || { CT_OWN_WHY=no_snapshot;            return 1; }
  [ "$clid" = "$slid" ] || { CT_OWN_WHY=launch_id_mismatch; return 1; }
  CT_OWN_LID="$clid"; CT_OWN_PORT="$port"
  return 0
}

# 目标端口是否被**任意**容器占用（不限本包名字前缀）
#   ★ 不能用 _by_port：它只扫 name=$PFX 的容器，看不见别人的容器。
#     换端口时若新端口被外来容器占着而漏判，就会先把旧实例停掉、再在下一轮迭代里
#     报 AMBIGUOUS —— 服务没了、也没起来。故此处按端口做全量扫描，与循环顺序无关。
ct_port_taken_by_ctn(){
  local want="$1" skip="${2:-}" n p
  # ★ 只扫**运行中**容器（2026-09-23 验证机B 实机查实）：已退出容器的历史 --port
  #   声明不占用任何端口，把它算作占用会让换端口永远 PORT_BUSY。宿主监听另行检查。
  for n in $($DOCKER ps --format '{{.Names}}' 2>/dev/null); do
    [ -n "$skip" ] && [ "$n" = "$skip" ] && continue
    p=$(ct_ctn_label "$n" port)
    [ -z "$p" ] && p=$($DOCKER inspect "$n" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
    [ -n "$p" ] && [ "$p" = "$want" ] && { echo "$n"; return 0; }
  done
  return 1
}

# 目标端口是否被宿主进程占用（修 #6：不能只看 docker）
ct_host_port_busy(){
  local p="$1"
  (command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -qE "[:.]$p[[:space:]]") && return 0
  (command -v netstat >/dev/null && netstat -ltn 2>/dev/null | grep -qE "[:.]$p[[:space:]]") && return 0
  return 1
}

contract_resolve(){ # $1=port $2=gpus  → 0 可继续新建；非 0 = 已回报，调用方按 $CT_EXIT 退出
  # ★ 不能用返回值兼表状态码（2026-09-22 真实行为测试查实）：ST_READY=0，
  #   REUSE 分支 `CT_EXIT=$ST_READY; return 99` 与「可继续」同为 0，start 因此在复用后又派发了 Worker。
  #   故终态一律置 CT_EXIT 后 return 99，调用方 `|| exit $CT_EXIT`。
  CT_EXIT=0
  [ "${CONTRACT_LOADED:-0}" = 1 ] || return 0
  local port="$1" gpus="$2"
  local mydir; mydir="$(ct_line_model)"
  local mymid; mymid=$(ct_model_id "$mydir")
  local myg;   myg=$(ct_norm_gpus "$gpus")
  local line; line="$(echo "$CFGNAME" | cut -c1-2)"
  local pkg;  pkg="$(awk '{print $2}' "$HERE/VERSION" 2>/dev/null)"
  local c lb cmid cg cp cmode clid cline cpkg rel own relevant crun

  # ── ① 目标端口快照前置（2026-09-23 实机查实的缺陷 4）────────────────────
  #   Worker 在 precheck/S1/补丁阶段还没建容器（实测约 4 秒窗口）。原实现只遍历容器，
  #   这期间重复 start 会给同一端口再派一个 Worker，两个生命周期抢同一组卡。
  #   故先看快照：STARTING 且该 launch_id 仍活（Worker 进程或容器在）⇒ 挂接原 launch_id。
  local _ssn _sln
  _ssn=$(lc_read_snapshot_field "$port" state 2>/dev/null)
  _sln=$(lc_read_snapshot_field "$port" launch_id 2>/dev/null)
  if [ "$_ssn" = STARTING ] && [ -n "$_sln" ]; then
    # 只有「同一组卡」才谈得上是同一个生命周期（R25.2），不同卡组不得误挂接
    local _sgp; _sgp=$(ct_norm_gpus "$(lc_read_snapshot_field "$port" GPUS 2>/dev/null)")
    # ★ 卡组不同 ⇒ 不是同一个生命周期，但**更不能另起一个**（2026-09-23 实机查实的缺陷 5）：
    #   原实现在这里直接放行，于是第二次 start 用别的卡组在同一端口又派了一个 Worker，
    #   并把 8101 的快照覆盖成自己的 launch_id。后果是第一个还在跑的生命周期账本被毁：
    #   state 报 START_FAILED，stop 报 launch_id_mismatch —— 包再也停不掉自己起的服务。
    if [ -n "$_sgp" ] && [ "$_sgp" != "$myg" ] && lc_worker_alive "$_sln" "$HERE/up.sh"; then
      emit $ST_PORT_BUSY PORT_BUSY "端口 $port 已被本包另一个正在拉起的生命周期占用（卡组 $_sgp）" \
        "PORT=$port" "LAUNCH_ID=$_sln" "WANT_GPUS=$myg" "HAVE_GPUS=$_sgp" "REASON=port_held_by_other_launch"
      CT_EXIT=$ST_PORT_BUSY; return 99
    fi
    if [ -z "$_sgp" ] || [ "$_sgp" = "$myg" ]; then
      # 证明该 launch_id 的 Worker 确实活着：PID 存活 **且** /proc cmdline 精确属于本包 __worker
      if lc_worker_alive "$_sln" "$HERE/up.sh"; then
        emit $ST_STARTING STARTING "该端口已有生命周期在拉起，挂接原 launch_id（R27/R29）" \
          "PORT=$port" "LAUNCH_ID=$_sln" "GPUS=$myg" "REASON=starting_worker_alive"
        CT_EXIT=$ST_STARTING; return 99
      fi
      # 没有活 Worker 但容器已建起来：交给下面的容器遍历去判（ATTACH/RECREATE）
      [ -n "$(_by_port "$port" 2>/dev/null)" ] || \
        echo "  · 快照 STARTING 但 Worker 已不在（PID/cmdline 核验未过）且无容器 ⇒ 视为夭折，新建生命周期"
    fi
  fi

  # ── ② 目标端口被**未知 running 容器**占用 ⇒ 恒 PORT_BUSY（缺陷 3：不能靠枚举顺序）──
  local _tk; _tk=$(ct_port_taken_by_ctn "$port" 2>/dev/null || true)
  if [ -n "$_tk" ] && ! ct_owned_proof "$_tk" "$port"; then
    emit $ST_PORT_BUSY PORT_BUSY "目标端口 $port 被归属不明的运行中容器占用（R26 情形 5 / R32）" \
      "PORT=$port" "HOLDER=$_tk" "REASON=port_taken_unknown" "DETAIL=${CT_OWN_WHY:-unknown}"
    CT_EXIT=$ST_PORT_BUSY; return 99
  fi

  for c in $($DOCKER ps -a --format '{{.Names}}' 2>/dev/null); do
    # ── 从标签读身份（修 #5：不拿容器内 /models/target 与宿主 realpath 比）──
    cmid=$(ct_ctn_label "$c" model_id)
    cg=$(ct_ctn_label "$c" gpus);   [ -z "$cg" ] && cg=$($DOCKER inspect "$c" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^HIP_VISIBLE_DEVICES=//p' | head -1)
    cp=$(ct_ctn_label "$c" port);   [ -z "$cp" ] && cp=$($DOCKER inspect "$c" --format '{{join .Config.Cmd " "}}' 2>/dev/null | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
    cmode=$(ct_ctn_label "$c" launch_mode); [ -z "$cmode" ] && cmode=docker   # 读既有标签，不硬编码本次 mode
    clid=$(ct_ctn_label "$c" launch_id)
    cline=$(ct_ctn_label "$c" line); cpkg=$(ct_ctn_label "$c" package_id)
    cg=$(ct_norm_gpus "$cg")
    crun=$($DOCKER inspect "$c" --format '{{.State.Running}}' 2>/dev/null)

    # ── 相关性（修 #6）：只有占目标端口、或 GPU 相交，才算与本次相关 ──
    rel=$(ct_gpu_relation "$myg" "${cg:-}")
    relevant=no
    [ -n "$cp" ] && [ "$cp" = "$port" ] && relevant=yes
    [ -n "$cg" ] && [ "$rel" != disjoint ] && relevant=yes
    [ "$relevant" = no ] && continue          # 无关容器不阻断

    # ── 未知归属 + 已退出 ⇒ 不构成现实占用，不阻断也不清理（2026-09-23 实机查实）──
    #   验证机B 上三个 6-8 天前 exited 的容器声明 HIP_VISIBLE_DEVICES=0..7，
    #   与任何卡组都相交，于是本包在该机上**任何 GPU 组合都起不来**。
    #   已退出的容器既不占卡也不占端口：R38.1 说的「实际相关」要的是现实占用事实，
    #   不是历史声明。故此处先判运行状态——但顺序要紧：**先证归属再看运行状态**，
    #   否则本包自己 FAILED 的已退出残留也会被跳过，R27 的 RECREATE 就失效了。
    #   同名未知 exited 容器若最终造成 docker 名字冲突，安全失败即可，不自动删除。

    # ── 归属：唯一证明（publisher+package_id+line+容器自身端口快照 launch_id 全等）──
    #   ★ 收紧（2026-09-22）：原实现把 package_id 缺失当作「可以接受」，也不看 publisher，
    #     于是别的发布方、别的版本、甚至没打标签的容器都可能被当成本包资源去复用或删除。
    #     现在一项不符即未知归属，失败关闭：不复用、不清理、不派发 Worker。
    own=unknown
    ct_owned_proof "$c" && own=own      # ★ 按容器自己的端口快照互证（端口迁移的旧实例在旧端口上）
    if [ "$own" != own ]; then
      # 未知归属：只有**正在运行**才是现实冲突；已退出的一律跳过，不阻断不清理。
      if [ "$crun" != true ]; then
        echo "  · 跳过已退出的未知容器 $c（历史声明 GPUS=[${cg:-无}] PORT=[${cp:-无}] 不构成当前占用）"
        continue
      fi
      emit $ST_AMBIGUOUS AMBIGUOUS "相关资源归属无法证明，需人工干预（R26 情形 5 / R32）" \
        "HOLDER=$c" "PORT=${cp:-?}" "REASON=unknown_ownership" "DETAIL=${CT_OWN_WHY:-unknown}"
      CT_EXIT=$ST_AMBIGUOUS; return 99
    fi

    # ── 本包资源：读快照状态，但**必须用现实核实**（2026-09-23 实机查实的缺陷 2）──
    #   原实现直接采信快照：容器被 kill 后快照仍是 READY，start 就回「已就绪，复用」，
    #   调用方以为服务在跑，实际是死的，也不会触发任何恢复。
    #   R27 的 REUSE 前提是「该实例确实在服务」，不是「快照曾经写过 READY」。
    local st; st=$(lc_read_snapshot_field "${cp:-$port}" state 2>/dev/null)
    [ -z "$st" ] && { [ "$crun" = true ] && st=STARTING || st=FAILED; }
    if [ "$st" = READY ]; then
      if [ "$crun" != true ]; then
        echo "  · 快照为 READY 但容器已不在运行（现实核实）⇒ 按 FAILED 处理"; st=FAILED
      else
        local _hc; _hc=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${cp:-$port}/v1/models" 2>/dev/null)
        [ "$_hc" = 200 ] || { echo "  · 快照为 READY 但接口不可用（HTTP=$_hc）⇒ 按 FAILED 处理"; st=FAILED; }
      fi
    elif [ "$st" = STARTING ] && [ "$crun" != true ]; then
      # 快照 STARTING、容器已退出：Worker 半途死了，属残留，不能一直挂着
      if ! pgrep -f "up[.]sh __worker $(lc_read_snapshot_field "${cp:-$port}" launch_id 2>/dev/null)" >/dev/null 2>&1; then
        echo "  · 快照为 STARTING 但容器已退出且无 Worker（现实核实）⇒ 按 FAILED 处理"; st=FAILED
      fi
    fi
    local ME EX dec act why
    ME=$(ct_identity "$mymid" "$myg" docker "$port")
    EX=$(ct_identity "${cmid:-none}" "$cg" "$cmode" "${cp:-0}")
    dec=$(ct_decide "$ME" "$EX" "$st" own); act=${dec%% *}; why=${dec#* }
    case "$act" in
      REUSE)  emit $ST_READY READY "同身份实例已就绪，复用（R27）" \
                "NAME=$c" "PORT=$port" "LAUNCH_ID=$clid" "REASON=$why"; CT_EXIT=$ST_READY; return 99;;
      ATTACH) emit $ST_STARTING STARTING "同身份实例正在拉起，挂接原 launch_id（R27）" \
                "NAME=$c" "PORT=$port" "LAUNCH_ID=$clid" "REASON=$why"; CT_EXIT=$ST_STARTING; return 99;;
      RECREATE)
              echo "  · FAILED 残留经标签+快照互证属本包，清理后新建 launch_id（R27）"
              $DOCKER rm -f "$c" >/dev/null 2>&1 || true;;
      MIGRATE_PORT)
              # ★ 两步，次序不能颠倒（R28.1）：
              #   第一步（上方）旧实例已按**它自己的端口**快照互证属本包；
              #   第二步在此独立体检**新目标端口**：任意容器占用、或宿主进程监听，
              #   两者只要有一个成立就失败关闭，**绝不停掉旧实例**——
              #   否则旧服务没了、新端口又起不来，等于把可用服务换成不可用。
              local _occ; _occ=$(ct_port_taken_by_ctn "$port" "$c" 2>/dev/null || true)
              if [ -n "$_occ" ] || ct_host_port_busy "$port"; then
                emit $ST_PORT_BUSY PORT_BUSY "换端口：新端口 $port 不可用，不停旧服务（R28.1）" \
                  "PORT=$port" "OLD=$c" "HOLDER=${_occ:-host_listener}" "REASON=$why"
                CT_EXIT=$ST_PORT_BUSY; return 99
              fi
              echo "  · 换端口（R28）：新端口 $port 两项体检均空闲，定点停旧 $c（launch_id=$CT_OWN_LID，旧端口 $CT_OWN_PORT）"
              $DOCKER rm -f "$c" >/dev/null 2>&1 || true;;
      FAIL)   local code=$ST_GPU_BUSY
              case "$why" in unknown_ownership) code=$ST_AMBIGUOUS;; esac
              emit $code "$(case $code in $ST_AMBIGUOUS) echo AMBIGUOUS;; *) echo GPU_BUSY;; esac)" \
                "合同判定失败（R26）" "HOLDER=$c" "REASON=$why"; CT_EXIT=$code; return 99;;
    esac
  done

  # 目标端口的宿主监听事实（修 #6）
  if ct_host_port_busy "$port"; then
    emit $ST_PORT_BUSY PORT_BUSY "目标端口 $port 已被宿主进程监听，归属无法证明（R32）" \
      "PORT=$port" "REASON=host_port_busy"; CT_EXIT=$ST_PORT_BUSY; return 99
  fi
  return 0
}

# ── 参数分发 ──
if [ -f "$HERE/.primary" ]; then
  # ★ 一线一包模式：本包只服务 .primary 指定的配置
  CFGNAME=$(cat "$HERE/.primary")
  NUM=$(echo "$CFGNAME" | cut -c1-2)
  # 容忍多余的编号前缀参数（up.sh 09 status 等价于 up.sh status）
  [ "${1:-}" = "$NUM" -o "${1:-}" = "$CFGNAME" ] && shift
  _action="${1:-up}"
  case "$_action" in *=*) [ "$CONTRACT_LOADED" = 1 ] && _action=start;; esac
  if [ "$CONTRACT_LOADED" != 1 ] && [[ "$_action" = start || "$_action" = up || "$_action" = menu ]]; then
    # 其余线继续执行原同步主流程，不派发 Worker，也不创建新合同快照。
    [ $# -eq 0 ] || shift
    _action=""
  fi
  case "$_action" in
    status) shift; do_status "$NUM" "$@"; exit 0;;
    stop)   shift; do_stop_safe "$@"; exit $?;;
    restart) shift; do_restart "$@"; exit 0;;
    # ★ logs/benchmark 必须同样 exit —— 少一个 exit 0 就会掉进下方拉起主流程，
    #   带着 PORT= 去做「幂等预清」，把该端口上的实例删掉再重拉。
    #   2026-09-17 实机验收就栽在这里：一个缺失的 exit 0 连锁导致 5 项失败。
    # logs) 已由下方支持 LAUNCH_ID= 的分支接管（case 先匹配先命中，旧分支须删）
    state)  shift; do_state "$@"; exit $?;;        # ★ 纯机器可读状态，无副作用
    benchmark|bench) shift; do_benchmark "$@"; exit $?;;
    # ── R24.1 四动作之 start：同步判定 + 快速返回（R29）──
    #   ★ 同步段**只做**规范化与判定，绝不执行 precheck/S1/deploy_patches/launch/S7/S8/smoke。
    start|up|menu)
      if [[ "${1:-}" = start || "${1:-}" = up || "${1:-}" = menu ]]; then shift; fi
      _iid=$(lc_new_id); _sp=""; _sg=""
      for _a in "$@"; do case "$_a" in PORT=*) _sp="${_a#PORT=}";; GPUS=*) _sg="${_a#GPUS=}";; esac; done
      _sp="${_sp:-$(cfg_port "$CFGNAME")}"
      _sg="${_sg:-$(meta "$HERE/$CFGNAME" gpus)}"
      export LC_GPUS="$_sg" LC_MODE=docker LC_LINE="${CFGNAME:0:2}"
      contract_resolve "$_sp" "$_sg" || exit ${CT_EXIT:-1}      # REUSE/ATTACH/FAIL 已回报并退出
      _lid=$(lc_new_id)
      # ★ 兜底（缺陷 5）：绝不覆盖一个**仍然活着**的生命周期的账本。
      #   快照是 stop/state 唯一的归属依据，被覆盖就等于把还在跑的服务变成无主资源。
      _pv=$(lc_read_snapshot_field "$_sp" launch_id 2>/dev/null)
      _pvs=$(lc_read_snapshot_field "$_sp" state 2>/dev/null)
      if [ -n "$_pv" ] && [ "$_pvs" = STARTING ] && lc_worker_alive "$_pv" "$HERE/up.sh"; then
        emit $ST_PORT_BUSY PORT_BUSY "端口 $_sp 上已有存活的生命周期，拒绝覆盖其账本" \
          "PORT=$_sp" "LAUNCH_ID=$_pv" "REASON=refuse_clobber_live_snapshot"
        exit $ST_PORT_BUSY
      fi
      lc_write_snapshot "$_sp" "$_lid" "$_iid" STARTING 20 start_dispatched "GPUS=$_sg"
      lc_emit_event "$_lid" "$_iid" "$_sp" STARTING 20 start_dispatched
      mkdir -p "$HERE/logs" 2>/dev/null || true
      WORKER_LOG="$HERE/logs/worker-$_lid.log"
      LAUNCH_ID="$_lid" WORKER_LOG="$WORKER_LOG" LC_STATE_DIR="${LC_STATE_DIR:-}" \
        setsid nohup bash "$HERE/up.sh" __worker "$_lid" "$_iid" "$_sp" "$_sg" "$@" \
        >>"$WORKER_LOG" 2>&1 </dev/null &
      lc_write_pid "$_lid" "$!"          # ★ 在 start 返回**之前**落盘，紧随其后的第二次 start 才挂得上
      emit $ST_STARTING STARTING "已触发后台启动，不等待就绪（R29）" \
        "PORT=$_sp" "GPUS=$_sg" "LAUNCH_ID=$_lid" "INVOCATION_ID=$_iid"
      exit $ST_STARTING;;
    # ── 隐藏入口：Worker 包装（非公开合同）──
    __worker) shift
      _wl="$1"; _wi="$2"; _wp="$3"; _wg="$4"; shift 4
      lc_write_pid "$_wl" "$$"                              # setsid 之后的真实 Worker PID，覆盖父进程记的那个
      trap 'lc_clear_pid "$_wl" "$$"' EXIT                  # 只清与自身 PID 一致的记录
      lc_worker_run "$CFGNAME" "$_wp" "$_wl" "$_wi" "$_wg" "$@"; exit $?;;
    # ── 隐藏入口：完整同步链（precheck→S1→patch→launch(S2-S8)→smoke）──
    #   只有 Worker 会调它。落入下方主流程。
    __run) shift;;
    # ── R30.4 logs 按 launch_id ──
    logs) shift
      _ll=""
      for _a in "$@"; do case "$_a" in LAUNCH_ID=*) _ll="${_a#LAUNCH_ID=}";; esac; done
      if [ -n "$_ll" ]; then
        _j=$(lc_jsonl "$_ll"); _w="$HERE/logs/worker-$_ll.log"; _got=0
        [ -f "$_j" ] && { echo "── JSONL 事件（launch_id=$_ll）──"; cat "$_j"; _got=1; }
        [ -f "$_w" ] && { echo "── Worker 原始日志 ──"; cat "$_w"; _got=1; }
        [ "$_got" = 1 ] || { echo "✗ 无该 launch_id 的日志: $_ll"; exit $ST_NOT_FOUND; }
        exit 0
      fi
      do_logs "$@"; exit 0;;
  esac
else
  ACT="${1:-menu}"
  case "$ACT" in
    menu|"") menu; exit 0;;
    status)  shift; do_status "$@"; exit 0;;
    stop)    shift; do_stop "$@"; exit 0;;
    restart) shift; do_restart "$@"; exit 0;;
    logs)    shift; do_logs "$@"; exit 0;;          # ★ 同上，必须 exit
    state)  shift; do_state "$@"; exit $?;;        # ★ 纯机器可读状态，无副作用
    benchmark|bench) shift; CFGNAME=""; do_benchmark "$@"; exit $?;;
  esac
  CFGNAME=$(cfg_name "$ACT"); [ -n "$CFGNAME" ] || die "找不到配置「$ACT」" "bash up.sh 看可用编号"
  NUM=$(echo "$CFGNAME" | cut -c1-2); shift
  case "${1:-up}" in
    status) shift; do_status "$NUM" "$@"; exit 0;;
    stop)   shift; do_stop "$NUM" "$@"; exit 0;;
    restart) shift; do_restart "$@"; exit 0;;
    logs)   shift; do_logs "$@"; exit 0;;          # ★ 同上，必须 exit
    state)  shift; do_state "$@"; exit $?;;        # ★ 纯机器可读状态，无副作用
    benchmark|bench) shift; do_benchmark "$@"; exit $?;;
  esac
fi

# ── 拉起主流程：体检 → S1 → S2..S8 → 冒烟（全程落 logs/，幂等）──
# ★ 先解析 KEY=VAL，使拉起日志名能带上实例句柄（端口）——N 个实例各自一份日志
#   同机起第二个实例时，幂等预清若按 @name 元数据会把第一个实例删掉（两实例元数据相同），
#   故 NAME=/PORT= 一律以命令行为准。2026-09-17 查实。
for _kv in "$@"; do
  case "$_kv" in
    NAME=*) OVR_NAME="${_kv#NAME=}" ;;
    PORT=*) PORT="${_kv#PORT=}" ;;
    # ★ GPUS 也必须在此解析成变量。它虽然会透传给 launch.sh，但本脚本自己要用：
    #   起服前的卡占用体检、以及 READY 状态里回报「实际用了哪几张卡」。
    #   2026-09-17 实机验收教训：漏解析时 ${GPUS:-} 恒空 ⇒ GPU_BUSY 整段成死代码，
    #   用已占用的卡起第二个实例不但没被挡下，还把该端口上的实例预清掉了，
    #   连锁造成 41 项里 9 项失败；READY 也只会报「GPUS=默认」。
    GPUS=*) GPUS="${_kv#GPUS=}" ;;
  esac
done
PORT="${PORT:-$(cfg_port "$CFGNAME")}"
RUNLOG="$HERE/logs/${NUM}-${PORT}-up-$(date +%m%d-%H%M%S).log"
ln -sf "$(basename "$RUNLOG")" "$HERE/logs/latest-${PORT}-up.log" 2>/dev/null || true
ln -sf "$(basename "$RUNLOG")" "$HERE/logs/latest-up.log" 2>/dev/null || true
exec > >(tee -a "$RUNLOG") 2>&1
echo "◇ 运行日志: $RUNLOG（实时跟随: tail -f $HERE/logs/latest-${PORT}-up.log）"
precheck
s1
deploy_patches          # ★ 若有补丁需重启，本函数内直接 exit 10 回报，不会往下走
say "③④ 拉起 $CFGNAME（S2 自动补镜像/权重，S3 门禁，S7 等就绪，S8 自证）"
# （KEY=VAL 与 PORT 已在 RUNLOG 之前解析，见上）
# ★ 只给了 PORT 没给 NAME 时，按端口派生实例名 —— 这样命令行里只需写「哪几张卡 + 哪个端口」，
#   同一个包即可在一台机器上起 N 个实例，无需改任何文件、也不必自己编名字。
if [ -z "${OVR_NAME:-}" ] && [ "$PORT" != "$(cfg_port "$CFGNAME")" ]; then
  _MN="$(meta "$HERE/$CFGNAME" name)"
  OVR_NAME="${PFX}${_MN}-${PORT}"
  echo "  · 未指定 NAME，按端口派生实例名: $OVR_NAME"
fi

# ★ 起服前的占用体检：端口与卡都可能已被**别的实例**占着。
#   不先查就起，会撞出难懂的运行期错误，故在此给出明确状态码。

_OTHER=""
for _n in $($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null); do
  [ "$_n" = "${OVR_NAME:-$(cfg_names "$CFGNAME")}" ] && continue    # 自己（幂等重拉）不算占用
  _op=$($DOCKER inspect "$_n" --format '{{join .Config.Cmd " "}}' 2>/dev/null \
        | grep -oE '\-\-port [0-9]+' | grep -oE '[0-9]+$')
  [ "$_op" = "$PORT" ] && die_st $ST_PORT_BUSY PORT_BUSY "端口 $PORT 已被实例 $_n 占用" \
      "PORT=$PORT" "HOLDER=$_n" "FIX=换端口，或先 bash up.sh stop PORT=$PORT"
  _og=$($DOCKER inspect "$_n" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
        | grep -oP 'HIP_VISIBLE_DEVICES=\K.*' | head -1)
  if [ -n "${GPUS:-}" ] && [ -n "$_og" ]; then
    for _g in ${GPUS//,/ }; do
      case ",$_og," in *",$_g,"*)
        die_st $ST_GPU_BUSY GPU_BUSY "卡 $_g 已被实例 $_n 占用（其卡组 $_og）" \
          "WANT=$GPUS" "HOLDER=$_n" "HOLDER_GPUS=$_og" "FIX=换卡组，或先停掉该实例";;
      esac
    done
  fi
done

# 幂等预清：只清"本次要起的这个实例名"，不碰同线的其它实例
if [ -n "${OVR_NAME:-}" ]; then
  _PRECLEAN="$OVR_NAME"
else
  _PRECLEAN="$(cfg_names "$CFGNAME") $(cfg_legacy "$CFGNAME")"
fi
# ★ 收紧（修 #6）：预清**只删经归属证明的容器**，不得按名字盲删。
#   原实现按名字一律 rm -f，会删掉恰好同名、但不属于本包的容器——
#   这与 R26 情形 5「未知占用失败关闭、不自动清理」直接冲突。
for n in $_PRECLEAN; do
  if $DOCKER inspect "$n" >/dev/null 2>&1; then
    if [ "${CONTRACT_LOADED:-0}" = 1 ] && ! ct_owned_proof "$n"; then
      die_st $ST_AMBIGUOUS AMBIGUOUS "同名容器 $n 归属无法证明，拒绝预清，需人工干预（R26 情形 5 / R32）" \
        "HOLDER=$n" "REASON=preclean_unknown_ownership" "DETAIL=${CT_OWN_WHY:-unknown}"
    fi
    $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  预清旧实例（归属已证明）: $n"
  fi
done
if [ "$NUM" = 02 ]; then
  PFX="$PFX" bash "$HERE/$CFGNAME/up.sh" || die "02 全套拉起失败" "看上方日志；单独排障可先跑 01"
else
  NAME="${OVR_NAME:-$(cfg_names "$CFGNAME")}" bash "$HERE/common/launch.sh" "$CFGNAME" "$@" \
    || die_st $ST_START_FAILED START_FAILED "$CFGNAME 拉起失败（S2-S8 哪步失败见上方日志）" \
         "PORT=$PORT" "GPUS=${GPUS:-默认}" "LOG=$RUNLOG" "FIX=按日志提示处理后重跑"
fi
smoke "$PORT" || die_st $ST_ATTEST_FAILED ATTEST_FAILED "服务已起但自证请求失败（S8）" \
  "PORT=$PORT" "LOG=$RUNLOG" "FIX=bash up.sh logs 看日志；冷启暖场后可重试一次"
echo
echo "════════════════════════════════════════════════════════"
echo "  ✓ $CFGNAME 就绪"
echo "    接口   http://<本机IP>:$PORT/v1   （OpenAI 兼容，model=qwen38）"
if [ -f "$HERE/.primary" ]; then
  echo "    状态   bash $HERE/up.sh status"
  echo "    停止   bash $HERE/up.sh stop"
else
  echo "    状态   bash $HERE/up.sh $NUM status"
  echo "    停止   bash $HERE/up.sh $NUM stop"
fi
echo "════════════════════════════════════════════════════════"
# ★ 机器可读收尾：拉起已成功，并回报本次实际使用的参数
STATE_PORT="$PORT" emit $ST_READY READY "拉起已成功" \
  "NAME=${OVR_NAME:-$(cfg_names "$CFGNAME")}" "PORT=$PORT" "GPUS=${GPUS:-默认}" \
  "CFG=$CFGNAME" "ENDPOINT=http://127.0.0.1:$PORT/v1" "LOG=$RUNLOG"
