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

do_state(){
  # 只读地判定当前状态，不产生任何副作用。多实例用 PORT= 指定。
  local want="" n="" p=""
  for _a in "$@"; do case "$_a" in PORT=*) want="${_a#PORT=}";; NAME=*) n="${_a#NAME=}";; esac; done

  # ① 主机前提（最先判，因为后面的判断都建立在它之上）
  command -v docker >/dev/null || { emit $ST_NO_DOCKER NO_DOCKER "没有 docker 命令"; return $ST_NO_DOCKER; }
  $DOCKER info >/dev/null 2>&1 || { emit $ST_NO_DOCKER NO_DOCKER "docker 不可用（权限或服务未起）"; return $ST_NO_DOCKER; }
  { [ -e /dev/kfd ] && [ -d /opt/hyhal ]; } || {
    emit $ST_NO_DRIVER NO_DRIVER "驱动未安装（缺 /dev/kfd 或 /opt/hyhal）——本包不负责装驱动"
    return $ST_NO_DRIVER; }

  # ② 等待重启优先于一切服务状态：未重启时补丁没生效，起来了也不算数
  [ -f "$REBOOT_FLAG" ] && {
    emit $ST_NEED_REBOOT NEED_REBOOT "补丁已部署，等待主机重启" \
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
      n=$(_by_port "$want") || { emit $ST_NOT_FOUND NOT_FOUND "端口 $want 上没有本流程的实例" "PORT=$want"
                                 return $ST_NOT_FOUND; }
    else
      # ★ 歧义判定只数**在跑**的实例。用 ps -a 会把早已退出的旧容器算进来，
      #   机上若留有两具「尸体」，无实例时会被误报成 AMBIGUOUS 而非 STOPPED。
      #   （按端口查单个实例仍用 _by_port/ps -a，那里需要能查到已停的实例。）
      local all cnt; all=$($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null)
      cnt=$(printf '%s\n' "$all" | grep -c . || true)
      if [ "$cnt" = 0 ]; then
        emit $ST_STOPPED STOPPED "无实例，服务已停止"
        return $ST_STOPPED
      elif [ "$cnt" -gt 1 ]; then
        emit $ST_AMBIGUOUS AMBIGUOUS "同机有 $cnt 个实例，须用 PORT= 指定要看哪一个" \
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
    emit $ST_STOPPED STOPPED "实例已停止" "NAME=$n" "PORT=${p:-?}" "EXIT_CODE=${ec:-?}"
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
    [ "$bad" = 1 ] && { emit $ST_SERVICE_ERROR SERVICE_ERROR "服务异常，表现为:$sym" \
                          "NAME=$n" "PORT=$p" "HTTP=200"; return $ST_SERVICE_ERROR; }
    # ⑥·5 压测判定：只认**打在本实例端口上**的 benchmark 进程（R13.6）
    local _bp _bpid=""
    for _bp in $(pgrep -f 'standard_load\.py' 2>/dev/null); do
      tr '\0' ' ' < "/proc/$_bp/cmdline" 2>/dev/null \
        | grep -qE -- "--port[= ]$p( |$)" && { _bpid="$_bp"; break; }
    done
    if [ -n "$_bpid" ]; then
      emit $ST_BENCH_RUNNING BENCH_RUNNING "正在 benchmark，仍未结束" \
        "NAME=$n" "PORT=$p" "BENCH_PID=$_bpid"
      return $ST_BENCH_RUNNING
    fi
    local g; g=$($DOCKER inspect "$n" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
                 | grep -oP 'HIP_VISIBLE_DEVICES=\K.*' | head -1)
    emit $ST_READY READY "服务正常，等待命令中" "NAME=$n" "PORT=$p" "GPUS=${g:-?}" \
      "UPTIME=$($DOCKER inspect "$n" --format '{{.State.StartedAt}}' 2>/dev/null)"
    return $ST_READY
  fi

  if [ "$bad" = 1 ]; then
    emit $ST_STARTING_BAD STARTING_BAD "正在拉起但状态异常:$sym" "NAME=$n" "PORT=$p" "HTTP=$http"
    return $ST_STARTING_BAD
  fi
  emit $ST_STARTING STARTING "正在拉起，状态正常，仍在等待就绪" "NAME=$n" "PORT=$p" "HTTP=$http" \
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
  EXPECT_GPUS="$NGPU" $SUDO -E bash "/usr/local/sbin/${PFX}machine_prep.sh" --now | sed 's/^/  /'
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

# ── 参数分发 ──
if [ -f "$HERE/.primary" ]; then
  # ★ 一线一包模式：本包只服务 .primary 指定的配置
  CFGNAME=$(cat "$HERE/.primary")
  NUM=$(echo "$CFGNAME" | cut -c1-2)
  # 容忍多余的编号前缀参数（up.sh 09 status 等价于 up.sh status）
  [ "${1:-}" = "$NUM" -o "${1:-}" = "$CFGNAME" ] && shift
  case "${1:-up}" in
    status) shift; do_status "$NUM" "$@"; exit 0;;
    stop)   shift; do_stop "$NUM" "$@"; exit 0;;
    restart) shift; do_restart "$@"; exit 0;;
    # ★ logs/benchmark 必须同样 exit —— 少一个 exit 0 就会掉进下方拉起主流程，
    #   带着 PORT= 去做「幂等预清」，把该端口上的实例删掉再重拉。
    #   2026-09-17 实机验收就栽在这里：一个缺失的 exit 0 连锁导致 5 项失败。
    logs)   shift; do_logs "$@"; exit 0;;
    state)  shift; do_state "$@"; exit $?;;        # ★ 纯机器可读状态，无副作用
    benchmark|bench) shift; do_benchmark "$@"; exit $?;;
    up|menu) set --;;                     # 无参/up → 拉起（KEY=VAL 参数自然透传）
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
for n in $_PRECLEAN; do
  if $DOCKER inspect "$n" >/dev/null 2>&1; then
    $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  预清旧实例: $n"
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
