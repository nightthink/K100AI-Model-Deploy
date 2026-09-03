#!/bin/bash
# ============================================================================
# ★ 拉起包统一入口（固定名称 up.sh）—— 放在包根目录，目录中立
#
#   bash up.sh                        # 自检 + 配置一览 + 推荐
#   bash up.sh 09                     # 拉起 09 号线（体检→S1→S2..S8→真实请求冒烟）
#   bash up.sh 10 GPUS=4,5,6,7        # 带参数拉起（KEY=VAL 透传给 launch.sh）
#   bash up.sh 09 status              # 观察该线状态
#   bash up.sh status                 # 观察全部
#   bash up.sh 09 stop                # 停止该线（S9：容器清场）
#   bash up.sh stop                   # 停止全部并报告 KFD 残留
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
mkdir -p "$HERE/logs" 2>/dev/null || true

cfg_dirs(){ ls -d "$HERE"/[0-9][0-9]-* 2>/dev/null; }
meta(){ grep -E "^# @$2[ \t]" "$1/serve.sh" 2>/dev/null | sed -E "s/^# @$2[ \t]+//" | head -1; }
cfg_name(){  # 编号/目录名 → 目录名
  local d; d=$(ls -d "$HERE/$1"-* 2>/dev/null | head -1)
  [ -z "$d" ] && d=$(ls -d "$HERE/$1" 2>/dev/null | head -1)
  [ -n "$d" ] && basename "$d"
}
cfg_port(){ local d="$HERE/$1"
  local p; p=$(meta "$d" port); echo "${p:-8101}"; }
cfg_names(){ local d="$HERE/$1"   # 该线管理的容器名列表（带标记前缀）
  local n; n=$(meta "$d" name); [ -n "$n" ] && echo "$PFX$n"; }
cfg_legacy(){ local d="$HERE/$1"  # 迁移期：无前缀旧名（老包/手工拉起的同线实例）
  meta "$d" name; }

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

do_status(){
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
  echo "  KFD 进程: $(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)    权重根: $MODELS_ROOT    标记前缀: $PFX"
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

do_logs(){  # 本线容器 docker 日志
  local n; n=$($DOCKER ps --format '{{.Names}}' --filter "name=$PFX" 2>/dev/null | head -1)
  [ -n "$n" ] || { echo "无在跑实例"; exit 1; }
  if [ "${1:-}" = "-f" ]; then exec $DOCKER logs -f --tail 100 "$n"
  else $DOCKER logs --tail 200 "$n" 2>&1 | cut -c1-200; fi
}

do_stop(){
  local d base num n
  for d in $(cfg_dirs); do
    base=$(basename "$d"); num=$(echo "$base" | cut -c1-2)
    [ -n "${1:-}" ] && [ "$num" != "$1" ] && continue
    for n in $(cfg_names "$base") $(cfg_legacy "$base"); do
      [ -z "$n" ] && continue
      if $DOCKER inspect "$n" >/dev/null 2>&1; then
        $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  已停 $n"
      fi
    done
  done
  # ★ 兜底：按标记前缀清扫（覆盖任何中间状态遗留，与上面按名清理互补）
  if [ -z "${1:-}" ]; then
    for n in $($DOCKER ps -aq --filter "name=$PFX" 2>/dev/null); do
      $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  已清残留(按标记) $n"
    done
  fi
  sleep 3
  echo "  S9 校验：KFD 进程 $(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)（全停后应为 0；单线停止时余数属其它在跑配置）"
}

precheck(){
  say "① 环境体检（硬性前提）"
  command -v docker >/dev/null || die "没有 docker 命令" "安装 docker 后重试"
  $DOCKER info >/dev/null 2>&1 || die "docker 不可用" "把用户加入 docker 组，或配置 sudo NOPASSWD"
  [ -e /dev/kfd ] || die "没有 /dev/kfd —— DCU 驱动未装载" "先装 DTK/驱动，本包不负责装驱动"
  NGPU=$(ls -d /sys/bus/pci/drivers/hycu/0000:* 2>/dev/null | wc -l)
  [ "$NGPU" -ge 4 ] || die "hycu 只绑定 $NGPU 张卡（需 ≥4）" "检查驱动与硬件"
  [ -d /opt/hyhal ] || die "缺 /opt/hyhal（容器必挂）" "从 DTK 安装或从已装机器拷贝"
  sudo -n true 2>/dev/null || [ "$(id -u)" = 0 ] || die "S1 需要 root" "以 root 运行或配置 sudo NOPASSWD"
  echo "  ✓ docker / 驱动($NGPU 卡) / hyhal / 提权"
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
p = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "用一句话自我介绍。"}],
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
    status) do_status "$NUM"; exit 0;;
    stop)   do_stop "$NUM"; exit 0;;
    logs)   shift; do_logs "${1:-}";;
    up|menu) set --;;                     # 无参/up → 拉起（KEY=VAL 参数自然透传）
  esac
else
  ACT="${1:-menu}"
  case "$ACT" in
    menu|"") menu; exit 0;;
    status)  do_status; exit 0;;
    stop)    do_stop; exit 0;;
    logs)    shift; do_logs "${1:-}";;
  esac
  CFGNAME=$(cfg_name "$ACT"); [ -n "$CFGNAME" ] || die "找不到配置「$ACT」" "bash up.sh 看可用编号"
  NUM=$(echo "$CFGNAME" | cut -c1-2); shift
  case "${1:-up}" in
    status) do_status "$NUM"; exit 0;;
    stop)   do_stop "$NUM"; exit 0;;
  esac
fi

# ── 拉起主流程：体检 → S1 → S2..S8 → 冒烟（全程落 logs/，幂等）──
RUNLOG="$HERE/logs/${NUM}-up-$(date +%m%d-%H%M%S).log"
ln -sf "$(basename "$RUNLOG")" "$HERE/logs/latest-up.log" 2>/dev/null || true
exec > >(tee -a "$RUNLOG") 2>&1
echo "◇ 运行日志: $RUNLOG（实时跟随: tail -f $HERE/logs/latest-up.log）"
precheck
s1
say "③④ 拉起 $CFGNAME（S2 自动补镜像/权重，S3 门禁，S7 等就绪，S8 自证）"
PORT=$(cfg_port "$CFGNAME")
# 幂等预清：本线的带标记实例与迁移期旧名实例先清场
for n in $(cfg_names "$CFGNAME") $(cfg_legacy "$CFGNAME"); do
  if $DOCKER inspect "$n" >/dev/null 2>&1; then
    $DOCKER rm -f "$n" >/dev/null 2>&1 && echo "  预清旧实例: $n"
  fi
done
if false; then
  :
else
  NAME="$(cfg_names "$CFGNAME")" bash "$HERE/common/launch.sh" "$CFGNAME" "$@" \
    || die "$CFGNAME 拉起失败（S2-S8 哪步失败见上方日志）" "按日志提示处理后重跑"
fi
smoke "$PORT" || die "服务已起但请求失败" "bash up.sh logs 看日志；稍候重试一次（冷启暖场）"
echo
echo "════════════════════════════════════════════════════════"
echo "  ✓ $CFGNAME 就绪"
echo "    接口   http://<本机IP>:$PORT/v1   （OpenAI 兼容，model=deepseek-v4-flash）"
if [ -f "$HERE/.primary" ]; then
  echo "    状态   bash $HERE/up.sh status"
  echo "    停止   bash $HERE/up.sh stop"
else
  echo "    状态   bash $HERE/up.sh $NUM status"
  echo "    停止   bash $HERE/up.sh $NUM stop"
fi
echo "════════════════════════════════════════════════════════"
