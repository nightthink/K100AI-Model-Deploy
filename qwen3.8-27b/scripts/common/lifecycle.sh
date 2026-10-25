#!/bin/bash
# 生命周期运行时（R29–R31）—— 事件、快照、Callback、后台 Worker
#
# 与 contract.sh 的分工：contract.sh 是纯判定逻辑（无副作用、可直接测）；
# 本文件负责「把判定结果落成事实」——写 JSONL 事件、原子写状态快照、派发 Callback、
# 以及以后台 Worker 形式执行 S1–S8 同步链。
#
# ★ 不含任何「跳过驱动/设备检查」的开关。测试通过 PATH 注入伪 docker/curl 验证行为。

LC_DIR_DEFAULT_REL="state"

lc_dir(){ echo "${LC_STATE_DIR:-$HERE/$LC_DIR_DEFAULT_REL}"; }
lc_snap(){ echo "$(lc_dir)/snapshot-${1}.json"; }       # $1=port
lc_jsonl(){ echo "$(lc_dir)/events-${1}.jsonl"; }       # $1=launch_id

lc_new_id(){ # 唯一 id：时间+随机，不依赖外部工具
  printf '%s-%s' "$(date +%Y%m%d%H%M%S)" "$(od -An -N4 -tx4 /dev/urandom 2>/dev/null | tr -d ' \n' || echo $$)"
}

lc_esc(){ printf '%s' "${1:-}" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/ /g; s/\r//g' | tr -d '\n'; }

# R30.3 原子写快照：先写临时文件再 mv，避免 state 读到半截
lc_write_snapshot(){ # port launch_id invocation_id state code reason [k=v...]
  local port="$1" lid="$2" iid="$3" st="$4" code="$5" reason="$6"; shift 6
  local d; d=$(lc_dir); mkdir -p "$d" 2>/dev/null || true
  local f; f=$(lc_snap "$port"); local tmp="$f.tmp.$$"
  local extra="" kv
  for kv in "$@"; do extra="$extra,\"${kv%%=*}\":\"$(lc_esc "${kv#*=}")\""; done
  cat > "$tmp" <<J
{"schema_version":"1","port":"$(lc_esc "$port")","launch_id":"$(lc_esc "$lid")","invocation_id":"$(lc_esc "$iid")","state":"$(lc_esc "$st")","code":$code,"reason":"$(lc_esc "$reason")","observed_at":"$(date -Is)"$extra}
J
  mv -f "$tmp" "$f"
}

# ════ Worker PID 记账（R29）——「STARTING 但还没建容器」时证明生命周期确实在跑 ════
#   只信快照会把夭折的生命周期永远挂住；只 kill -0 PID 又会被 PID 复用或无关进程骗过。
#   故记 PID 并在核验时读 /proc/<pid>/cmdline，要求它确实是**本包 up.sh 的 __worker <launch_id>**。
lc_pidfile(){ echo "$(lc_dir)/worker-${1}.pid"; }        # $1=launch_id

lc_write_pid(){ # launch_id pid —— 原子写
  local d; d=$(lc_dir); mkdir -p "$d" 2>/dev/null || true
  local f; f=$(lc_pidfile "$1"); printf '%s\n' "$2" > "$f.tmp.$$" && mv -f "$f.tmp.$$" "$f"
}

# 退出时只清理「与自己 PID 一致」的记录，绝不误删别人的
lc_clear_pid(){ # launch_id pid
  local f; f=$(lc_pidfile "$1")
  [ -f "$f" ] || return 0
  [ "$(head -1 "$f" 2>/dev/null)" = "$2" ] && rm -f "$f"
  return 0
}

# $1=launch_id [$2=本包 up.sh 路径] → 0 = 该 launch_id 的 Worker 确实活着
lc_worker_alive(){
  local lid="$1" up="${2:-$HERE/up.sh}" pid cl
  _lc_cmdline_ok(){ # $1=pid —— cmdline 必须同时含本包 up.sh、__worker 与该 launch_id
    local c; c=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
    case "$c" in *"$up"*) ;; *) return 1;; esac
    case "$c" in *__worker*) ;; *) return 1;; esac
    case "$c" in *"$lid"*) return 0;; *) return 1;; esac
  }
  pid=$(head -1 "$(lc_pidfile "$lid")" 2>/dev/null)
  case "$pid" in ''|*[!0-9]*) pid="";; esac
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && _lc_cmdline_ok "$pid"; then return 0; fi
  # 兜底：setsid 会再 fork 一层，父进程记的 PID 可能不是最终 Worker；按 cmdline 全表扫一遍
  for cl in /proc/[0-9]*; do
    pid=${cl#/proc/}
    _lc_cmdline_ok "$pid" && { lc_write_pid "$lid" "$pid"; return 0; }
  done
  return 1
}

# 只终止经既有 PID/命令行证明、且确由 setsid 隔离的 Worker 进程组。
# 不升级为强杀；有界等待后仍不能确认退出，就由入口失败关闭。
lc_stop_worker(){ # launch_id state → 0 已无活动启动链，非 0 不得回报 STOPPED
  local lid="$1" state="${2:-}" pid pg sid i status
  LC_STOP_REASON=""
  pid=$(head -1 "$(lc_pidfile "$lid")" 2>/dev/null)
  if ! lc_worker_alive "$lid" "$HERE/up.sh"; then
    case "$pid" in
      ''|*[!0-9]*) ;;
      *) if kill -0 "$pid" 2>/dev/null; then LC_STOP_REASON=worker_identity_unknown; return 1; fi;;
    esac
    # Worker 退出或 PID 记录消失，不证明首次 TERM 后的子链已退出。
    # 未确认的 STARTING 重试继续交人工，不能升级为 STOPPED。
    if [ "$state" = STARTING ]; then LC_STOP_REASON=worker_exit_unproved; return 1; fi
    return 0
  fi
  pid=$(head -1 "$(lc_pidfile "$lid")" 2>/dev/null)
  read -r pg sid < <(ps -p "$pid" -o pgid=,sid=)
  if [ "$pg" != "$pid" ] || [ "$sid" != "$pid" ]; then
    LC_STOP_REASON=worker_not_isolated; return 1
  fi
  _lc_group_running(){
    local processes; processes=$(ps -eo pgid=,stat=) || return 2
    printf '%s\n' "$processes" | awk -v group="$1" '$1 == group && $2 !~ /^Z/ {live=1} END {exit !live}'
  }
  kill -TERM -- "-$pid" 2>/dev/null || {
    _lc_group_running "$pid"; status=$?
    [ "$status" = 1 ] || { LC_STOP_REASON=worker_signal_failed; return 1; }
  }
  for ((i=0; i<50; i++)); do
    _lc_group_running "$pid"; status=$?
    case "$status" in
      1) lc_clear_pid "$lid" "$pid"; return 0;;
      0) ;;
      *) LC_STOP_REASON=worker_exit_unproved; return 1;;
    esac
    sleep 0.1
  done
  LC_STOP_REASON=worker_stop_incomplete
  return 1
}

lc_read_snapshot_field(){ # port field
  local f; f=$(lc_snap "$1"); [ -f "$f" ] || return 1
  sed -n "s/.*\"$2\":\"\([^\"]*\)\".*/\1/p" "$f" | head -1
}

# R30.3 JSONL 事件：sequence 单调递增（按 launch_id 计数）
# #8 仅状态变化才回调：记录每个 launch 的上一状态
lc_last_state_file(){ echo "$(lc_dir)/laststate-${1}"; }

lc_emit_event(){ # launch_id invocation_id port state code reason
  local lid="$1" iid="$2" port="$3" st="$4" code="$5" reason="$6"
  local d; d=$(lc_dir); mkdir -p "$d" 2>/dev/null || true
  local f; f=$(lc_jsonl "$lid")
  local seq=1; [ -f "$f" ] && seq=$(( $(wc -l < "$f") + 1 ))
  printf '{"schema_version":"1","invocation_id":"%s","launch_id":"%s","state":"%s","code":%s,"reason":"%s","sequence":%s,"observed_at":"%s","identity":{"port":"%s","gpus":"%s","launch_mode":"%s","line":"%s"}}\n' \
    "$(lc_esc "$iid")" "$(lc_esc "$lid")" "$(lc_esc "$st")" "$code" "$(lc_esc "$reason")" \
    "$seq" "$(date -Is)" "$(lc_esc "$port")" "$(lc_esc "${LC_GPUS:-}")" "$(lc_esc "${LC_MODE:-docker}")" "$(lc_esc "${LC_LINE:-}")" \
    >> "$f"
  # ★ #8：同状态不重复回调。start 写一次 STARTING，Worker 不得再发重复 STARTING；
  #   READY/FAILED/NEED_REBOOT 各转换一次。
  local lsf; lsf=$(lc_last_state_file "$lid")
  local prev=""; [ -f "$lsf" ] && prev=$(cat "$lsf" 2>/dev/null)
  if [ "$prev" != "$st" ]; then
    printf '%s' "$st" > "$lsf" 2>/dev/null || true
    lc_callback "$f" "$seq"
  fi
}

# R31 Callback：未配置零调用；有界超时；失败不改主结果；不重试
#   ★ 事件不含凭据、主机名/IP、模型绝对路径原文（见 lc_emit_event 的字段集）
lc_callback(){ # jsonl_file seq
  local cb="${LAUNCH_CALLBACK:-}"; [ -z "$cb" ] && return 0
  local ev; ev=$(tail -1 "$1" 2>/dev/null); [ -z "$ev" ] && return 0
  local to="${LAUNCH_CALLBACK_TIMEOUT:-5}"
  case "$cb" in
    http://*|https://*)
      curl -s -m "$to" -X POST -H 'Content-Type: application/json' -d "$ev" "$cb" >/dev/null 2>&1 || true ;;
    *)
      [ -x "$cb" ] || [ -f "$cb" ] || return 0
      printf '%s' "$ev" | timeout "$to" bash "$cb" >/dev/null 2>&1 || true ;;
  esac
  return 0    # ★ 永远不改变主结果
}

# R29 后台 Worker：执行 S1–S8 同步链，终态由真实结果写入
#   非公开动作——由 start 内部以 __worker 调起，不进四动作合同。
lc_worker_run(){ # cfgname port launch_id invocation_id gpus [extra args...]
  local cfg="$1" port="$2" lid="$3" iid="$4" gpus="$5"; shift 5
  export LC_GPUS="$gpus" LC_MODE=docker LC_LINE="${cfg:0:2}" LAUNCH_ID="$lid"
  # ★ #3：走隐藏 __run —— 它执行 precheck→S1→deploy_patches→launch.sh(S2-S8)→smoke
  #   完整旧链路。此前只调 launch.sh 却声称执行 S1-S8，是不实陈述。
  # ★ #8：此处**不再**发 STARTING —— start 已发过一次，重复发会触发重复回调。
  local rc=0
  bash "${LC_UP_SH:-$HERE/up.sh}" __run "$@" >>"${WORKER_LOG:-/dev/null}" 2>&1 || rc=$?
  # 终态由真实退出码与 .reboot-required 决定
  if [ -f "${HERE:-.}/.reboot-required" ] || [ "$rc" = 10 ]; then
    lc_write_snapshot "$port" "$lid" "$iid" NEED_REBOOT 10 patch_deployed_reboot_required "GPUS=$gpus"
    lc_emit_event "$lid" "$iid" "$port" NEED_REBOOT 10 patch_deployed_reboot_required
    return 10
  fi
  if [ "$rc" = 0 ]; then
    lc_write_snapshot "$port" "$lid" "$iid" READY 0 worker_ready "GPUS=$gpus"
    lc_emit_event "$lid" "$iid" "$port" READY 0 worker_ready
  else
    lc_write_snapshot "$port" "$lid" "$iid" FAILED 22 "worker_rc_$rc" "GPUS=$gpus"
    lc_emit_event "$lid" "$iid" "$port" FAILED 22 "worker_rc_$rc"
  fi
  return "$rc"
}
