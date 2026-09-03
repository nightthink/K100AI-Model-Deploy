#!/bin/bash
# ============================================================================
# 载体适配层 —— 见 docs/方案脚本规范-设计文档.md §一·五
#
# 载体是**正交维度**：阶段定义契约，本文件决定契约在某种载体上怎么落地。
# 各阶段脚本只调用这里的 carrier_* 函数，**不允许直接写 docker**。
# 换成裸机 / systemd / k8s 时，只替换本文件。
#
# 用法： CARRIER=docker source lib/carrier.sh    （默认 docker）
# ============================================================================
CARRIER="${CARRIER:-docker}"

# ---- docker 载体 ----------------------------------------------------------
if [ "$CARRIER" = "docker" ]; then
  if docker info >/dev/null 2>&1; then _D="docker"; else _D="sudo docker"; fi
  carrier_name(){ echo "容器 $1"; }
  carrier_exists(){ $_D ps -a --format '{{.Names}}' | grep -qx "$1"; }
  carrier_alive(){  $_D ps    --format '{{.Names}}' | grep -qx "$1"; }
  carrier_restarts(){ $_D inspect -f '{{.RestartCount}}' "$1" 2>/dev/null || echo 0; }
  carrier_exitcode(){ $_D inspect -f '{{.State.ExitCode}}' "$1" 2>/dev/null || echo -1; }
  carrier_logs(){ $_D logs "$1" 2>&1 | tr '\r' '\n'; }
  # 本实例在宿主上的 PID 集合（用于算「哪些 KFD 上下文是我的」）
  carrier_pids(){ $_D inspect -f '{{.State.Pid}}' "$1" 2>/dev/null | grep -v '^0$'; }
  carrier_unmanage(){ $_D update --restart=no "$1" >/dev/null 2>&1 || true; }
  carrier_stop(){ $_D stop -t "${2:-60}" "$1" >/dev/null 2>&1 || true; }
  carrier_remove(){ $_D rm "$1" >/dev/null 2>&1 || true; }

# ---- 裸机载体（骨架；换 NVIDIA 直接起 vLLM 时用）--------------------------
elif [ "$CARRIER" = "bare" ]; then
  # 约定：实例标识 = PID 文件 $RUNDIR/<name>.pid，日志 = $RUNDIR/<name>.log
  RUNDIR="${RUNDIR:-/var/run/llm}"
  _pid(){ cat "$RUNDIR/$1.pid" 2>/dev/null; }
  carrier_name(){ echo "进程 $1"; }
  carrier_exists(){ [ -f "$RUNDIR/$1.pid" ]; }
  carrier_alive(){  p=$(_pid "$1"); [ -n "$p" ] && [ -d "/proc/$p" ]; }
  carrier_restarts(){ cat "$RUNDIR/$1.restarts" 2>/dev/null || echo 0; }  # 由 supervisor 维护
  carrier_exitcode(){ cat "$RUNDIR/$1.exitcode" 2>/dev/null || echo -1; }
  carrier_logs(){ cat "$RUNDIR/$1.log" 2>/dev/null; }
  carrier_pids(){ _pid "$1"; }
  carrier_unmanage(){ touch "$RUNDIR/$1.nomanage"; }   # 供 supervisor 识别，勿再拉起
  carrier_stop(){ p=$(_pid "$1"); [ -n "$p" ] && { kill -TERM "$p" 2>/dev/null
      for _ in $(seq 1 "${2:-60}"); do [ -d "/proc/$p" ] || break; sleep 1; done
      [ -d "/proc/$p" ] && kill -KILL "$p" 2>/dev/null; }; true; }
  carrier_remove(){ rm -f "$RUNDIR/$1.pid" "$RUNDIR/$1.nomanage"; }
else
  echo "未知载体 CARRIER=$CARRIER" >&2; exit 1
fi
