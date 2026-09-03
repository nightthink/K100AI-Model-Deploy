#!/usr/bin/env bash
# ============================================================================
# S1 · 机器级准备（L1）—— 见 docs/方案脚本规范-设计文档.md
#
# 取代原 acs_clear_guarded.sh：除清 ACS 外，还负责内核参数的**自愈**。
#
# 为什么需要「自愈」而不只是「设置」（2026-08-26 的教训）：
#   * kernel.numa_balancing 手工设过，8-21 重启后失效，无人发现，
#     当天全部测量在未知状态下进行
#   * acs-clear.service 报告成功退出，19 小时后 hycu 驱动把 ACS 又改回去了
#   → **设置会悄悄失效**。L1 必须在每次开机/驱动重载后重新收敛到既定状态，
#     而 L2 每次启动前还要再断言一次（stage3_gate）。
#
# 判据：ACS 必须在 hycu 绑定完全部 GPU **之后**才清
#   （不能用 KFD 拓扑节点数——早早就填好；也不能用 /dev/dri/renderD* 计数——
#     rmmod 后 udev 删得慢，会数到旧的）
#
# 用法:
#   sudo bash machine_prep.sh              # 等 GPU 就绪后收敛
#   sudo bash machine_prep.sh --now        # 不等待，立刻收敛（hycu 重载后用）
#   EXPECT_GPUS=8 WAIT_SECS=300 sudo -E bash machine_prep.sh
# ============================================================================
set -u
[ "$(id -u)" -eq 0 ] || { echo "需要 root：sudo bash $0"; exit 1; }
EXPECT_GPUS="${EXPECT_GPUS:-8}"; WAIT_SECS="${WAIT_SECS:-300}"
SETTLE="${SETTLE:-5}"; ROUNDS="${ROUNDS:-6}"
NOW=0; [ "${1:-}" = "--now" ] && NOW=1
log(){ echo "[machine-prep $(date +%H:%M:%S)] $*"; }
bound(){ ls -d /sys/bus/pci/drivers/hycu/0000:* 2>/dev/null | wc -l; }
residue(){ lspci -vvv -D 2>/dev/null | grep -c "ACSCtl:.*SrcValid+"; }

# ---------------------------------------------------------- 1) 内核参数自愈
log "内核参数"
for kv in "kernel.numa_balancing=0"; do
  k="${kv%%=*}"; want="${kv#*=}"
  cur=$(sysctl -n "$k" 2>/dev/null)
  if [ "$cur" = "$want" ]; then log "  ✓ $k=$cur"
  else sysctl -w "$kv" >/dev/null 2>&1 && log "  ✓ $k $cur → $want（已收敛）" \
       || log "  ✗ $k 设置失败（当前 $cur）"; fi
done
[ -f /etc/sysctl.d/99-dcu-llm.conf ] && log "  ✓ 声明式配置在位（重启后自动生效）" \
  || log "  ⚠ 缺 /etc/sysctl.d/99-dcu-llm.conf，重启后会丢"

# ---------------------------------------------------------- 2) 等 GPU 就绪
if [ "$NOW" -eq 0 ]; then
  log "等待 hycu 绑定 $EXPECT_GPUS 张卡（最多 ${WAIT_SECS}s）…"
  t0=$SECONDS
  while [ "$(bound)" -lt "$EXPECT_GPUS" ]; do
    [ $((SECONDS-t0)) -ge "$WAIT_SECS" ] && { log "  ⚠ 超时：只绑定 $(bound)/$EXPECT_GPUS"; break; }
    sleep 5
  done
  n=$(bound); [ "$n" -ge "$EXPECT_GPUS" ] && log "  ✓ 已绑定 $n 张，用时 $((SECONDS-t0))s"
  sleep "$SETTLE"
fi

# ---------------------------------------------------------- 3) ACS 收敛
before=$(residue); log "ACS：清除前 $before 个桥置位"
for i in $(seq 1 "$ROUNDS"); do
  /usr/local/sbin/acs_clear_all.sh >/dev/null 2>&1
  r=$(residue); log "  第 $i 轮后残留 $r"
  [ "$r" = "0" ] && { log "✓ 全部收敛（ACS $before→0）。P2P 已解锁"; exit 0; }
  sleep 5
done
log "✗ $ROUNDS 轮后仍有 $r 个残留 —— P2P 会被 IOMMU 重定向且不报错"
exit 1
