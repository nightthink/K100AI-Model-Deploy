#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
# ============================================================================
# ACS 清除 —— 以「hycu 已绑定全部 GPU」为闸
#
# 为什么需要它（2026-08-25 在 验证机B 上定位）：
#   hycu 驱动加载时会把所有 PCIe 桥的 ACS 重新打开（ACSCtl=0x1D，即
#   SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+）。谁后跑谁说了算。
#
#   验证机B  开机 15:15:43 → acs-clear 完成 15:15:57 → hycu 加载 15:19:00
#             清除跑在驱动之前，被驱动改回，之后无人重跑 → 22 个桥带 ReqRedir+
#             卡的直接上级桥全部中招，任意一对卡的 P2P 都不成立
#   验证机A  hycu 16:55:35 → acs-clear 16:55:41（手工重跑）→ 保住了
#
#   注意 acs_clear_retry.sh 治的是另一个病（开机时 PCIe 桥尚未枚举完），
#   它最多重试 6 轮 × 5 秒 = 30 秒，等不到 3 分钟后才加载的 hycu。
#
# 判据：/sys/bus/pci/drivers/hycu/0000:* 的数量 == EXPECT_GPUS。
#   不能用 KFD 拓扑节点数（早早就填好），也不能用 /dev/dri/renderD* 计数
#   （rmmod 后 udev 删得慢，会数到旧的）——见 CLAUDE.md 护栏第 10 条。
#
# 用法:
#   sudo bash acs_clear_guarded.sh              # 等 GPU 就绪后清除
#   EXPECT_GPUS=8 WAIT_SECS=300 sudo -E bash acs_clear_guarded.sh
#   sudo bash acs_clear_guarded.sh --now        # 不等待，立刻清（hycu 重载后用）
#
# 幂等，任何时候都能重跑。**每次 rmmod/insmod hycu 之后都必须重跑。**
# ============================================================================
set -u
[ "$(id -u)" -eq 0 ] || { echo "需要 root：sudo bash $0"; exit 1; }

EXPECT_GPUS="${EXPECT_GPUS:-8}"
WAIT_SECS="${WAIT_SECS:-300}"
SETTLE="${SETTLE:-5}"
ROUNDS="${ROUNDS:-6}"
NOW=0; [ "${1:-}" = "--now" ] && NOW=1

log(){ echo "[acs-guard $(date +%H:%M:%S)] $*"; }

bound(){ ls -d /sys/bus/pci/drivers/hycu/0000:* 2>/dev/null | wc -l; }
residue(){ lspci -vvv -D 2>/dev/null | grep -c "ACSCtl:.*SrcValid+"; }

# ---- 闸：等 hycu 绑定全部 GPU ----
if [ "$NOW" -eq 0 ]; then
  log "等待 hycu 绑定 $EXPECT_GPUS 张卡（最多 ${WAIT_SECS}s）…"
  t0=$SECONDS
  while [ "$(bound)" -lt "$EXPECT_GPUS" ]; do
    if [ $((SECONDS - t0)) -ge "$WAIT_SECS" ]; then
      log "⚠ 超时：只绑定了 $(bound)/$EXPECT_GPUS 张。仍尝试清除，但之后 hycu"
      log "⚠ 再加载会把 ACS 改回去 —— 届时需手工重跑本脚本（--now）"
      break
    fi
    sleep 5
  done
  n=$(bound)
  [ "$n" -ge "$EXPECT_GPUS" ] && log "hycu 已绑定 $n 张卡，用时 $((SECONDS - t0))s"
  log "再等 ${SETTLE}s 让枚举稳定"
  sleep "$SETTLE"
fi

# ---- 清除 + 复核，直到残留为 0 ----
before=$(residue)
log "清除前 ACSCtl 置位的桥：$before 个"
for i in $(seq 1 "$ROUNDS"); do
  /usr/local/sbin/acs_clear_all.sh >/dev/null 2>&1
  r=$(residue)
  log "第 $i 轮后残留 $r 个"
  if [ "$r" = "0" ]; then
    log "✓ 全部清零（$before → 0，共 $i 轮）。P2P 已解锁"
    exit 0
  fi
  sleep 5
done

log "✗ $ROUNDS 轮后仍有 $r 个残留。P2P 会被 IOMMU 重定向拖慢且不报错"
log "  逐个排查： lspci -vvv -D | grep -B40 'ACSCtl:.*SrcValid+' | grep '^0000'"
exit 1
