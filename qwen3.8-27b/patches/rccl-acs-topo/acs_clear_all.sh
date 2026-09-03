#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
# ============================================================================
# 清除全机 PCIe 桥的 ACS 控制位（两个 socket 全覆盖，卡 0-7 任意组合均受益）
#
# 背景：BIOS 默认 ACS 全开会把经根桥的 P2P TLP 强制上送 IOMMU 重定向，
#       导致每次 P2P 操作 623ms 固定停顿，RCCL 只能退回 SHM（3.5 GB/s）。
#       清除后 all-reduce 从 3.5 → 26.4 GB/s（实测卡 4-7）。
#
# 安全性：机器已 iommu=pt 且这些卡不做 VFIO 直通，关 ACS 代价可接受。
#        每次运行都会备份原值；重启后 BIOS 默认值自动恢复。
#
# 用法： sudo bash acs_clear_all.sh            # 清除
#        sudo bash acs_clear_all.sh --dry-run  # 只看不改
#        sudo bash acs_restore.sh [备份文件]    # 回滚
# ============================================================================
set -u

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ "$(id -u)" -eq 0 ] || { echo "需要 root：sudo bash $0"; exit 1; }

BACKUP=/root/acs_backup_$(date +%Y%m%d-%H%M%S).txt
: > "$BACKUP"

total=0; had_acs=0; cleared=0; failed=0

# 遍历全部 PCI 桥（两个 socket），只处理真正带 ACS 能力的
for dev in $(lspci -D | awk '/PCI bridge/ {print $1}'); do
    total=$((total + 1))
    cur=$(setpci -s "$dev" ECAP_ACS+6.w 2>/dev/null) || continue   # 无 ACS 能力
    had_acs=$((had_acs + 1))
    echo "$dev $cur" >> "$BACKUP"
    [ "$cur" = "0000" ] && continue
    if [ "$DRY" -eq 1 ]; then
        echo "  [dry-run] $dev $cur -> 0000"
        cleared=$((cleared + 1))
    elif setpci -s "$dev" ECAP_ACS+6.w=0000 2>/dev/null; then
        new=$(setpci -s "$dev" ECAP_ACS+6.w 2>/dev/null)
        if [ "$new" = "0000" ]; then
            cleared=$((cleared + 1))
        else
            failed=$((failed + 1)); echo "  ⚠ $dev 写入未生效: $cur -> $new"
        fi
    else
        failed=$((failed + 1)); echo "  ⚠ $dev 写入失败"
    fi
done

echo "桥总数 $total，带 ACS 能力 $had_acs，本次清零 $cleared，失败 $failed"
echo "备份: $BACKUP"

# 复核：按 socket 统计仍处于 SrcValid+ 的端口
echo "--- 复核（SrcValid+ 残留，应为 0）---"
lspci -vvv -D 2>/dev/null |
awk '/^0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]/{d=$1}
     /ACSCtl:.*SrcValid\+/{split(d,p,":"); print (p[2]<"80" ? "socket0" : "socket1")}' |
sort | uniq -c | awk '{printf "  %s 残留 %s 个\n", $2, $1}'
echo "  （无输出=全部清零）"
