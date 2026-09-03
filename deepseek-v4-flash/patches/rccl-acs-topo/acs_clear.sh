#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
# 清 socket-1 树（总线 80-96，卡 4-7）全部桥端口的 ACS 控制位；先备份可回滚
set -u
BACKUP=/root/acs_backup_$(date +%Y%m%d-%H%M%S).txt
CLEARED=0
for dev in $(lspci -D | awk '/PCI bridge/ {print $1}' | grep -E '^0000:(8[0-9a-f]|9[0-6])'); do
  cur=$(setpci -s "$dev" ECAP_ACS+6.w 2>/dev/null) || continue
  echo "$dev $cur" >> "$BACKUP"
  if [ "$cur" != "0000" ]; then
    setpci -s "$dev" ECAP_ACS+6.w=0000 && CLEARED=$((CLEARED+1))
  fi
done
echo "备份: $BACKUP ($(wc -l < "$BACKUP") 端口)，清零 $CLEARED 个"
echo "--- 末级端口复核（应全为 SrcValid-） ---"
for b in 86 89 8e 95; do
  echo -n "$b:00.0  "; lspci -vvv -s "$b:00.0" 2>/dev/null | grep -m1 -oE 'ACSCtl:.*'
done
echo -n "socket1 树残留 SrcValid+ 端口数: "
lspci -vvv -D 2>/dev/null | awk '/^0000:(8[0-9a-f]|9[0-6])/{d=$1} /ACSCtl:.*SrcValid\+/{if (d!="") print d}' | sort -u | wc -l
