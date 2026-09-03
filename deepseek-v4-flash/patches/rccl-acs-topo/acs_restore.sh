#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
# 回滚：按最新备份写回原值。用法: sudo bash acs_restore.sh [备份文件]
set -u
F=${1:-$(ls -t /root/acs_backup_*.txt | head -1)}
echo "从 $F 恢复"
while read -r dev val; do setpci -s "$dev" ECAP_ACS+6.w="$val"; done < "$F"
echo "已恢复 $(wc -l < "$F") 端口"
