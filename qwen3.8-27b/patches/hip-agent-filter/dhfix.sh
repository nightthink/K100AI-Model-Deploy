#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
cd /b
echo "=== 截断 agent 列表能否救回来 ==="
echo "  基线（可见 0,1，无 hook）:"
HIP_VISIBLE_DEVICES=0,1 timeout 80 ./ipcrtt 0 1 1 50 2>&1 | grep -E '往返|不可见' | sed 's/^/    /'
echo "  可见 0,1,4，无 hook:"
HIP_VISIBLE_DEVICES=0,1,4 timeout 80 ./ipcrtt 0 1 1 50 2>&1 | grep -E '往返|不可见' | sed 's/^/    /'
for M in 1 2; do
  echo "  ★ 可见 0,1,4 + DLHOOK_MAXAG=$M:"
  DLHOOK_MAXAG=$M LD_PRELOAD=/b/dlhook.so HIP_VISIBLE_DEVICES=0,1,4 \
    timeout 80 ./ipcrtt 0 1 1 50 2>&1 | grep -E '往返|不可见|ERR' | sed 's/^/    /'
done
echo "  ★ 可见 0-7 + DLHOOK_MAXAG=2:"
DLHOOK_MAXAG=2 LD_PRELOAD=/b/dlhook.so HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  timeout 80 ./ipcrtt 0 1 1 50 2>&1 | grep -E '往返|不可见|ERR' | sed 's/^/    /'
echo DONE
