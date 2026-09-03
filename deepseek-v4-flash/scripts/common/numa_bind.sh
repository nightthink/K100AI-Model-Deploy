#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
# ============================================================================
# 按参与的 GPU 列表推导 docker 的 NUMA 绑定参数（--cpuset-cpus / --cpuset-mems）
#
# 本机为 NPS4：8 个 NUMA node，每 node 12 核 / 64 GB。
#   卡 0-3 → PCIe numa_node=0（本地 cpus 0-11）  ，所属 socket0 = node 0-3, cpus 0-47
#   卡 4-7 → PCIe numa_node=4（本地 cpus 48-59），所属 socket1 = node 4-7, cpus 48-95
#
# 策略（默认 node 级，实测定）：
#   A/B 实测（不绑/socket级/node级）三组差异在噪声内，node 级两项指标略优且从不更差：
#     单流均值 48.7 / 50.0 / 50.9   16并发 257.0 / 260.9 / 269.3
#     prefill 三组几乎逐位相同——完全不受绑定影响
#   原担心"12 核会饿死 API/tokenizer 线程"未发生（vLLM CPU 侧压力小于预期）。
#   注：P2P 打通后 allreduce 走 GPU 直连不碰宿主内存，绑定收益本就有限，非必须项。
#   跨 socket（8 卡）时不绑 CPU，只提示应改用 TP4×PP2。
#
# 用法：  eval "$(bash numa_bind.sh 4,5,6,7)"     # 导出 NUMA_ARGS
#         $DOCKER run $NUMA_ARGS ...
#     或  bash numa_bind.sh 0,1,2,3 --print       # 只打印参数
# ============================================================================
set -u
GPUS="${1:-4,5,6,7}"
MODE="${2:-}"

has0=0; has1=0
IFS=',' read -ra ARR <<< "$GPUS"
for g in "${ARR[@]}"; do
    g=$(echo "$g" | tr -d ' ')
    if [ "$g" -le 3 ] 2>/dev/null; then has0=1; else has1=1; fi
done

if [ "$has0" -eq 1 ] && [ "$has1" -eq 1 ]; then
    # 跨 socket：不绑 CPU（绑了反而限制一半 rank），并提示架构建议
    NUMA_ARGS=""
    HINT="跨 socket（8 卡）：不做 CPU 绑定；强烈建议 TP4×PP2 而非 TP8（每层 allreduce 跨 socket 是灾难）"
elif [ "$has0" -eq 1 ]; then
    NUMA_ARGS="--cpuset-cpus=0-11 --cpuset-mems=0-3"
    HINT="socket0：CPU 绑 node0(0-11) 保局部性，内存放开到整 socket(0-3) 留余量"
else
    NUMA_ARGS="--cpuset-cpus=48-59 --cpuset-mems=4-7"
    HINT="socket1：CPU 绑 node4(48-59) 保局部性，内存放开到整 socket(4-7) 留余量"
fi

if [ "$MODE" = "--print" ]; then
    echo "GPUS=$GPUS  →  $HINT"
    echo "NUMA_ARGS=$NUMA_ARGS"
else
    echo "export NUMA_ARGS=\"$NUMA_ARGS\""
    echo "# $HINT" >&2
fi
