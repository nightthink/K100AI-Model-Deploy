#!/bin/bash
# v4 补丁 —— 打通跨 socket DMA，消除驱逐风暴
#
# 根因：跨 socket 走 PCI_P2PDMA_MAP_THRU_HOST_BRIDGE，而内核 drivers/pci/p2pdma.c
#       的 host_bridge_whitelist[] 只列了 Intel 根桥，海光不在其中
#       ⇒ pci_p2pdma_distance_many() 返回 -1（实测：跨 socket -1，同 socket 8）
#       ⇒ gpu_dma_buf_attach 清零 attach->peer2peer（实测 5057 次全为 0）
#       ⇒ gpu_dma_buf_map 给出 domains=GTT（实测 domains=6 出现 0 次）
#       ⇒ 每次 map 把导出方 BO 踢出 VRAM → 等驱逐 fence → 排驱逐
#       ⇒ restore_process_bos 重做 map → 再踢 → 活锁（VRAM⇄GTT 往返 5021/5006 次）
#
# 补丁：gpu_dma_buf_attach (gpu_dma_buf.c:250)
#   .text+0x2b3dd:  79 04  jns  +4     ← 仅当 distance>=0 才跳过清零
#                →  eb 04  jmp  +4     ← 一律跳过，peer2peer 保持为 1
#
# 前提：内核 iommu=pt（passthrough，无地址翻译），且实测 hipMemcpyPeer 跨 socket
#       5.16 GB/s > 主机两跳 3.86 GB/s ⇒ 硬件确实能一跳直达
#
# 用法: sudo bash patch_v4.sh [apply|revert|status]
set -e
KO=/usr/local/hyhal/dkms/hycu.ko
OFF=$((0xa0 + 0x2b3dd))          # = 0x2B47D
OLD="79"; NEW="eb"
ACT="${1:-status}"

cur () { xxd -p -s $OFF -l 1 "$KO"; }

case "$ACT" in
  status)
    printf "file offset 0x%X  当前字节 = %s  (79=原始 eb=v4已打)\n" $OFF "$(cur)"
    xxd -s $OFF -l 8 "$KO"
    ;;
  apply)
    C=$(cur)
    [ "$C" = "$NEW" ] && { echo "已是 v4，无需重复"; exit 0; }
    [ "$C" != "$OLD" ] && { echo "✗ 拒绝：期望 $OLD，实际 $C —— 偏移可能不对"; exit 1; }
    [ -f "$KO.v2" ] || cp -a "$KO" "$KO.v2"      # 备份 v2 态
    printf "\x$NEW" | dd of="$KO" bs=1 seek=$OFF count=1 conv=notrunc status=none
    echo "✓ v4 已写入：$(cur)"
    xxd -s $OFF -l 8 "$KO"
    echo "⚠ 必须重启后才能测量（rmmod/insmod 会破坏驱动，见第九轮）"
    ;;
  revert)
    C=$(cur)
    [ "$C" = "$OLD" ] && { echo "已是原始态"; exit 0; }
    printf "\x$OLD" | dd of="$KO" bs=1 seek=$OFF count=1 conv=notrunc status=none
    echo "✓ 已回退：$(cur)"
    ;;
esac
