#!/bin/bash
# ============================================================================
# 起服务前的 ACS 体检：确认目标 socket 的 P2P 通路没有被 ACS 挡住。
#
# 为什么必须查：终版配置启用了 custom all-reduce（走 HIP IPC）。若 ACS 仍开着，
# 经根桥的 P2P TLP 会被强制送去 IOMMU 重定向，每次固定停顿 623ms
# —— 服务能正常起来、日志无任何报错，但解码降到 ~0.05 tok/s（20 秒一个 token）。
# 这种"起得来但慢 1000 倍"的失败极难当场判断，所以宁可拦在启动前。
#
# 用法：  bash preflight_acs.sh 4,5,6,7     # 返回 0 放行，非 0 拦截
#         SKIP_ACS_CHECK=1 …               # 逃生开关（明知故犯时用）
# ============================================================================
set -u
GPUS="${1:-4,5,6,7}"

[ "${SKIP_ACS_CHECK:-0}" = "1" ] && { echo "[ACS] 已按 SKIP_ACS_CHECK=1 跳过体检"; exit 0; }

# 目标 GPU 落在哪些 socket（PCI 段 <0x80 为 socket0，>=0x80 为 socket1）
want0=0; want1=0
IFS=',' read -ra A <<< "$GPUS"
for g in "${A[@]}"; do
    g="${g// /}"
    if [ "$g" -le 3 ] 2>/dev/null; then want0=1; else want1=1; fi
done

residual() {   # 打印指定 socket 上仍处 SrcValid+ 的桥数量
    sudo lspci -vvv -D 2>/dev/null |
    awk -v w0="$want0" -v w1="$want1" '
        /^0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]/ { d=$1 }
        /ACSCtl:.*SrcValid\+/ {
            split(d, p, ":")
            s = (p[2] < "80") ? 0 : 1
            if ((s == 0 && w0) || (s == 1 && w1)) n++
        }
        END { print n + 0 }'
}

n=$(residual)
if [ "$n" = "0" ]; then
    echo "[ACS] ✓ 目标 socket 的 ACS 已全部清零，P2P 通路正常"
    exit 0
fi

echo "[ACS] ⚠ 检出 $n 个桥仍处 SrcValid+（P2P 会走 IOMMU 重定向，623ms/次）"
if [ -x /usr/local/sbin/acs_clear_all.sh ] && sudo -n true 2>/dev/null; then
    echo "[ACS]   尝试自动清除…"
    sudo /usr/local/sbin/acs_clear_all.sh >/dev/null 2>&1
    n=$(residual)
    if [ "$n" = "0" ]; then
        echo "[ACS] ✓ 已自动清除，继续启动"
        exit 0
    fi
fi

cat >&2 <<MSG

[ACS] ✗ 拦截启动：仍有 $n 个桥的 ACS 未清除。
      直接起服务会得到约 0.05 tok/s（能起来、无报错、慢 1000 倍）。

      三选一：
        1) sudo systemctl restart acs-clear.service     # 首选
        2) sudo /usr/local/sbin/acs_clear_all.sh        # 手工清
        3) 给启动命令加回 --disable-custom-all-reduce   # 退回 SHM 路径
           （性能腰斩但可用：all-reduce 26.3 → 3.5 GB/s）

      详见 docs/ACS-P2P解锁.md。确认要带病启动可设 SKIP_ACS_CHECK=1。
MSG
exit 1
