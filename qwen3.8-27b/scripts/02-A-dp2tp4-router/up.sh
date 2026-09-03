#!/bin/bash
# ============================================================================
# 02 · A 线 · 8 卡 · DP2×TP4 + 网关 —— 一键拉起 / 一键停止
#
#   卡 0-3 → 副本 A（端口 8101）      ← 复用配置 01 的 serve.sh
#   卡 4-7 → 副本 B（端口 8102）
#   网关   → 端口 8100（common/router.py，粘性 + 按在飞数均衡）
#
# 两个副本串行拉起（同机并发起两个实例会互抢带宽，见护栏 1·七——那是测量禁忌；
# 拉起阶段串行只是稳妥：第二个副本的图捕获不与第一个的 warmup 抢卡）。
#
# 用法：
#   bash up.sh              # 拉起全套（副本 A → 副本 B → 网关）
#   bash up.sh down         # 停止全套（网关 → 副本，倒序）
#
# 就绪判断： curl http://127.0.0.1:8100/health
# 客户端只连 8100。同一会话固定落同一副本（4 级粘性键，见 router.py 头注）。
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(cd "$HERE/.." && pwd)"             # scripts/ 或远端 /data/q38-work（编号目录直下）
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
PFX="${PFX:-}"   # 拉起包统一入口传入的标记前缀（独立运行时为空，行为不变）

if [ "${1:-}" = "down" ]; then
  echo "── 停止（倒序）──"
  for c in ${PFX}q38-router ${PFX}q38-b ${PFX}q38-a q38-router q38-b q38-a; do
    $DOCKER update --restart=no "$c" >/dev/null 2>&1 || true
    $DOCKER rm -f "$c" >/dev/null 2>&1 && echo "  已停 $c" || true
  done
  exit 0
fi

echo "── 1/3 副本 A：卡 0-3 → 8101 ──"
env GPUS=0,1,2,3 PORT=8101 NAME=${PFX}q38-a READY_TIMEOUT=2400 \
    bash "$WORK/common/launch.sh" 01-A-sglang-tp4 || { echo "✗ 副本 A 失败"; exit 1; }

echo "── 2/3 副本 B：卡 4-7 → 8102 ──"
env GPUS=4,5,6,7 PORT=8102 NAME=${PFX}q38-b READY_TIMEOUT=2400 \
    bash "$WORK/common/launch.sh" 01-A-sglang-tp4 || { echo "✗ 副本 B 失败"; exit 1; }

echo "── 3/3 网关 → 8100 ──"
RIMG=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
$DOCKER image inspect "$RIMG" >/dev/null 2>&1 || { echo "拉取网关镜像: $RIMG"; $DOCKER pull "$RIMG" || exit 1; }
UPSTREAMS="http://127.0.0.1:8101,http://127.0.0.1:8102" PORT=8100 RNAME=${PFX}q38-router \
    bash "$WORK/common/serve_router.sh" || { echo "✗ 网关失败"; exit 1; }
for i in $(seq 1 30); do
  curl -s -m 2 -o /dev/null http://127.0.0.1:8100/health && { echo "网关就绪(${i}s)"; break; }
  [ "$i" = 30 ] && { echo "✗ 网关 30s 未就绪"; exit 1; }
  sleep 1
done

echo
echo "════ 全套就绪 ════"
echo "  入口   http://127.0.0.1:8100   （网关，客户端只连这里）"
echo "  副本A  http://127.0.0.1:8101   卡 0-3"
echo "  副本B  http://127.0.0.1:8102   卡 4-7"
echo "  停止   bash $HERE/up.sh down"
