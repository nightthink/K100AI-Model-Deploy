#!/bin/bash
# DP2×TP4 单一入口：8100 → 8101/8102 粘性路由（详见 router.py 头部说明）
# router.py 与本脚本同目录，挂载点自适应——不再写死 /data/q38-work
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/router.py" ] || { echo "✗ 找不到 $HERE/router.py"; exit 1; }
# 本账号可能不在 docker 组但有 sudo NOPASSWD
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
RNAME="${RNAME:-q38-router}"
$DOCKER rm -f "$RNAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$RNAME" --network host \
  -v "$HERE":/w:ro -w /w \
  -e UPSTREAMS="${UPSTREAMS:-http://127.0.0.1:8101,http://127.0.0.1:8102}" \
  --entrypoint python3 \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706 \
  -m uvicorn router:app --host 0.0.0.0 --port "${PORT:-8100}" --log-level warning
echo "$RNAME started on ${PORT:-8100}"
