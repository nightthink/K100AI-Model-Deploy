#!/bin/bash
# ============================================================================
# 构建「拉起包」—— 一线一包。
#
#   bash deploy-kit/build_package.sh 09          # 只打 09 号线
#   bash deploy-kit/build_package.sh all         # 十条线各打一包
#   bash deploy-kit/build_package.sh 09 /out/dir
#
# 产出:  q38-kit-<NN>-<日期>.tar.gz
# 现场:  tar xzf q38-kit-09-*.tar.gz && cd q38-kit-09 && bash up.sh
#        （bash up.sh = 拉起本线；status/stop 同理；除镜像/权重外自包含）
# ============================================================================
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$HERE/scripts"; P="$HERE/patches"
SEL="${1:?用法: build_package.sh <NN|all> [输出目录]}"
OUT="${2:-$PWD}"
TS="$(date +%Y%m%d)"
GIT="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo n/a)"

COMMON_CORE="launch.sh ensure_ready.sh fetch_hf_model.sh machine_prep.sh numa_bind.sh preflight_acs.sh"

build_one(){
  local NN="$1"
  local CFGDIR; CFGDIR=$(ls -d "$S/$NN"-* 2>/dev/null | head -1)
  [ -n "$CFGDIR" ] || { echo "✗ 无编号 $NN"; return 1; }
  local CFG; CFG=$(basename "$CFGDIR")
  local TMP; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' RETURN
  local ROOT="$TMP/q38-kit-$NN"
  mkdir -p "$ROOT/common" "$ROOT/lib" "$ROOT/patches" "$ROOT/docs"

  # 骨架
  install -m0755 "$S/up.sh" "$ROOT/up.sh"
  echo "$CFG" > "$ROOT/.primary"
  echo "q38-kit-$NN $TS (git $GIT)" > "$ROOT/VERSION"
  for f in $COMMON_CORE; do cp "$S/common/$f" "$ROOT/common/"; done
  cp -r "$S/lib/." "$ROOT/lib/"
  cp "$P/rccl-acs-topo/acs_clear_all.sh" "$P/rccl-acs-topo/acs-clear.service" "$ROOT/patches/"
  cp "$HERE/docs/方案脚本规范-设计文档.md" "$ROOT/docs/"
  cp -r "$CFGDIR" "$ROOT/$CFG"

  # 按线附件
  case "$NN" in
    01)
      cp -r "$P/model-1m-farm" "$ROOT/patches/model-1m-farm";;   # v2：0828树+AR，不再需要 dlhook2-sg
    03)
      cp "$P/hip-agent-filter/prebuilt/dlhook2.so" "$ROOT/dlhook2-sg.so"
      cp -r "$P/model-1m-farm" "$ROOT/patches/model-1m-farm";;
    04)
      cp -r "$P/model-1m-farm" "$ROOT/patches/model-1m-farm";;
    02)
      cp -r "$S/01-A-sglang-tp4" "$ROOT/01-A-sglang-tp4"
      cp -r "$P/model-1m-farm" "$ROOT/patches/model-1m-farm"
      cp "$S/common/router.py" "$S/common/serve_router.sh" "$ROOT/common/";;
    05|06)
      cp "$P/vllm-gdn-cache/"*.py "$P/vllm-triton-attn/"*.py "$ROOT/patches/";;
    07|08)
      cp "$P/vllm-gdn-cache/"*.py "$P/vllm-triton-attn/"*.py "$ROOT/patches/"
      cp "$P/hip-agent-filter/prebuilt/dlhook2.so" "$ROOT/dlhook2.so";;
    09|10|11|12)
      cp -r "$HERE/quant" "$ROOT/quant";;
  esac

  # 包根 README：一句话 + 指向线内 README
  cat > "$ROOT/README.md" <<EOF
# q38-kit-$NN（拉起包 · 一线一包）

本包只含 **$CFG** 这一条配置线。除 **镜像** 与 **模型权重**（首次拉起自动获取）外自包含，目录中立。

\`\`\`bash
bash up.sh              # 拉起（体检→S1机器准备→S2..S8→真实请求冒烟）
bash up.sh GPUS=...     # 带参拉起（KEY=VAL 透传）
bash up.sh status       # 观察状态
bash up.sh stop         # 停止（S9 清场）
\`\`\`

配置详情（卡数/引擎/镜像URL/实测数据/硬性前提）见 [\`$CFG/README.md\`]($CFG/README.md)。
十步契约方法论见 \`docs/方案脚本规范-设计文档.md\`。升级：新包覆盖解包（勿 rm -rf，保留 triton 缓存目录）。
EOF

  local PKG="$OUT/q38-kit-$NN-$TS.tar.gz"
  tar -C "$TMP" -czf "$PKG" "q38-kit-$NN"
  printf "✓ %-28s %6s  %3d 文件\n" "$(basename "$PKG")" "$(du -h "$PKG" | cut -f1)" "$(find "$ROOT" -type f | wc -l)"
  rm -rf "$TMP"; trap - RETURN
}

if [ "$SEL" = all ]; then
  for d in "$S"/[0-9][0-9]-*/; do build_one "$(basename "$d" | cut -c1-2)"; done
else
  build_one "$SEL"
fi
echo "现场用法: tar xzf q38-kit-<NN>-*.tar.gz && cd q38-kit-<NN> && bash up.sh"
