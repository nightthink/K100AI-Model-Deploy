#!/bin/bash
# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
#
# 从本地 0.5.12 镜像提取 gfx928 paged-varlen 修复件（repair.py）。
#
# 为什么要提取而不是随包发布：
#   repair.py 出自海光 SourceFind 的 K100 补丁树（镜像内 /data/qwen38-27b-k100ai-int8-opt/），
#   不是我方原创，**不可按 MIT 随公开包再分发**。故包内只放本提取脚本，
#   由现场从自己已拉取的 0.5.12 镜像中取出。
#
# 为什么 0.5.18 基版仍然需要它：
#   实测 0.5.18 上游**没有修** gfx928 的 paged-varlen 正确性问题
#   （flashattention_interface.py 内 gfx928/paged 相关命中 0），
#   而 13 线日志里该修复是八个 rank 各一条 ACTIVE、确实在生效。
#   它 import 的三个模块在 0.5.18 上路径未变，可直接加载。
set -euo pipefail
SRC_IMG="${SRC_IMG:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828}"
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IN_IMG_PATH=/data/qwen38-27b-k100ai-int8-opt/runtime_patch_sglang_gfx928_paged_varlen_repair_v1/repair.py
D=docker; docker info >/dev/null 2>&1 || D="sudo docker"

$D image inspect "$SRC_IMG" >/dev/null 2>&1 || {
  echo "✗ 本地没有源镜像：$SRC_IMG"
  echo "  先 docker pull 它，或用 SRC_IMG=<tag> 指定另一个含该补丁树的 0.5.12 镜像。"; exit 1; }

C=$($D create "$SRC_IMG"); trap '$D rm "$C" >/dev/null 2>&1 || true' EXIT
mkdir -p "$OUT" || { echo "✗ 建不了输出目录 $OUT"; exit 1; }
$D cp "$C:$IN_IMG_PATH" "$OUT/repair.py"
[ -s "$OUT/repair.py" ] || { echo "✗ 提取为空"; exit 1; }
grep -q 'gfx928 SGLang paged-varlen correctness repair' "$OUT/repair.py" || {
  echo "✗ 提取到的文件不是预期内容（首行标识不符）"; exit 1; }
echo "✓ repair.py → $OUT/repair.py（$(wc -c < "$OUT/repair.py") 字节）"
echo "  校验：$(sha256sum "$OUT/repair.py" | cut -c1-16)"
