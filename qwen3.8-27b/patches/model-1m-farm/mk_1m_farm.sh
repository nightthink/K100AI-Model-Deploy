#!/bin/bash
# 制作 Qwen3.8-27B 的 1M 软链农场：
#   权重/分词器全部软链原目录，唯 config.json 用本目录的 1M YaRN 版
#   （max_position 262144→1000000；rope_type default→yarn factor=4.0，
#    original_max_position_embeddings=262144，其余逐字段同原版）。
set -euo pipefail
SRC="${1:-/data/models/Qwen3.8-27B}"
DST="${2:-/data/models/Qwen3.8-27B-1M}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -d "$SRC" ] || { echo "✗ 原模型不存在: $SRC"; exit 1; }
mkdir -p "$DST"
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ "$b" = "config.json" ] && continue
  ln -sfn "$f" "$DST/$b"
done
cp "$HERE/config.json" "$DST/config.json"
n=$(ls -A "$DST" | wc -l); l=$(find "$DST" -maxdepth 1 -type l | wc -l)
echo "✓ $DST：$n 项（软链 $l + config.json）"
find "$DST" -maxdepth 1 -xtype l | grep -q . && { echo "✗ 有断链"; exit 1; } || echo "✓ 无断链"
