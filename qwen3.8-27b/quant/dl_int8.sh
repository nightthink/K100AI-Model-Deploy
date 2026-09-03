#!/usr/bin/env bash
# 下载 Qwen3.8-27B SmoothQuant W8A8 INT8（第三方权重，31.2 GB）
# huggingface.co 直连不通时走 hf-mirror；8 路并发 + 断点续传。
set -uo pipefail
H="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO=Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
DEST="${1:-/data/models/Qwen3.8-27B-W8A8-INT8}"
J="${J:-8}"
mkdir -p "$DEST"; cd "$DEST"
echo "[$(date +%H:%M:%S)] 取文件清单"
curl -sk --max-time 60 "$H/api/models/$REPO" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for f in d.get('siblings') or []:
    n=f['rfilename']
    if not n.startswith('.'): print(n)
" > files.txt
echo "  共 $(wc -l < files.txt) 个文件"
get(){
  local f="$1"; [ -s "$f" ] && return 0
  mkdir -p "$(dirname "$f")" 2>/dev/null || true
  for a in 1 2 3 4 5; do
    curl -skL --retry 3 --retry-delay 5 -C - --max-time 3600 \
         --speed-limit 10240 --speed-time 120 \
         -o "$f" "$H/$REPO/resolve/main/$f" && [ -s "$f" ] && return 0
    sleep $((a*5))
  done
  echo "FAIL $f" >> failed.txt; return 1
}
export -f get; export H REPO
xargs -P "$J" -I{} bash -c 'get "{}"' < files.txt
echo "[$(date +%H:%M:%S)] 校验分片完整性（对 index）"
python3 - <<'PY'
import json,os,glob
d=json.load(open("model.safetensors.index.json"))
need=set(d["weight_map"].values())
have={os.path.basename(p) for p in glob.glob("*.safetensors")}
miss=sorted(need-have)
print("  索引要求 %d 分片，缺 %d %s" % (len(need), len(miss), miss[:3]))
print("  索引总大小 %.1f GB" % (d["metadata"]["total_size"]/1e9))
PY
[ -s failed.txt ] && { echo "✗ 有失败："; cat failed.txt; exit 1; } || echo "✓ 完成"
echo "下一步：python3 ../patches/runtime/fix_actorder_int8_config.py   # 修 config.json"
