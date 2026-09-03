#!/bin/bash
# 通用 HF 模型拉取器（走 hf-mirror；机器多半连不上 huggingface.co 直连）。
# 幂等：已有的文件跳过；结束后按 safetensors 索引核对完整性。
# 用法: fetch_hf_model.sh <repo_id> <目标目录> [并发=8]
set -uo pipefail
REPO="${1:?用法: fetch_hf_model.sh <repo_id> <目标目录> [并发]}"
DEST="${2:?缺目标目录}"
J="${3:-8}"
H="${HF_ENDPOINT:-https://hf-mirror.com}"
say(){ echo "[$(date +%H:%M:%S)][fetch] $*"; }
mkdir -p "$DEST"; cd "$DEST"
say "清单: $H/api/models/$REPO"
curl -sk --max-time 60 "$H/api/models/$REPO" | python3 -c "
import json,sys
d=json.load(sys.stdin)
sib=d.get('siblings') or []
assert sib, '取不到文件清单（仓库名错？镜像站不可达？）'
for f in sib:
    n=f['rfilename']
    if not n.startswith('.'): print(n)
" > .filelist || { say "✗ 清单失败"; exit 1; }
say "共 $(wc -l < .filelist) 个文件，目标 $DEST，并发 $J"
get(){
  local f="$1"; [ -s "$f" ] && return 0
  mkdir -p "$(dirname "$f")" 2>/dev/null || true
  for a in 1 2 3 4 5; do
    curl -skL --retry 3 --retry-delay 5 -C - --max-time 7200 \
         --speed-limit 10240 --speed-time 120 \
         -o "$f" "$H/$REPO/resolve/main/$f" && [ -s "$f" ] && return 0
    sleep $((a*5))
  done
  echo "$f" >> .failed; return 1
}
export -f get; export H REPO
rm -f .failed
xargs -P "$J" -I{} bash -c 'get "{}"' < .filelist
[ -s .failed ] && { say "✗ 失败清单:"; sed 's/^/  /' .failed; exit 1; }
if [ -f model.safetensors.index.json ]; then
  python3 - <<'PY' || exit 1
import json,os,glob,sys
d=json.load(open("model.safetensors.index.json"))
need=set(d["weight_map"].values())
have={os.path.basename(p) for p in glob.glob("*.safetensors")}
miss=sorted(need-have)
if miss: print("  ✗ 缺分片:", miss[:5]); sys.exit(1)
print("  ✓ 分片完整（%d 个，索引总大小 %.1f GB）" % (len(need), d["metadata"]["total_size"]/1e9))
PY
fi
say "✓ 完成"
