#!/bin/bash
# ============================================================================
# 构建「拉起包」—— 一线一包。
#
#   bash deploy-kit/build_package.sh 101         # 只打 101 号线
#   bash deploy-kit/build_package.sh all         # 两条线各打一包
#   bash deploy-kit/build_package.sh 101 /out/dir
#
# 产出:  dsv4-kit-<NN>-<日期>.tar.gz
# 现场:  tar xzf dsv4-kit-101-*.tar.gz && cd dsv4-kit-101 && bash up.sh
#        （bash up.sh = 拉起本线；status/stop 同理；除镜像/权重外自包含）
# ============================================================================
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$HERE/scripts"; P="$HERE/patches"
SEL="${1:?用法: build_package.sh <NN|all> [输出目录]}"
OUT="${2:-$PWD}"
TS="$(date +%Y%m%d)"
# 溯源与确定性（发布器在净化副本里构建时无 .git，由环境显式传入）
SOURCE_REPO="${SOURCE_REPO:-$(basename "$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo n/a)")}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo n/a)}"
SOURCE_EPOCH="${SOURCE_EPOCH:-$(git -C "$HERE" log -1 --format=%ct 2>/dev/null || echo 0)}"
# 版本 = <YYYYMMDD>-g<Git短哈希>（同日多构建可区分；commit 未知时退化为纯日期）
if [ "$SOURCE_COMMIT" != n/a ]; then VER="$TS-g${SOURCE_COMMIT:0:7}"; else VER="$TS"; fi

COMMON_CORE="launch.sh ensure_ready.sh fetch_hf_model.sh machine_prep.sh numa_bind.sh preflight_acs.sh"

build_one(){
  local NN="$1"
  local CFGDIR; CFGDIR=$(ls -d "$S/$NN"-* 2>/dev/null | head -1)
  [ -n "$CFGDIR" ] || { echo "✗ 无编号 $NN"; return 1; }
  local CFG; CFG=$(basename "$CFGDIR")
  local TMP; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' RETURN
  local ROOT="$TMP/dsv4-kit-$NN"
  mkdir -p "$ROOT/common" "$ROOT/lib" "$ROOT/patches" "$ROOT/docs"

  # 骨架
  install -m0755 "$S/up.sh" "$ROOT/up.sh"
  echo "$CFG" > "$ROOT/.primary"
  echo "dsv4-kit-$NN $VER" > "$ROOT/VERSION"
  for f in $COMMON_CORE; do cp "$S/common/$f" "$ROOT/common/"; done
  cp -r "$S/lib/." "$ROOT/lib/"
  cp "$P/rccl-acs-topo/acs_clear_all.sh" "$P/rccl-acs-topo/acs-clear.service" "$ROOT/patches/"
  cp "$HERE/docs/方案脚本规范-设计文档.md" "$ROOT/docs/"
  cp -r "$CFGDIR" "$ROOT/$CFG"

  # 按线附件：两条线都带 w8a8 量化配方
  cp -r "$HERE/quant" "$ROOT/quant"

  # 包根 README：一句话 + 指向线内 README
  cat > "$ROOT/README.md" <<EOF
# dsv4-kit-$NN（拉起包 · 一线一包）

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

  # ── 通用机器可读清单 MANIFEST.json（消费方中立；自身不入 files；除自身外全量列举）──
  ROOT="$ROOT" NN="$NN" CFG="$CFG" VER="$VER" \
  SOURCE_REPO="$SOURCE_REPO" SOURCE_COMMIT="$SOURCE_COMMIT" \
  python3 - <<'PYEOF'
import hashlib, json, os, re, sys
from pathlib import Path

root = Path(os.environ["ROOT"])
nn, cfg, ver = os.environ["NN"], os.environ["CFG"], os.environ["VER"]
serve_path = root / cfg / "serve.sh"
# 若某线无单一 serve.sh（多容器编排），引擎/模型字段降级为 null
serve = serve_path.read_text(encoding="utf-8", errors="replace") if serve_path.exists() else ""
# ★ DSv4 的线把真正的启动参数放在 inner/run_prodfix.sh 里（serve.sh 只负责起容器再调它），
#   只扫 serve.sh 会让 served_model / context_length 恒为 null。把 inner/ 下的脚本一并纳入。
for extra in sorted((root / cfg / "inner").glob("*.sh")) if (root / cfg / "inner").is_dir() else []:
    serve += "\n" + extra.read_text(encoding="utf-8", errors="replace")

def meta(tag, split=False):
    """收集 `# @tag 值` 行。split=True 时按空白拆成多项——
    @requires 语义上是**多个路径**，整串返回会让消费方无法逐项校验。"""
    out = []
    for line in serve.splitlines():
        m = re.match(rf"#\s*@{tag}[ \t]+(.*)", line)
        if m:
            v = m.group(1).strip()
            out.extend(v.split()) if split else out.append(v)
    return out

def resolve_num(token):
    """serve.sh 里的字面数字，或 VAR="${VAR:-默认}" 形式的默认值。"""
    if re.fullmatch(r"\d+", token):
        return int(token)
    m = re.fullmatch(r"\$\{?([A-Z_][A-Z0-9_]*)[:}-].*", token) or re.fullmatch(r"\$([A-Z_][A-Z0-9_]*)", token)
    if m:
        var = m.group(1)
        d = re.search(rf'{var}="\$\{{{var}:-(\d+)\}}"', serve)
        if d:
            return int(d.group(1))
    return None

engine = "sglang" if re.search(r"launch_server|sglang", serve) else ("vllm" if re.search(r"vllm", serve) else "unknown")
sm = re.search(r"--served-model-name[= ]+(\S+)", serve) or re.search(r"SERVED_MODEL_NAME=([A-Za-z0-9_.-]+)", serve)
served_model = sm.group(1).strip('"') if sm else None
cl = re.search(r"--context-length[= ]+(\S+)", serve) or re.search(r'\bCTX="\$\{CTX:-(\d+)\}"', serve)
context_length = resolve_num(cl.group(1).strip('"')) if cl else None

files = []
for p in sorted(x for x in root.rglob("*") if x.is_file()):
    rel = p.relative_to(root).as_posix()
    if rel == "MANIFEST.json":
        continue
    files.append({"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})

manifest = {
    "schema_version": 1,
    "kit": {"id": f"dsv4-kit-{nn}", "line": nn, "version": ver},
    "source": {"repo": os.environ["SOURCE_REPO"], "commit": os.environ["SOURCE_COMMIT"]},
    "engine": engine,
    "entrypoints": {
        "orchestrated": "up.sh",
        "direct": ("common/launch.sh <config_dir> [KEY=VAL ...]" if serve else None),
    },
    "config_dir": cfg,
    "parameters": {
        "GPUS": "可见加速卡列表（逗号分隔，如 0,1,2,3）",
        "PORT": "服务监听端口",
        "NAME": "容器实例名",
    },
    "env_switches": {
        "SKIP_S2": "跳过镜像/权重自举获取（规范变量；离线现场必设 1）",
        "SKIP_FETCH": "SKIP_S2 的兼容别名",
        "MODELS_ROOT": "模型权重根目录（默认 /data/models）",
    },
    "requires": meta("requires", split=True),
    "served_model": served_model,
    "context_length": context_length,
    "files": files,
}
(root / "MANIFEST.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PYEOF

  local PKG="$OUT/dsv4-kit-$NN-$VER.tar.gz"
  tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$SOURCE_EPOCH" \
      -C "$TMP" -c "dsv4-kit-$NN" | gzip -n > "$PKG"
  printf "✓ %-28s %6s  %3d 文件\n" "$(basename "$PKG")" "$(du -h "$PKG" | cut -f1)" "$(find "$ROOT" -type f | wc -l)"
  rm -rf "$TMP"; trap - RETURN
}

if [ "$SEL" = all ]; then
  for d in "$S"/[0-9][0-9]*-*/; do b=$(basename "$d"); build_one "${b%%-*}"; done
else
  build_one "$SEL"
fi
echo "现场用法: tar xzf dsv4-kit-<NN>-*.tar.gz && cd dsv4-kit-<NN> && bash up.sh"
