#!/bin/bash
# ============================================================================
# S2 · 宿主环境自举（十步契约中此前"文档定义、无脚本"的那一步，2026-08-28 落实）
#
# 读 serve.sh 头部元数据，缺什么补什么、有则跳过：
#   # @image   <完整镜像 URL>          → docker pull（本地有则跳过）
#   # @weights <HF repo_id> <目录名>   → fetch_hf_model.sh 到 $MODELS_ROOT/<目录名>
#   # @farm    <基权重目录名> <农场目录名> → mk_1m_farm.sh 生成（已存在则跳过）
#
# 开关：SKIP_FETCH=1 跳过全部补齐（只检查并报缺）；DRY=1 只打印将执行的动作。
# ============================================================================
set -u
SERVE="${1:?用法: ensure_ready.sh <serve.sh 路径>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$(dirname "$HERE")}"
export MODELS_ROOT="${MODELS_ROOT:-/data/models}"
say(){ echo "[$(date +%H:%M:%S)] S2 $*"; }
meta(){ grep -E "^# @$1[ \t]" "$SERVE" | sed -E "s/^# @$1[ \t]+//"; }
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
run(){ [ "${DRY:-0}" = 1 ] && { say "DRY: $*"; return 0; }; "$@"; }
fail=0

# ── 镜像 ──
while read -r img; do
  [ -n "$img" ] || continue
  if $DOCKER image inspect "$img" >/dev/null 2>&1; then
    say "✓ 镜像已在: $img"
  elif [ "${SKIP_FETCH:-0}" = 1 ]; then
    say "✗ 缺镜像（SKIP_FETCH=1 不拉）: $img"; fail=1
  else
    say "拉取镜像（本地无）: $img"
    ok=0
    for a in 1 2 3; do run $DOCKER pull "$img" && { ok=1; break; }; say "  第 $a 次失败，重试"; sleep 10; done
    [ "$ok" = 1 ] || { say "✗ 镜像拉取失败: $img"; fail=1; }
  fi
done < <(meta image)

# ── 权重 ──
while read -r repo dirname; do
  [ -n "$repo" ] || continue
  dest="$MODELS_ROOT/$dirname"
  if [ -e "$dest/config.json" ] || [ -e "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    say "✓ 权重已在: $dest"
  elif [ "${SKIP_FETCH:-0}" = 1 ]; then
    say "✗ 缺权重（SKIP_FETCH=1 不拉）: $dest（源 $repo）"; fail=1
  else
    say "拉取权重（本地无）: $repo → $dest"
    if [ "${DRY:-0}" = 1 ]; then say "DRY: fetch_hf_model.sh $repo $dest"
    else
      # 目标目录可能需要提权创建（如 /data/models 属 root）
      mkdir -p "$dest" 2>/dev/null || { sudo mkdir -p "$dest" && sudo chown "$(id -u):$(id -g)" "$dest"; }
      bash "$HERE/fetch_hf_model.sh" "$repo" "$dest" || { say "✗ 权重拉取失败: $repo"; fail=1; }
    fi
  fi
done < <(meta weights)

# ── ModelScope 权重（@weights-ms <ms-org/repo> <目录名>，git clone + lfs）──
while read -r repo dirname; do
  [ -n "$repo" ] || continue
  dest="$MODELS_ROOT/$dirname"
  if [ -e "$dest/config.json" ]; then
    say "✓ 权重已在: $dest"
  elif [ "${SKIP_FETCH:-0}" = 1 ]; then
    say "✗ 缺权重（SKIP_FETCH=1 不拉）: $dest（源 modelscope:$repo）"; fail=1
  else
    say "拉取权重（ModelScope）: $repo → $dest"
    if [ "${DRY:-0}" = 1 ]; then say "DRY: git clone modelscope $repo $dest"
    else
      command -v git-lfs >/dev/null 2>&1 || command -v git >/dev/null 2>&1 || { say "✗ 需要 git+git-lfs"; fail=1; continue; }
      mkdir -p "$(dirname "$dest")" 2>/dev/null || { sudo mkdir -p "$(dirname "$dest")" && sudo chown "$(id -u):$(id -g)" "$(dirname "$dest")"; }
      GIT_LFS_SKIP_SMUDGE=0 git clone "https://www.modelscope.cn/${repo}.git" "$dest" \
        || { say "✗ ModelScope 克隆失败: $repo（断点续传：cd $dest && git lfs pull）"; fail=1; }
    fi
  fi
done < <(meta weights-ms)

# ── 1M 农场 ──
while read -r base farm; do
  [ -n "$base" ] || continue
  dest="$MODELS_ROOT/$farm"
  if [ -e "$dest/config.json" ]; then
    say "✓ 农场已在: $dest"
  else
    MK="$WORK/patches/model-1m-farm/mk_1m_farm.sh"
    [ -f "$MK" ] || MK="$WORK/../patches/model-1m-farm/mk_1m_farm.sh"
    [ -f "$MK" ] || { say "✗ 找不到 mk_1m_farm.sh"; fail=1; continue; }
    say "生成 1M 农场: $base → $farm"
    run bash "$MK" "$MODELS_ROOT/$base" "$dest" || { say "✗ 农场生成失败"; fail=1; }
  fi
done < <(meta farm)

[ "$fail" = 0 ] && say "通过" || say "未通过（缺项见上）"
exit $fail
