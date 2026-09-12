#!/bin/bash
# ============================================================================
# 中立编排器：按标准口径串行跑若干条线的基准。
#
#   bash run_lines.sh <profile> <线清单文件> [每线小时数]
#
# 线清单文件每行四列（# 开头为注释）：
#   TAG   拉起包目录                 端口   容器名
#   13    /home/opsuser/q38-13prod   8113   q38-bf16tp8
#
# profile 决定 model 字段、thinking 键与就绪探针，见 profiles.py。
#
# 串行是硬要求：同机并发实例会互相污染测量。
#
# 用法（必须分离式启动，否则 ssh 一断就全丢；历史上已因此损失三次长跑）：
#   ssh <host> 'setsid nohup bash <此脚本> qwen3.8-27b lines.txt 2.0 \
#                 > /tmp/runlines.boot 2>&1 < /dev/null &'
#
# 起跑前必查（缺一项就可能白跑数小时）：
#   ① 远端 bench/*.py 与本地的 md5 **逐项一致**
#      —— 曾因 scenarios.py 未同步，在 27 秒时才截住，差点白跑 6.5 小时
#   ② 加速卡状态正常、链路宽度满速、故障计数为 0
#   ③ 若该线依赖驱动补丁，确认补丁在**运行中**的驱动里生效
#      （符号地址每次开机都变，必须动态查 /proc/kallsyms，不能只查文件）
# ============================================================================
set -u
PROFILE="${1:?用法: run_lines.sh <profile> <线清单文件> [每线小时数]}"
LINES_FILE="${2:?缺线清单文件}"
HOURS="${3:-2.0}"
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/bench_out}"
LOG="${LOG:-/tmp/run_lines.log}"
: > "$LOG"
say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }

# 各线跑完后登记，供末尾出双线对照报告用（脚本开头是 set -u，必须先声明）
DONE_OUTS=(); DONE_TAGS=()

# 就绪探针要匹配什么，问档案要——不同模型的 served-model-name 不同
READY=$(python3 -c "
import sys; sys.path.insert(0,'$BENCH')
import profiles; print(profiles.get('$PROFILE')['ready_token'])") || exit 1
MODELNAME=$(python3 -c "
import sys; sys.path.insert(0,'$BENCH')
import profiles; print(profiles.get('$PROFILE')['model'])") || exit 1
say "档案 $PROFILE：model=$MODELNAME 就绪探针=$READY 每线 ${HOURS}h"

# 停容器后必须等显存真正退干净，不能只 sleep 固定秒数。
#
# 大模型线的显存余量本就很紧（实测：mem-fraction 0.85 时 57.5GB 已占，
# 16K prompt 的单 chunk 激活还要 4GB；某卡只剩 604MB 时直接 HIP OOM、
# 调度器崩溃）。上一条线的显存若没退完，下一条按同样的 mem-fraction 去算
# KV 池就会踩到 OOM 线——白跑一整组，而且失败形态看起来像「这条线起不来」，
# 极易误判成配置问题。

# 找 smi 工具：**非交互 ssh 的 PATH 不含 /opt/hyhal/bin**，直接写 `hy-smi` 会
# command not found；若再把错误吞掉，就成了「看起来查过、其实没查」。
_find_smi(){
  local c
  for c in /opt/hyhal/bin/hy-smi /opt/dtk/bin/rocm-smi \
           "$(command -v hy-smi 2>/dev/null)" "$(command -v rocm-smi 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && "$c" --showmeminfo vram >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  return 1
}

wait_vram_free(){
  # 阈值单位是 **MiB**：hy-smi 打的是
  #   HCU[0]  : vram Total Used Memory (MiB): 63717
  # 不是字节。按字节写阈值（或按 9 位以上数字去抓）会永远匹配不到，
  # 于是函数每次都「立刻通过」——这个坑我踩过一次，记在这里。
  local lim_mib=${1:-2000} smi i u=""
  if ! smi=$(_find_smi); then
    say "  ⚠ 找不到可用的 smi 工具，无法确认显存已释放 —— 改为固定等待 90s"
    sleep 90; return 0
  fi
  for i in $(seq 1 36); do          # 最多 3 分钟
    u=$("$smi" --showmeminfo vram 2>/dev/null \
        | grep -i "Total Used Memory" | grep -oE '[0-9]+$' | sort -rn | head -1)
    if [ -z "$u" ]; then
      say "  ⚠ smi 输出解析不到显存数值（格式可能变了）—— 改为固定等待 90s"
      sleep 90; return 0
    fi
    [ "$u" -lt "$lim_mib" ] && { say "  显存已释放（峰值卡 ${u} MiB）"; return 0; }
    sleep 5
  done
  say "  ⚠ 3 分钟后峰值卡仍占 ${u} MiB，继续但需留意 OOM"
}

run_line(){
  local TAG=$1 DIR=$2 PORT=$3 CT=$4
  say "===== 开始 $TAG 线（$DIR, port $PORT, 容器 $CT）====="
  sudo docker rm -f "$CT" >/dev/null 2>&1; sleep 5
  wait_vram_free
  ( cd "$DIR" && bash serve.sh ) >> "$LOG" 2>&1

  local T0; T0=$(date +%s)
  for _ in $(seq 1 100); do
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$READY" && break
    sudo docker ps --format '{{.Names}}' | grep -q "$CT" || { say "$TAG 就绪前即退出"; return 1; }
    sleep 15
  done
  say "$TAG 就绪 $(( $(date +%s) - T0 ))s ；custom_AR_ranks=$(sudo docker logs "$CT" 2>&1 | grep -c 'using PCIe.s custom allreduce')"

  for _ in 1 2 3; do
    curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODELNAME\",\"messages\":[{\"role\":\"user\",\"content\":\"写一个快排\"}],\"max_tokens\":128}" >/dev/null
  done
  local BEG; BEG=$(date +%T); say "$TAG 预热完成，基准开始"

  # ★ 每线开跑前腾空输出目录：Recorder 以 "a" 模式打开 requests.jsonl，
  #   若沿用旧目录会把上一轮的行混进本轮，报告就是新旧数据的平均值（已踩过：
  #   凌晨 514 行与新 4 行混在同一文件里）。改名而非删除，旧数据仍可回查。
  local OUT="$OUT_ROOT/${PROFILE}_$TAG"
  [ -d "$OUT" ] && { mv "$OUT" "${OUT}.old_$(date +%m%d%H%M)"; say "$TAG 旧输出已改名保留"; }
  mkdir -p "$OUT"

  python3 -u "$BENCH/run_bench.py" --profile "$PROFILE" --port "$PORT" \
    --hours "$HOURS" --seed 7 --container "$CT" --out "$OUT" >> "$LOG" 2>&1 &
  local BP=$!
  # 看门狗：容器一死就掐掉压测，避免对着死容器空打满整轮（某线历史上会 VMFault 崩，
  # 曾产生 252 条「打空气」的垃圾失败记录，污染成功率统计）
  ( while kill -0 $BP 2>/dev/null; do
      sudo docker ps --format '{{.Names}}' | grep -q "$CT" || {
        echo "[$(date +%T)] $TAG 容器已退出，提前结束压测" >> "$LOG"; kill $BP 2>/dev/null; break; }
      sleep 30
    done ) &
  local WD=$!
  wait $BP 2>/dev/null; kill $WD 2>/dev/null
  local END; END=$(date +%T)

  say "$TAG 基准结束（$BEG → $END）"
  { echo "----- $TAG 实测缓存命中率 -----"
    python3 "$BENCH/cachestat.py" "$CT" "$BEG" "$END"
    echo "----- $TAG 故障计数 -----"
    for p in VMFault HSA_STATUS_ERROR SIGABRT; do
      echo "  $p $(sudo docker logs "$CT" 2>&1 | grep -c $p)"
    done
  } >> "$LOG" 2>&1
  # 容器日志单独留存：cachestat 依赖的 `Prefill batch` 行只在容器日志里
  # （曾因日志被下一轮清空而误判「命中率数据无出处」）
  sudo docker logs "$CT" > "$OUT/container.log" 2>&1

  # 同口径报告：prefill 与 decode 分开统计。run_bench.py 内置的 report.md 是简报，
  # 这份才是历次对照文档所用的口径（prefill=prompt_tok/ttft、decode=1/tpot、
  # 合计吞吐=(ptok+ctok)/e2el），两条线必须用同一个生成器出，否则不可比。
  python3 "$BENCH/report2.py" "$OUT/requests.jsonl" "$PROFILE-$TAG" \
    > "$OUT/report2.md" 2>>"$LOG" && say "$TAG 同口径报告 → $OUT/report2.md"
  DONE_OUTS+=("$OUT"); DONE_TAGS+=("$TAG")

  sudo docker rm -f "$CT" >/dev/null 2>&1; sleep 10
  say "===== $TAG 完成（输出 $OUT）====="
}

n=0
while read -r TAG DIR PORT CT; do
  case "${TAG:-}" in ""|\#*) continue;; esac
  run_line "$TAG" "$DIR" "$PORT" "$CT"; n=$((n+1))
done < "$LINES_FILE"

# 恰好两条线时，把两份同口径报告合订成一个文件。
# 注意 report2.py 的 --compare 并不做并排逐项对比，它是先后输出两份报告、
# 以 --- 分隔；但两份出自同一生成器、同一套口径，可以直接对读——
# 价值在「口径一致」而不在「自动算差值」，别把它当成算好的对比表。
# 三条以上不自动合订：谁跟谁比取决于实验设计，猜错不如不猜。
if [ "${#DONE_OUTS[@]}" -eq 2 ]; then
  CMP="$OUT_ROOT/compare_${PROFILE}_${DONE_TAGS[0]}_vs_${DONE_TAGS[1]}.md"
  if python3 "$BENCH/report2.py" \
       "${DONE_OUTS[0]}/requests.jsonl" "${DONE_OUTS[1]}/requests.jsonl" \
       --compare "${DONE_TAGS[0]}" "${DONE_TAGS[1]}" > "$CMP" 2>>"$LOG"; then
    say "两线同口径报告合订 → $CMP"
  else
    say "合订失败（见 $LOG）；两线各自的 report2.md 仍可用"
  fi
elif [ "${#DONE_OUTS[@]}" -gt 2 ]; then
  say "已跑 ${#DONE_OUTS[@]} 条线，未自动合订——请自行指定要比的两条："
  say "  python3 $BENCH/report2.py <A>/requests.jsonl <B>/requests.jsonl --compare A B"
fi

say "########## $n 条线全部完成 ##########"
