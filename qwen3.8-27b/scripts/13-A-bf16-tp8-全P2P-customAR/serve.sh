#!/bin/bash
# @image      harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828
# @name  q38-bf16tp8
# @port  8113
# @gpus  0,1,2,3,4,5,6,7
# @expect-gpu 8
# @requires $MODELS_ROOT/Qwen3.8-27B-1M $MODELS_ROOT/Qwen3.8-27B $CFGDIR/minichain5n/sitecustomize.py
# @attest 上下文长度:context_length
# @attest 投机解码NEXTN:speculative
# @attest custom all-reduce启用:custom allreduce
# ============================================================================
# 13 · bf16 + NEXTN · TP8 全 P2P —— 01 线的八卡版（2026-09-10 改版）
#
# ★ 2026-09-10 重大改版：驱逐风暴根因已定位并修复（hycu.ko v4 补丁），
#   本线由「混合传输 + agent 过滤 + 关 AR」改为「全 P2P + 开 AR」。
#   旧方案的两条前提（agent 过滤必需、custom-AR 必崩）**都已不成立**。
#
# 相对 01 线（TP4·同socket）只改两处：
#   ① 放开「只允许同 socket 四卡组」门禁 —— 本线要横跨两个 socket
#   ② --tp 4 → --tp 8
#   （不再需要 --disable-custom-all-reduce，也不再需要 dlhook2-sg.so agent 过滤；
#     NCCL_P2P_LEVEL 用 SYS 放开全机 P2P）
#
# 硬前提改为：**驱动必须打了 v4 补丁**（脚本启动前自检，未打拒启）。
#
# 根因（详见 docs/hycu.ko符号级还原-第一轮.md 第十七轮）：
#   内核 drivers/pci/p2pdma.c 的 host_bridge_whitelist[] 只列 Intel 根桥，
#   海光不在其中 ⇒ 跨 socket 的 pci_p2pdma_distance_many() 恒返回 -1
#   ⇒ gpu_dma_buf_attach 清零 attach->peer2peer
#   ⇒ gpu_dma_buf_map 只给 GTT（实测 domains=6 出现 0 次）
#   ⇒ 每次 dmabuf map 把导出方 BO 踢出 VRAM ⇒ 等驱逐 fence ⇒ 排驱逐
#   ⇒ restore_process_bos 重做 map ⇒ 活锁。v4 = 1 字节（79 jns → eb jmp）。
#
# 实测（验证机B，标准 harness tests/bench/run_bench.py，同 seed，各 1 小时）：
#   S3 并发合计 tok/s     01线TP4    13线(旧·SHM)   本线(全P2P)
#     4 路                2,031        1,653        2,912  (+43% / +76%)
#     8 路                2,619        2,252        3,962  (+51% / +76%)
#    16 路                2,199        1,732        3,117  (+42% / +80%)
#   S1 E2EL P50 10.89s → 5.40s（-50%）；TPOT 27.8 → 16.6ms（-40%）
#   全程 evict_process_worker = 0、schedule_evict = 0；服务错误 0 次
#
#   custom AR 单变量 A/B（两侧同一 #running-req:1 过滤）：
#     服务端单流 gen throughput 开 41.7~56.7 vs 关 23.0~28.2 = 2.15×
#     accept rate 两侧持平（0.49~0.65 vs 0.46~0.68）⇒ 排除投机解码解释
#   见 docs/bench-2026-09-09-选线指南/custom-all-reduce-AB定案.md
#
# ⚠ 硬性前提：
#   - **驱动必须打 v4 补丁**，否则 TP8 全 P2P 会触发驱逐活锁（起不来）。
#     本脚本启动前自检 .ko 字节，未打直接拒启。
#   - 驱动**不可用 rmmod/insmod 重载**后直接投产：重载会使 KFD 驱逐机制永久
#     打摆，性能塌 ~550 倍（详见同文档第九轮）。
#     打补丁一律「改文件 → 重启机器」。
#   - 其余边界（1M 农场符号链接、overrides、hygon tokenizer）同 01 线。
# ============================================================================
set -e
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; PORT="${PORT:-8113}"; NAME="${NAME:-q38-bf16tp8}"
MEM_FRAC="${MEM_FRAC:-0.90}"
D="${WORK:-/data/q38-work}"
CFGDIR="${CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-/data/models}"
M="$MODELS_ROOT/Qwen3.8-27B-1M"
TOK="$MODELS_ROOT/Qwen3.8-27B-Channel-INT8-w8a8-hygon"
mkdir -p "$D/tritoncache-int8" 2>/dev/null || true
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

# ── v4 补丁自检：TP8 全 P2P 的硬前提，未打则拒启 ──────────────────────
# 读的是**磁盘上的 .ko 文件**：若文件已打补丁但机器未重启，运行中的仍是旧驱动，
# 本检查查不出来——所以打完补丁必须重启（见文件头「硬性前提」）。
KO="${HYCU_KO:-/usr/local/hyhal/dkms/hycu.ko}"
V4_OFF=$((0xa0 + 0x2b3dd))                       # = 0x2B47D，gpu_dma_buf_attach 内的 jns
if [ -f "$KO" ]; then
  V4B=$(sudo xxd -p -s "$V4_OFF" -l 1 "$KO" 2>/dev/null || echo "")
  case "$V4B" in
    eb) ;;                                        # v4 在位，放行
    79) echo "✗ hycu.ko 未打 v4 补丁（.text+0x2b3dd = 79/jns）。"
        echo "  TP8 全 P2P 在未打补丁的驱动上会触发驱逐活锁，服务起不来。"
        echo "  修复：sudo bash patches/patch_v4.sh apply  然后**重启机器**"
        exit 1;;
    "") echo "⚠ 读不到 $KO 的补丁字节（权限或 xxd 缺失），跳过 v4 自检";;
    *)  echo "⚠ v4 补丁状态无法判定（读到 '$V4B'，期望 eb 或 79）——"
        echo "  偏移可能随驱动版本变化，请人工确认后再投产";;
  esac
else
  echo "⚠ 未找到 $KO，跳过 v4 自检（若本机驱动路径不同，用 HYCU_KO= 指定）"
fi

TOKMOUNTS=()
if [ -f "$TOK/tokenizer.json" ]; then
  TOKMOUNTS=(-v "$TOK/tokenizer.json":/models/target/tokenizer.json:ro
             -v "$TOK/tokenizer_config.json":/models/target/tokenizer_config.json:ro)
else
  echo "⚠ 未找到 hygon tokenizer（$TOK），沿用农场自带——若 think_end 报错请补齐该目录"
fi

echo "13-bf16 TP8 全P2P $NAME: GPU=$GPUS PORT=$PORT mem=$MEM_FRAC (开AR + 全P2P)"
$DOCKER run -d --name "$NAME" \
  --network host --ipc host --privileged --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$D/tritoncache-int8":/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$D":/w \
  -v "$MODELS_ROOT":"$MODELS_ROOT":ro \
  -v "$M":/models/target:ro \
  -v "$CFGDIR/overrides/preprocessor_config.json":/models/target/preprocessor_config.json:ro \
  -v "$CFGDIR/overrides/video_preprocessor_config.json":/models/target/video_preprocessor_config.json:ro \
  "${TOKMOUNTS[@]}" \
  -v "$CFGDIR/minichain5n":/minichain4:ro \
  -e HIP_VISIBLE_DEVICES=$GPUS \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_KV_LAYOUT_DCU_FA=true -e SGLANG_USE_LIGHTOP=0 \
  -e SGLANG_USE_CAUSAL_CONV1D=0 -e SGLANG_USE_TRITON_VLLM_FA=0 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_ALGO=Ring -e NCCL_DEBUG=WARN \
  -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  -e PYTHONPATH=/minichain4 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828 -c "
  export PYTHONPATH=/minichain4
  python3 -m sglang.launch_server \
    --model-path /models/target --served-model-name qwen38 \
    --trust-remote-code --tp 8 --page-size 64 \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --attention-backend fa3 --mm-attention-backend fa3 \
    --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 --disable-piecewise-cuda-graph \
    --mem-fraction-static $MEM_FRAC \
    --context-length 1000000 \
    --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 2048 --pack-paged-kv-to-varlen-min-kv-tokens 2048 \
    --speculative-algorithm NEXTN --speculative-num-steps 2 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
    --watchdog-timeout 7200 --dist-timeout 7200 --skip-server-warmup \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port $PORT" >/dev/null

echo "$NAME started on $PORT（就绪看 /v1/models；TP8 首次就绪约 7 分钟）"
