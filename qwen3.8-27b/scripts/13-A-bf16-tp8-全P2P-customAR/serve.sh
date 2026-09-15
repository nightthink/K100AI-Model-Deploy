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
# mamba 状态槽数。**默认 16 = 原配方**，未设时与既有发布包逐字一致。
# 2026-09-14 实测（本线线上 TP8）：设 64 可正常起服，就绪 434s，
#   OOM / Traceback / evict_process_worker / schedule_evict 全 0，冒烟正常。
# ⚠ 显存代价不是「每槽固定」，别拿 16 槽的 0.52GB 直接除：
#   16→64 槽，四项合计 0.52GB → 1.86GB（conv 0.01→0.02、ssm 0.30→1.14 近线性 3.8×、
#   intermediate_ssm 0.21→0.69 次线性 3.29×、intermediate_conv 0.00→0.01）。
#   外推请用**边际值约 0.0279 GB/槽**，不是 0.0325 也不是 0.0194。
# ⚠ 加槽不只吃空闲显存，KV 池会让位：max_total_num_tokens 3050496 → 2994112。
#   本线 KV 利用率实测长期 0.00，这笔交换划算；但估算上限时 KV 的退让空间也是约束。
# ★ 槽数是怎么变成并发上限的（2026-09-14 查 sglang 源码坐实，非拟合）：
#   model_runner_kv_cache_mixin.py::_resolve_max_num_reqs()
#     max_running_requests = min(server_args.max_running_requests, max_mamba_cache_size // ratio)
#   同文件 _calculate_mamba_ratio()：radix cache 开 ⇒ 不返回 1；
#     ratio = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO(3) + ADDITIONAL_RATIO_OVERLAP(2) = 5
#   ⇒ 16 槽 → 并发 3；64 槽 → 并发 12。两次实测运行时值逐位吻合。
#   注意：server_args 里的 max_running_requests=48 是**投机解码的默认值**
#        （server_args.py EAGLE 分支），本线从未成为瓶颈，别拿它当上限看。
#   ★ ratio 只取决于 radix cache / extra_buffer / overlap schedule，**与投机无关**——
#     去掉 NEXTN 不会提高并发上限，这条路是死的。
#   头寸：128 槽 → 并发 25，增量约 1.79GB（实测空闲 3.40GB，可行）；
#        要吃满 48 那道闸需 240 槽、约 4.91GB，不可行。
#        关 overlap(ratio=4→并发16) 或关 radix(ratio=1→并发48) 也能抬，但各有大代价，不建议。
#   ⚠ 上面是**服务端上限**，不等于实际吞吐——实际收益还被客户端并发封顶。
#     2026-09-14 实测：客户端并发 16 时服务端上限 12 是瓶颈；一旦放宽到 25，
#     瓶颈就换到客户端侧，实际收益封顶约 16/12 ≈ 1.33×，而不是 25/12 ≈ 2.08×。
#     ⇒ **加槽必须与客户端并发同步上调，单独做没有意义。**
#   ⚠ 且「客户端并发数」不等于服务端可见的需求：按行配对统计 #running-req + #queue-req，
#     实测峰值 12、从不超过 12，而客户端侧设的是 16 ⇒ 约 4 个并发没到达调度器
#     （卡在 HTTP/tokenizer 层，或客户端有效并发低于设定值）。估算前先实测这个和。
#   ⚠ 统计这类量必须取**同一行内配对**的值：#running-req 与 mamba usage 本就打在同一行，
#     分两次 tail 再比极值会得出相反结论（此坑本次踩过两次，一次是我一次是调用方）。
# 2026-09-14 负载实测（同一客户端、客户端侧并发 16）：
#   ⚠ 口径：**只数 `POST /v1` 行**。日志里混着 Decode/Prefill batch 等调度器行，
#     按「全日志行/分」统计会虚高约 1.3 倍。更要命的是两侧口径不一致：
#     本次曾用「基线的 POST/分」去比「新配置的全日志行/分」，得出 1.8× 的假增益，
#     实际只有 1.44×。**比倍数前先确认两边数的是同一种行。**
#   16 槽稳态 229 POST/分（17:36–17:47 均值）
#   64 槽稳态 329 POST/分（19:10–19:16 均值）⇒ **约 1.44×**
#   换算 5.49 次/秒；客户端侧独立测得 5.5–5.7 calls/s，两侧交叉验证一致。
#   ⚠⚠ **但 1.44× 不是干净的 A/B，只能当「变更前后的观测」，不可当增益结论。**
#     ★★ 但别用错证据：**「每批 token 数」不能用来判断工况差异。**
#       它随批大小走，而批大小由并发上限决定——正是本次改动的**因变量**。
#       实测每批序列数 1.51（16槽·并发3）→ 4.04（64槽·并发12），
#       于是每批 new token 345 → 1334（3.9×），看着像「请求变大了」，其实请求没变。
#       判工况必须除以 #new-seq 看**每序列**：new 228 → 330，只有 1.45×。
#       （曾据「单批 2.8 倍」断言工况不同，那是把因变量当混淆变量，已撤回。）
#     真正残留的差异是缓存复用：每序列 cached 445 → 379，命中 66.0% → 53.5%。
#     **且这不是采样臂造成的**——同一台服务上实测
#       plain 臂时段 17:14–17:49： 每序列 229 / 445，命中 66.0%
#       json 臂时段 17:51–18:26： 每序列 232 / 446，命中 65.8%
#     两者几乎相同（臂改的是输出长度，影响 decode 不影响 prefill 输入）。
#     ⇒ 两次测量仍不可比（阶段不同，且高并发下 radix 复用变差），但原因不是「请求变大」。
#   ⇒ 做 A/B 时：比 POST/分 与**每序列** new/cached，**不要比每批量**；
#     并记下调用方的阶段与客户端并发——两者任一不同都不可比。
#   尽管如此默认仍保持 16：README 各性能表都是在 16 槽下测的，改默认会让那些数字失真。
#   要并发吞吐就显式传 MAMBA_CACHE=64。
# ★★★ 2026-09-14 关键实测：**加并发 / 加批 / 抬闸都提不了总吞吐**（结论按证据强弱分层，见下）
#   ⚠ 标题不用「零摊薄」一词：那是对「吞吐随 bs 平坦」的解释，而该观测本身可能被 prefill 交错
#     污染（见下），**未被干净证据支持**。可引用的是上面这句行为结论，不是它的机制解释。
#   按 decode 批大小分组统计 `gen throughput`。该字段语义（全批总生成吞吐）有**独立确认**：
#     调用方从客户端 SQLite 数 15,789 次调用 → 每调用 10.42 输出 token × 6.4 calls/s ≈ **67 tok/s**；
#     本侧从调度器读 gen throughput → **70–85 tok/s**。两条路径代码无共同来源。
#     ★ 方向也符合预期：本侧略高，因为窗口排除了部分空闲/纯 prefill 时段，而 6.4 calls/s 是
#       端到端持续值。**若反过来（本侧低于客户端），才说明有一侧口径错。**
#     ⚠ 它确认的是**字段语义**，不是**容量**。
#     bs      1     2     3     4     5     6     7     8     9    10    11    12
#     总tok/s 70.1  75.6  75.5  81.2  77.8  85.2  78.1  76.4  75.5  77.8  79.9  79.6
#     样本     13     7     9     5     8    11     7    13    14    28    42   143
#   表面看：bs 涨 12 倍，总吞吐只 +13.5%。
#   ⚠⚠ **但这张表原则上回答不了「批处理是否摊薄」，无论它长什么样——bs 不是受控变量。**
#     并发固定、闸固定，所以 bs 绝大多数时间贴在 10–12（bs=12 占 46%）；
#     bs=1–3 各只有 30–35 个样本，出现场合是**启动 / 队列抽干 / 批量完成后的瞬态**，
#     即**不同的运行状态**，而不是「同一状态下把 batch 调小」。
#     于是 bs 与队列深度、prefill 密度、运行阶段**共变**。这从来不是单变量实验。
#     ⇒ 要回答摊薄，必须**主动扫 bs**（固定并发逐档改闸，或离线压测），那是另一件事，
#       且不影响任何部署决策。**在那之前，「零摊薄」既未被证实也未被证伪。**
#   ⚠ 此处曾有一行「ms/序列 恒定 12–14ms」，**已删**：该量 = 单步ms÷bs ≡ 1000÷总吞吐，
#     与上一行是同一句话的两种写法、**不是独立证据**；且标签会被读成「每序列每 token 12–14ms」，
#     而实际 bs=12 时每序列每 token 约 150ms。
#
#   已逐一排除的「杠杆」（都指向"让更多请求同时在跑"，而这不影响总产出）：
#     · mamba 槽位 64→128 —— 池占用与 running **脱钩**（running=1 时 0.555、running=11 时 0.370），
#       谈不上被并发打满，加槽无用。
#     · `max_running_requests` 12→25 —— decode 时闸确实顶满（bs=12 占 48.6%），
#       **但顶满也没用**：客户端并发 16→32 时在飞请求远超闸值、队列常驻 20+，
#       而端到端吞吐只动 +0.4%（见下 ★ 的 A/B）。抬闸只是让更多请求进到批里等，
#       没有证据表明总产出会上升。
#       ⚠ 此处曾写「抬闸会让单步更长、总吞吐仍是这条平线」，**已删**——
#         「单步更长」依赖已撤回的 step_ms 推理，「平线」是已降级为佐证的观测。
#     · 客户端并发 16→32 —— 实测 A/B：416.1 → 417.8 POST/分（+0.4%，噪声内）。
#     · `--cuda-graph-bs` 放宽 —— 断层不在 bs=8：**bs 1–8 本就有 graph 覆盖**
#       （`Capture cuda graph bs [1..8]`），覆盖内同样不缩放。整条曲线从头就平。
#     · 换 mamba backend —— **本镜像内无 flashinfer**（`ModuleNotFoundError`，实测），
#       而 `_handle_mamba_backend()` 选 flashinfer 不可用时直接 raise、**不回退**。
#       故本机实际只有 `triton` 一个可用值。
#
#   ★ **唯一独立、可引用的证据：端到端单变量 A/B**
#     客户端并发 16 → 32，实测 **416.1 → 417.8 POST/分（+0.4%）**，而 A 段逐分钟波动 ±8%。
#     两次窗口等长（各 11 分钟）、同在单个 r1 step 内部、同采样臂、同配方。
#     **不依赖任何日志窗口语义。** ⇒ 并发超过服务端准入闸之后再加并发，收益为零。
#
#   ⚠⚠ **能力无法从本日志读出——不要引用任何「decode 天花板」数字。**
#     `gen throughput` 按 `decode_log_interval=40` 窗口平均，**每个窗口都混着交错的 prefill**：
#       · 最大 109.5 是恰好几乎没有 prefill 的那个窗口 ⇒ **高估**，不是能力；
#       · 中位 72.6 含大量 prefill 交错 ⇒ **低估**；
#       · 各档最大值在 87.8–109.5 波动、**不随 bs 单调趋于平台** ⇒ 噪声主导。
#     ⇒ 两端一高估一低估，**中间没有任何分位配称「decode 能力」**。
#       曾写「约 80 tok/s 封顶」「区间 73–107 tok/s」，**均已撤回**。
#       写成区间比写错的点估计更隐蔽地误导——它让人以为真值在区间内。
#
#   ⚠⚠ **不要再对 step_ms 做回归**：`step_ms := 1000·bs / throughput` 是**恒等式**。
#     若吞吐近似恒定，则 step_ms = (1000/T)·bs **自动**是过原点直线。所以
#     「截距≈0」「斜率 b=1000/T」「R²≥0.985」「1/b 对上吞吐分位、四档自洽」
#     **全部是恒等推论，不构成第二个事实**。曾据此写「固定开销 ≤8%」，**已撤回**。
#     ⇒ 教训：**先问一个量是不是由另一个量定义的**；两个符号不代表两次独立测量。
#
#   ★★ 完整表述需三句，缺一不可：
#     ① 服务端在 TP8 拓扑内没有可调项（五个杠杆已逐一排除，见上）；
#     ② 「降低每步固定开销」类改造（换拓扑 / 放宽 graph / 换通信后端）收益极小——
#        依据是 README 的端到端聚合实测（TP8 比 TP4 +60%/+39%/+30%），**不是**上面那个回归；
#     ③ 「减少每调用 token 数」是独立杠杆、按比例生效，**但前提是 decode 确实绑定**，
#        而本日志**无法确定 decode 是否绑定**。故「降到 3 token ⇒ 快 3.5 倍」只是**上界**，
#        不是预期值；若 decode 不绑定，收益可能接近零。
#
#   ⇒ 要分开 decode 与 prefill 的能力，**只剩实验一条路**：做一个只改 prefill 工作量、
#     而提示词/seed/采样参数/缓存键全不变的单变量改动，看吞吐动不动。
#
#   ⚠ 若「吞吐随 bs 平坦」为真，则对 27B TP8 是**反常的**（批处理本该摊薄权重读取、近线性缩放）。
#     嫌疑是 mamba/GDN 的**逐序列状态更新**不随批摊薄。
#     但**前提与嫌疑都未验证**（观测可能被 prefill 交错污染），**不要当结论**。
#     若将来确认，提升须动 mamba 实现或换镜像，不是调启动参数能解决的。
MAMBA_CACHE="${MAMBA_CACHE:-16}"
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
    --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size $MAMBA_CACHE \
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
