# 13 · bf16 + NEXTN · TP8 全 P2P —— 01 线的八卡版（2026-09-10 改版）

8 卡横跨两个 socket，Qwen3.8-27B bf16 全精度 + NEXTN 投机。
**01 线的直系八卡版**：同权重、同镜像、同 minichain5n 链、同投机参数，
差异只在并行度。

> ## ★ 2026-09-10 重大改版
>
> 驱逐风暴的根因已定位并用 1 字节驱动补丁（v4）修复，本线由
> 「混合传输 + agent 过滤 + 关 custom-AR」改为 **「全 P2P + 开 custom-AR」**。
>
> 旧版的两条"硬前提"**都已不成立**：
> - ~~`dlhook2-sg.so` agent 过滤是必需件~~ → 已退役，留着反而损失 76% 吞吐
> - ~~custom-AR 必须关闭，跨 socket 必崩~~ → 那个"必崩"就是驱逐风暴本身
>
> 新的硬前提是：**驱动必须打了 v4 补丁**（serve.sh 启动前自检，未打拒启）。

## 与 01 线对比

口径：标准 harness `tests/bench/run_bench.py`（内部基准工装，不随拉起包发布），
同 seed(7)、同语料（60 文件 / 647,894 字符）、各 1.0 小时。
真实负载 prompt:output ≈ 71:1，**prefill 主导**，所以「合计 tok/s」才是有效功率口径。

| S3 并发合计 tok/s | 01（TP4） | 13 旧版（TP8·SHM） | **13 新版（TP8·全P2P）** |
|---|---|---|---|
| 4 路 | 2,031 | 1,653 | **2,912**（+43% / +76%） |
| 8 路 | 2,619 | 2,252 | **3,962**（+51% / +76%） |
| 16 路 | 2,199 | 1,732 | **3,117**（+42% / +80%） |

| 延迟 P50 | 01（TP4） | 13 旧版 | **13 新版** |
|---|---|---|---|
| S1 E2EL | 10.89 s | 8.69 s | **5.40 s（−50%）** |
| S1 TPOT | 27.8 ms | 19.2 ms | **16.6 ms（−40%）** |
| S2 长上下文 TTFT | 36.23 s | 37.29 s | **22.06 s（−39%）** |
| S3 TPOT | 60.5 ms | 66.4 ms | **42.1 ms（−30%）** |

服务就绪 436 s（旧版 462 s）；全程 `evict_process_worker` = 0、`schedule_evict` = 0；
361 请求 / 1.10 小时**服务错误 0 次**。

> ### ⛔ 以下两节的数字均为**旧口径，已作废**（2026-09-17）
>
> 本节及上节用的是**字符**口径（输入 40K–350K 字符、均值约 150K、命中 92~94%）。
> 用户 2026-09-17 确认的唯一权威口径是 **token**：
> **输入 45K–350K token（均值 160K）／输出 500–4K token（均值 800）／
> 输入 cache 命中率均值 95%、最小 85%／涨到 350K 后切 ~70K 全新前缀。**
>
> 字符与 token **不是一回事**——本模型实测约 **1.745 字/token**，
> 「350K 字符」只相当于约 20 万 token，够不到真实工作点的 350K token。
> 本节自己也写着「新旧两套数字不可混比」，这条现在同样适用于第三套口径。
>
> ⇒ 性能数字须用 `bash up.sh benchmark` 重测（唯一标准测试，
> 见 `docs/标准性能测试.md` 与 `docs/拉起包-使用说明.md`）。重测前，下表仅供历史参考。

### ★ 2026-09-11 新口径复测（按生产实际负载）

上表是 09-10 的旧负载口径（输入均值仅 68,889 字符）。按用户给的生产口径
（**输入 40K–350K 字符、均值约 150K、缓存命中 92~94%**）重测，各 2 小时：

| | 13 线（TP8 全 P2P） | 01 线（TP4） | 增益 |
|---|---|---|---|
| **单流 decode P50** | **62.1 tok/s** | 35.0 tok/s | **+77%** |
| 冷启 prefill 40–80K 字符 | 4,045 tok/s | 3,214 | +26% |
| 冷启 prefill 80–150K 字符 | 2,665 tok/s | 1,757 | **+52%** |
| 冷启 prefill ≥350K 字符 | 1,601 tok/s | 936 | **+71%** |
| ≥350K 字符 TTFT | **65.70 s** | 112.10 s | −41% |
| 聚合消化 4/8/16 路 | 4,536 / 15,197 / 16,036 | 2,830 / 10,927 / 12,294 | +60/+39/+30% |
| 服务错误 | **0**（729 请求） | 1（510 请求） | — |

**两条规律**：输入越长 TP8 优势越大（+26%→+71%）；并发越高相对优势越小
（+60%→+30%，两线都接近各自 prefill 上限）。

⚠ **新旧两套数字不可混比**——负载口径完全不同（输入均值 68,889 vs 150,000 字符、
S3 命中率 52~70% vs 92~94%）。

⚠ 「聚合消化 tok/s」**不是 prefill 算力**：分子含命中缓存、未被重算的 token。
真算力只认上表的「冷启 prefill」行。另有两个看似可用实则不可用的指标已被否决，
见 负面结论（内部记录 `docs/bench-2026-09-09-选线指南/负面结论-哪些指标不能当prefill算力.md`，不随拉起包发布）。

完整对照见 对照定案（内部记录 `docs/bench-2026-09-09-选线指南/对照-13线TP8全P2P-vs-01线TP4-新口径.md`，不随拉起包发布）。

**旧版结论「TP4 在真实负载上胜过 TP8」已作废**——那个结论的前提是 TP8 只能走 SHM。

## 相对 01 线只改两处

```
① 放开 serve.sh 里「只允许同 socket 四卡组」门禁
② --tp 4 → --tp 8
   （不再加 --disable-custom-all-reduce；不再挂 dlhook2-sg.so；
     NCCL_P2P_LEVEL 用 SYS 放开全机 P2P）
```

## 镜像与权重

- 镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828`
- 权重：1M 农场 `Qwen3.8-27B-1M`（符号链接目录，**必须同时挂 `$MODELS_ROOT` 根**）
- tokenizer：借 hygon INT8 目录的两件（同词表，补 think 结束符）

## 硬性边界

1. **驱动必须打 v4 补丁，否则本线起不来。**
   未打补丁时 TP8 全 P2P 会触发驱逐活锁（实测驱逐 482 次/15s、
   BO 搬移 23 万次/15s、服务自杀）。serve.sh 已加启动前字节自检，未打直接拒启。
   打补丁：`sudo bash patches/patch_v4.sh apply`，**然后必须重启机器**。

   根因：内核 `drivers/pci/p2pdma.c` 的 `host_bridge_whitelist[]` 只列 Intel 根桥，
   海光不在其中 ⇒ 跨 socket 的 `pci_p2pdma_distance_many()` 恒返回 −1
   ⇒ `gpu_dma_buf_attach` 清零 `attach->peer2peer` ⇒ dmabuf map 只给 GTT
   ⇒ 每次 map 把导出方 BO 踢出 VRAM ⇒ 等驱逐 fence ⇒ 排驱逐
   ⇒ `restore_process_bos` 重做 map ⇒ 活锁。
   详见 第十七轮（内部记录 `docs/hycu.ko符号级还原-第一轮.md`，不随拉起包发布）。

2. **custom-AR 应当开启**（与旧版相反）。单变量 A/B：服务端单流
   `gen throughput` 开 41.7~56.7 vs 关 23.0~28.2 = **2.15×**，`accept rate`
   两侧持平（排除投机解码解释）。
   见 A/B 定案（内部记录 `docs/bench-2026-09-09-选线指南/custom-all-reduce-AB定案.md`，不随拉起包发布）。

3. **驱动不可 rmmod/insmod 重载后投产。** 重载会让 KFD 驱逐机制永久打摆、
   性能塌约 550 倍（第九轮实证，内部记录 `docs/hycu.ko符号级还原-第一轮.md`，不随拉起包发布）。
   换驱动/打补丁一律「改文件 → 重启机器」。

4. **v4 补丁会被驱动包重装静默覆盖。** `hycu.ko` 不走 dkms，
   由 hymgr 在开机约 3 分钟后 insmod `/usr/local/hyhal/dkms/hycu.ko`——
   正是被打补丁的那个文件。驱动包一旦重装/升级，补丁消失且**不报任何错**，
   只表现为性能悄悄掉回 SHM 水平。serve.sh 的自检就是为此。

5. 就绪探测用 `/v1/models` 或 `/model_info`（`/health` 会注入生成请求）。

6. 其余边界承 01 线：1M 农场符号链接、overrides 两个 preprocessor json、
   NEXTN steps 2 / draft 3（draft4 在本树深上下文塌方）。

7. **Think 默认开启，关闭用 `enable_thinking`——不是 `thinking`。**
   ```jsonc
   // 关闭思考（content 直接给答案）
   "chat_template_kwargs": {"enable_thinking": false}
   ```
   不传任何 override 时**默认就思考**：`content` 返回 null，`max_tokens` 全部进
   `reasoning_content`——短 `max_tokens` 场景下会表现为「只出思考、不出答案」。
   实测（2026-09-14，本线线上）：`{"enable_thinking": false}` → `content="0.5"`、
   `finish_reason=stop`、4 token；而 `{"thinking": false}` **无效，仍然思考**。
   与 11 线 README 的记载一致（同为 Qwen3.8 底座）。

   > ⚠ **`{"thinking": true/false}` 是 DeepSeek-V4 的写法，不适用于本线。**
   > 两个模型的 chat template 参数名不同，**勿跨模型套用**。误用该键名时服务
   > 既不报错、也不生效，表现为「参数明明传了却仍然思考」——最容易被误判成
   > 模型或权重出了问题，实际只是键名不对。

## 何时用这条线

- 需要 **bf16 全精度**且要吞吐 → 选本线（相对 01 线是纯增益，只是启动慢）
- 卡不够 8 张、或驱动没打 v4 补丁 → 回 01 线
- **不要选 INT8 八卡版（14 线）**：v4 之后已复测两次，VMFault 仍在
  （存活 19 分 26 秒 / 37 分钟，均在低负载、显存充裕时发作，无一致前兆）。
  v4 修的是 dmabuf 驱逐活锁，与 INT8 内核路径是两个子系统，本就不该期待它能修。
  INT8 全系（09/10/11/12/14）目前均不建议投入生产。

---

## 0.5.18 能带来什么（2026-09-17 实测盘点，结论：净收益为负）

用两版镜像的**真实 `--help` 参数面**对比（各 `rc=0`，1716 vs 2364 行），
外加逐项进镜像核实。0.5.18 净增 18 个参数，**0.5.12 独有集为空**——升级不丢东西，
本线现传的四个参数（`--speculative-algorithm` / `--speculative-num-steps` /
`--mamba-scheduler-strategy` / `--attention-backend`）全都还在。

### ★ 一个要纠正的说法：mamba radix cache **不是** 0.5.18 的新东西

曾有说法称「0.5.12 完全没有 mamba radix cache」。**那是错的**，成因是拿
0.5.18 的标识（`mamba_radix_cache_strategy` / `uses_mamba_radix_cache`）
去 0.5.12 里搜——参数**改了名**，机制本身一直在：

| | 0.5.12 | 0.5.18 |
|---|---|---|
| 实现文件 | `mem_cache/mamba_radix_cache.py` **1294 行 / 74 个 def** | 同名文件 **1427 行 / 79 个 def** |
| 参数名 | `--mamba-scheduler-strategy` | `--mamba-radix-cache-strategy`（旧名保留为 deprecated） |
| `auto` 解析 | 硬编码落 `no_buffer`（旁注 `TODO: when extra_buffer is more verified`） | 按架构分档；`Qwen3_5ForConditionalGeneration` 在 `_MAMBA_RADIX_CACHE_ARCHS` 与 `_MAMBA_EXTRA_BUFFER_ARCHS` ⇒ 落 `extra_buffer` |
| 启用门 | `scheduler.py:812` 的 `hybrid_gdn_config is not None`（本模型是 hybrid GDN，命中） | 架构名表 |

**本线与 01 线本来就显式传 `--mamba-scheduler-strategy extra_buffer`，
不走 `auto`，因此早已在用。** 且有运行时证据：`~/ab_logs/q38_mamba16_baseline.log`
的 server_args 转储为 `mamba_scheduler_strategy='extra_buffer'`——那是经过全部
`_handle_*` 后处理后的**最终值**，即确实挺住并生效，不只是「参数传了」。

> 曾担心投机场景会把它静默改回 `no_buffer`：查实**不会**。两处改写分别在
> 「策略为 `auto` 时」（我们显式传值，不命中）与 `_handle_pd_disaggregation()`
> 的 decode 分离分支（我们 `disaggregation_mode='null'`，不命中）。

⇒ 升级在这条上**只换来一个增量版本（+133 行）与更合理的 `auto` 默认**，
而我们本就不依赖 `auto`。**不构成收益。**

### 逐条盘点

| 项 | 是不是收益 | 依据 |
|---|---|---|
| mamba radix cache | **否**，早已在用 | 见上 |
| `linear_attn_backend` 3→8 个取值 | **否，海市蜃楼** | `flashinfer`/`helion`/`flash_kda`/`flashkda`/`cutedsl` 在两版镜像里**全部没装**，唯一可用仍是 `triton` |
| GDN 比值 3 融合 | **仅维护性** | +3%(8K)/+2%(32K) 在 0.5.12 上**已经拿到**。差别是实现代价：0.5.12 闸为硬编码 `in [1, 2, 4]`、内核 310 行无 `V_POW2`，需**两个**补丁；0.5.18 原生带 `V_POW2`，只需改白名单一行 |
| CUDA graph 分相控制（8 个新参数） | **可能**，未验 | 0.5.12 只有合一的 `--cuda-graph-bs`；0.5.18 可只调 decode 侧。⚠ 但本 README 记载的五个杠杆里，「放宽 `--cuda-graph-bs`」当年是被**排除**的（bs 1–8 本就有 graph 覆盖、整条曲线从头就平）——分相能否翻案没有证据 |
| `--speculative-use-rejection-sampling`、`--speculative-draft-kv-cache-dtype` | **可能**，未验 | 通用投机参数，可能影响 NEXTN 接受率与显存 |
| 四个 `--speculative-dspark-*` | 否 | DSv4 线的东西，与本模型无关 |

### 结论

**本线天花板在 decode（约 80 tok/s 封顶，五个杠杆已逐一排除），
而上述没有任何一条被实测证明能突破它。** 叠加该镜像当前**输出为非确定性乱码**
（16 线八轮实机，详见其 README），**净收益为负**——还要背五个自写补丁。

**⇒ 01 线与 13 线均维持 0.5.12 基版，不升级。** 待镜像修好且有实测收益再议。
