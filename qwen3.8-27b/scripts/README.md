# 部署配置总索引（拉起包 · 一线一包）

Qwen3.8-27B on 海光 K100-AI ×8。每条配置线可打成一个独立拉起包；
能构建不等于已通过全部验收，镜像、模型和额外资产须按各线 README 准备。公开源码按实际验证状态标注，实验配置不作为生产推荐。
方法论见 [`../docs/方案脚本规范-设计文档.md`](../docs/方案脚本规范-设计文档.md)。

## 厂商 vLLM 0.28.1：17／18／19 的验证状态

| 线 | 默认配置 | 已有历史结果 | 不得据此宣称 |
|---|---|---|---|
| [17](17-C-vllm0281-tp2/README.md) | TP2，BF16，32K | 七项有限功能检查通过 | 1M、完整模型精度、长期稳定性或生产全量验收通过 |
| [18](18-C-vllm0281-tp4/README.md) | TP4，BF16 权重／KV，1M 配置上限 | 初始化与六项有限 API 检查通过；标准周期未完成 | 完整性能矩阵有效、满 1M 请求已验证或比 01 更快 |
| [19](19-C-vllm0281-tp8/README.md) | TP8，配置同上，跨 socket | 初始化与六项有限 API 检查通过；标准周期未完成 | 完整性能矩阵有效、满 1M 请求已验证或比 13 更快 |

18／19 各自的标准负载在 7200 秒执行时限到达时结束，没有取得合格完整性能矩阵。该结果不等于模型不能初始化或不能推理，也不能据此确定超时根因。厂商原配方明确的是 TP2／32K，TP4／TP8 仍按性能实验配置解释。

三线的默认端口都是 `8117`，默认仅监听 loopback；须按归属互证释放已有卡组和端口，实验对照串行运行。18／19 不自动准备 1M 模型农场与 tokenizer，19 另需运行驱动 v4 前提。历史结果不自动覆盖后续共享入口改动；正式 `start/state/stop` 的包级验证是单独检查。

> ## ★ 2026-09-11 状态（以此为准，下方 09-02 表述已过期）
>
> **推荐主力：13 线（bf16 · TP8 全 P2P + custom AR）。**
> 同口径对照 01 线（TP4）：单流 decode **62.1 vs 35.0 tok/s（+77%）**，
> 冷启 prefill 长输入 **+71%**（≥350K 字符：TTFT 66s vs 112s），
> 聚合消化吞吐 +30~60%，服务错误 0 vs 1，**每一项都胜出、无一项回退**。
> 见 [对照定案](../../docs/bench-2026-09-09-选线指南/对照-13线TP8全P2P-vs-01线TP4-新口径.md)。
>
> **INT8 全系（09/10/11/14）暂不可投产**：VMFault 两次复现（19 分 26 秒、37 分钟），
> 低负载、显存充裕时发作，无一致前兆，v4 补丁不是解法（两个子系统）。
> 见 [INT8 VMFault 复测](../../docs/bench-2026-09-09-选线指南/INT8-VMFault-v4后复测.md)。
>
> ~~2026-09-02 状态：推荐主力是 11 + 10 双姿态，09 退居后备~~（**已过期**，
> 保留以记录判断变更轨迹；当时 INT8 的 VMFault 尚未暴露。）

---

## 一、打包脚本怎么用（仓库侧）

```bash
bash deploy-kit/build_package.sh 09        # 只打 09 号线一个包
bash deploy-kit/build_package.sh all       # 各线各打一包
bash deploy-kit/build_package.sh 09 /out   # 指定输出目录
```

产出 `q38-kit-<NN>-<日期>.tar.gz`（28K–152K/包）。每包内容：
统一入口 `up.sh` + 本线目录（serve.sh/README/补丁件）+ 公共骨架
（launch/ensure_ready/机器准备/门禁/契约库）+ S1 的 ACS 自愈件 + 契约文档
+ 按线附件（01/03: dlhook2+1M农场配方；02: 内嵌 01 副本配置与网关件；
05-08: vLLM 补丁组；09/10/11/14: minichain5 补丁链（09/11/14 另含 v122 非贪婪修复）
与量化方案；**13/14: v4 驱动补丁脚本 `patch_v4.sh`**——
dlhook2-sg agent 过滤件已于 2026-09-11 退役，见下方作废说明）。

## 二、打好的包怎么用（现场侧）

```bash
# 传包到目标主机任意目录，然后：
tar xzf q38-kit-09-*.tar.gz && cd q38-kit-09

bash up.sh                     # 拉起本线：①体检 ②S1机器准备 ③④S2-S8 ⑤真实请求冒烟
bash up.sh GPUS=4,5,6,7        # 带参拉起（KEY=VAL 透传，如换卡组/端口）
bash up.sh status              # 观察：实例/端口/接口探活/容器态/重启数/KFD
bash up.sh stop                # 停止（S9 清场 + KFD 校验）

# ★ 同机多实例：同一个包、同一个脚本，只改「哪几张卡 + 哪个端口」，不改任何文件
bash up.sh GPUS=0,1,2,3 PORT=8101      # 实例一（沿用 @name）
bash up.sh GPUS=4,5,6,7 PORT=8102      # 实例二（实例名按端口自动派生为 …-8102）
#   若模型只占 2 卡，同法可在 8 卡机上起 4 个。

# ★ 句柄统一为端口：起服时给的是卡+端口，管理时用同一个端口即可
bash up.sh status  PORT=8102   # 只看该实例（端口/接口/容器态/重启数 + 其拉起阶段 + 日志尾）
bash up.sh logs    PORT=8102   # 只看该实例的容器日志
bash up.sh restart PORT=8102   # 定点重启（卡组与端口从容器实况还原，无需重报）
bash up.sh stop    PORT=8102   # 只停该实例，不触碰同机其它实例
```

* **可观察**：`status` 显示三段——①阶段进展清单（体检/S1/S2/S3/S5/S4S6/S7/S8/冒烟，
  解析自最近一次拉起日志）②实例表（名称/端口/接口探活/容器态/重启数）③各在跑容器的
  docker 日志尾。全程拉起输出落 `logs/<线>-up-<时刻>.log`，实时跟随
  `tail -f logs/latest-up.log`；容器日志 `bash up.sh logs [-f]`。
* **幂等**：`up.sh` 可任意重复执行——S1 幂等安装，拉起前预清**本次要起的那个实例**再起新。
  （2026-09-17 前是「预清本线全部实例」，那会导致起第二个实例时把第一个删掉，已修。）
* **多实例**：实例名未显式给出时按端口派生，Triton 缓存亦按实例隔离
  （`$D/tritoncache-int8-$NAME`），故同一个包起 N 个实例不会互相覆盖。
  拉起日志按实例分开（`logs/<线>-<端口>-up-<时刻>.log`，软链 `latest-<端口>-up.log`）。
* **完整停止**：`up.sh stop` 不管前序 up 跑到哪一步（含中途失败/装载中），
  按标记前缀 **`01ma.cli-model-svc-`** 完整清场并做 KFD 校验。
* **标记体系**：本流程创建/设置的一切带 `01ma.cli-model-svc-` 前缀，可精确识别归属——
  容器名（如 `01ma.cli-model-svc-q38-int8A`）、systemd 单元（`…-acs-clear.service`）、
  `/usr/local/sbin/…-machine_prep.sh` 等脚本、`/etc/sysctl.d/99-…-dcu.conf`。
  旧的无标记 S1 单元会被自动停用接管。
* 首次拉起会自动 docker pull 镜像、自动从 hf-mirror/ModelScope 拉权重（S2）。
* **升级 = 新包覆盖解包到原处**（`tar xzf` 即可）；**勿 `rm -rf`**——
  运行期写入的 `tritoncache*/` 编译缓存要保留（重启秒级恢复的关键）。
* 硬性前提（体检会逐条明说）：DCU 驱动已装（装驱动超出包边界）、`/opt/hyhal` 在位、
  docker 可用（或 sudo NOPASSWD）、网络可达 harbor/hf-mirror
  （不可达时按各线 README 里的完整镜像 URL 离线导入）。

---

## 三、配置总表

> 公开仓仅收录十条精选线（01/02/09/10/11/12/13/17/18/19）；其余编号为内部研究/对照线，不随仓发布。

| 包 | 线 | 卡 | 引擎 | 端口 | 上下文 | 一句话定位与代表数据 |
|---|---|---|---|---|---|---|
| **17** ⚠ | C | **2** | 厂商 vLLM 0.28.1（BF16） | 8117 | 32K | 有限功能配置：七项历史检查通过，不是生产全量验收 |
| **18** ⚠ | C | **4** | 厂商 vLLM 0.28.1（BF16） | 8117 | 1M（配置上限） | TP4 性能实验：完整标准周期未完成，无合格性能矩阵 |
| **19** ⚠ | C | **8** | 厂商 vLLM 0.28.1（BF16） | 8117 | 1M（配置上限） | TP8 性能实验：同上，另需运行驱动 v4 前提 |
| **13** ★★ | A | **8** | sglang(bf16·TP8 全P2P) | 8113 | 1M | **推荐主力**：单流 decode **62.1 tok/s**、冷启 prefill 长输入较 01 线 **+71%**；需 v4 补丁 |
| **09** ⛔ | A | 4 | sglang(INT8) | 8109 | 262K/1M | INT8 全系暂不可投产（VMFault）：单流 d86 p2300，8路 148 |
| **11** ⛔ | A | 4 | sglang(INT8) | 8111 | 1M | INT8 全系暂不可投产（VMFault）：单流 50.0 / 聚合 97.1 |
| **12** ⛔ | A | 4 | v1.3.1整包 | 8112 | 458K | INT8 全系暂不可投产（VMFault）：120K冷TTFT 59s；离线镜像 |
| **10** ⛔ | A | 4 | sglang(INT8) | 8110 | 262K | INT8 全系暂不可投产（VMFault）：8路 243，p3990 |
| **01** ★ | A | 4 | sglang(bf16·v2) | 8101 | 1M | **四卡 bf16 主线**（13 线的省卡版）：单流 decode 35.0 tok/s |
| **02** ★ | A | 8 | sglang×2+网关 | 8100 | 1M×2 | 双副本粘性网关：≈01×2，conc8 碾压 TP8 |

（d=decode tok/s，p=prefill tok/s；09/10/13/14 目录名内嵌单流与聚合实测值，看名即选型）

> ### ~~⚠ custom AR 在本机从未启用（2026-09-09 实测更正）~~ ★ 已于 2026-09-11 撤回
>
> **那条更正是在 0811 树上做的观察，迁到 0828 树后不成立。**
> 实测：13 线 `custom_AR_ranks=8`、01 线 `custom_AR_ranks=4`，
> 拉起日志均出现 `using PCIe's custom allreduce`，**custom AR 确实在用**。
> 单变量 A/B（两份服务只差 `--disable-custom-all-reduce`）：
> 服务端单流 `gen throughput` 开 41.7~56.7 vs 关 23.0~28.2 = **2.15×**，
> `accept rate` 两侧持平（排除投机解码的解释）。
> 01/13 线均已加 `@attest custom all-reduce启用` 自证项，拉起时校验。
> 见 [A/B 定案](../../docs/bench-2026-09-09-选线指南/custom-all-reduce-AB定案.md)。
>
> **TP4 还是 TP8？**（★ 2026-09-11 生产口径实测，各 2 小时，
> 输入 40K–350K 字符/均值约 150K/命中率 92~94%；先验明两线命中率均值
> 92.7% vs 92.5% 方才比较）
>
> | | 01 线 TP4 | **13 线 TP8 全P2P** | 增益 |
> |---|---|---|---|
> | 单流 decode P50 | 35.0 tok/s | **62.1 tok/s** | **+77%** |
> | 冷启 prefill 80–150K 字符 | 1,757 tok/s | **2,665 tok/s** | **+52%** |
> | 冷启 prefill ≥350K 字符 | 936 tok/s | **1,601 tok/s** | **+71%** |
> | ≥350K 字符 TTFT | 112.10 s | **65.70 s** | −41% |
> | 聚合消化 4/8/16 路 | 2,830/10,927/12,294 | **4,536/15,197/16,036** | +60/+39/+30% |
> | 服务错误 | 1（510 请求） | **0**（729 请求） | — |
>
> **13 线每一项都胜出、无一项回退。** 两条规律：**输入越长优势越大**
> （+26%→+71%，长上下文正是编程助手主场）；**并发越高相对优势越小**
> （+60%→+30%，两线都接近各自 prefill 上限）。
> 代价是多占 4 张卡。
>
> ~~（2026-09-09 旧口径：bf16 28.95/65.21 → 45.45/93.05，+57%/+43%）~~
> **与新口径不可混比**——旧口径输入均值仅 68,889 字符、S3 命中率 52~70%。
>
> ~~TP8 三线（03/13/14）都必须挂 `dlhook2-sg.so` agent 过滤并关 custom-AR~~
>
> ★ **2026-09-11 作废**：那条"必须"的前提是驱逐活锁无解。根因已定位并用
> 1 字节驱动补丁（v4）修掉——内核 `drivers/pci/p2pdma.c` 的
> `host_bridge_whitelist[]` 只列 Intel 根桥，跨 socket 的
> `pci_p2pdma_distance_many()` 恒返回 −1，导致 `gpu_dma_buf_attach` 清零
> `attach->peer2peer`，dmabuf map 只给 GTT，每次 map 把导出方 BO 踢出 VRAM，
> 形成活锁。**13/14 线现为「全 P2P + 开 custom-AR」**，
> `dlhook2-sg.so` 已退役（留着反而损失 76% 吞吐），
> custom-AR 单变量 A/B 值 **2.15×**。新硬前提是**驱动必须打 v4 补丁**
> （serve.sh 启动前字节自检，未打拒启；`stages.sh` 另做来源+运行时双重检查）。
> 见 [第十七轮](../../docs/hycu.ko符号级还原-第一轮.md)、
> [三方对照](../../docs/bench-2026-09-09-选线指南/v4补丁-三方对照-TP4-vs-TP8SHM-vs-TP8全P2P.md)。
>
> ⚠ **驱动不可 `rmmod`/`insmod` 重载后投产**：重载会让 KFD 驱逐机制永久打摆、
> 性能塌约 550 倍。换驱动或打补丁一律「改文件 → 重启机器」。
> 实证见 [`docs/hycu.ko符号级还原-第一轮.md`](../../docs/hycu.ko符号级还原-第一轮.md) 第九轮。

**包级验收（2026-08-31，验证机B，每线：解包→up→S1-S8→真实请求→stop）**：

| 线 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 结果 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 至就绪+冒烟 | 4min | 5min | 3min | 3min | 10min | 10min | 12min | 12min | ~10min | ~9min |

十线全部以拉起包形态跑通"体检→S1→S2..S8→模型真实回复→完整停止"。
验收中修复并回归的三处：03 容器名参数化、05 的 preflight 路径按包布局、
02 网关端口就绪等待（消除 race）。

## 四、选择指引

* **交互/低延迟 → 09；批量/吞吐 → 10**。同镜像同权重，可同机不同卡组并存
  （09 卡0-3 + 10 卡4-7），外部网关按流量类型分流。
* **必须 1M/请求 或 拒绝量化 → 01**（4卡）/ **02**（8卡双副本）。
* **循环负载实测对比**（23K冷启→+1-2K/轮→350K→压缩回40K，出300 token）：

| 每轮 E2E | 09 | 01 | 02 |
|---|---|---|---|
| @100K | ~4-6s | 17.8s | 16.5s |
| @220K | **5.3s** | 19.7s | ~19s |
| @350K | **10.9s** | 25.4s | 23.7s |
| 冷启 23K | 11.0s | **7.5s** | 7.9s |
| 批量增长段 | 212-541 tok/s | **1028-2739** | 1001-2669 |

  → 逐轮交互型负载选 09（快 3-4×）；prefill 密集型（频繁灌大文档/重建）01/02 反超 2-5×。
* 四项能力（1M/Think/ToolCall/温度）：01-04 全满足；09 满足（1M 端到端已验，
  但 >26 万前缀 prefill 未调优、慢，日常按 262K 用）；10 同 09；05-08 见各线 README。

---

## 五、每条线：参数小结 + 实测数据

> 完整参数以各线 `serve.sh` 为准（本节是速查）；详细依据与踩坑记录在各线 README/NOTES。

### 17 · 厂商 vLLM 0.28.1 · TP2（有限功能配置）

使用按摘要固定的厂商镜像、现有 BF16 权重和 `TRITON_ATTN`，保留 MTP3、细粒度前缀缓存及 reasoning/tool parser；默认 `CTX=32768`、批量 token 上限 4096。2026-10-25 的七项有限功能检查通过，不能将其外推为完整性能、精度或生产验收。来源、版本后缀和边界见 [17 线 README](17-C-vllm0281-tp2/README.md)。

### 18 · 厂商 vLLM 0.28.1 · TP4（性能实验配置）

使用现有只读 1M 模型农场与指定 tokenizer、显式 BF16 KV、chunked prefill 16384 和显存比例 0.90，与 01 作历史口径对照。已有初始化与有限 API 历史结果，但完整标准周期未完成；不填写合格性能矩阵或速度增益。前提与解释边界见 [18 线 README](18-C-vllm0281-tp4/README.md)。

### 19 · 厂商 vLLM 0.28.1 · TP8（性能实验配置）

与 18 使用相同精度、模型、tokenizer 和量具口径，采用跨 socket 八卡并依赖已生效的运行驱动 v4 前提，与 13 作历史口径对照。已有初始化与有限 API 历史结果，但完整标准周期未完成；1M 仅为配置上限。详见 [19 线 README](19-C-vllm0281-tp8/README.md)。

### 09 · INT8-W8A8 + DFlash2 投机 · TP4（低延迟主力）★

**参数**：镜像 `…custom:sglang0.5.12-K100AI-qwen3.8-0828`；权重=ModelScope 官方
`hygon/Qwen3.8-27B-Channel-INT8-w8a8`（58G）+ HF `z-lab/Qwen3.8-27B-DFlash2` 草稿（3.85G，
serve.sh 自动派生原版架构名目录）。纯树 + `minichain5` 迷你补丁链（gfx928 varlen 修复、raw-q8 verifier
+ 去毒 q8split + q16k 长 prefill 桶）+ 厂商 GEMM 调优 JSON（`TRITON_JSON_DIR`）。
fa3 后端、page64、ctx 1048576、chunk/max-prefill 16384、pack min-q **4096**（2048 是雷）、
DFLASH steps1/draft8、cuda-graph bs **1/2/4/8**（bs16×投机=崩）、mamba extra_buffer 16、
`--disable-custom-all-reduce`、watchdog/dist-timeout 7200、skip-server-warmup。
**探活只用 `/v1/models` 或 `/model_info`，禁 `/health`**（该树的 /health 注入生成请求）。

**实测**：单流 decode **85.9**（代码，瞬时 105-113，accept 5.3-5.7）/ 47.0（essay，步/s 18.8）；
96K：decode 33.4-38.7（TPOT 26-30ms）、prefill ~2300（TTFT 43s）；聚合 4路 133.6 / **8路 148.3**；
1M/请求端到端已验（decode@996K=10.3）；循环画像：全周期 TTFT 0.4-2.3s，每轮 E2E 5-11s（出300），
500 token 出 8-19s（接受率随内容 28-90 波动）；冷启 3-10 分钟（含缓存暖）。并发甜点 8 路。

### 11 · INT8 + DFlash2 + custom-AR · TP4（深上下文冠军）★

09 的参数体系升级版（DocPang v30 吸收）：同权重/镜像/minichain5/v122，
关键差异 = **custom all-reduce 开启**（同 socket 四卡；主引擎，短 +40%/深暖 +64%）、
DocPang chat 模板、pack min-q 2048、mem 0.95、graphs 1-8、mamba 32、
max-total-tokens 1M、`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`（投机突发后
384/1M 槽记账误差降为告警）。
实测（卡0-3 弱卡组）：短 58-67 / 64K 63 / 120K 暖 53-77（瞬时 108）/ 8路 145.7；
120K 暖对照 09 的 52.3。**切勿换 Freaksterz SmoothQuant 权重（accept 归零）**；
卡组必须同 socket。详见线内 README。

### 13 · bf16 + NEXTN · TP8 混合传输（bf16 吞吐档）★

01 线的八卡版，改动同 14 线三处（门禁 / `--tp 8` + 关 AR / 混合传输 + agent 过滤）。

**实测**（包级验收）：单流 **45.45** / 并发16 聚合 **93.05**；就绪 462s。
相对 01 线：单流 **+57%**、聚合 **+43%** —— bf16 模型大，八卡收益远大于 INT8 线。

### 10 · INT8-W8A8 无投机 · TP4（高QPS主力）★

**参数**：同 09 镜像/权重（无草稿模型）；cuda-graph bs **1/2/4/8/16**、
mamba cache **48**（每请求≈5 槽 → 9 路上限）、`--max-running-requests 16`；其余同 09。

**实测**：**8路聚合 242.6 tok/s**（比 09 高 64%：批量下 draft 算力反成负担）；
单流 33.1；96K prefill **3990 tok/s**（服务端 25s，全场最快）；并发上限 9 路。

### 01 · bf16 + NEXTN · TP4（全精度主线 v2）★

**参数**：镜像 `…custom:sglang0.5.12-K100AI-qwen3.8-0828`（**0828 树**）；权重
`Qwen3.8-27B` + 1M 软链农场（S2 自动生成）。TP4 同 socket 卡组、**custom-AR 开启**
（`--ipc host`，0828 树已修 0811 树 aiter AR 的 SIGSEGV）、`minichain5n` 链
（gfx928 varlen 修复 + q≤4 直走原生，NEXTN verify 必经）、ctx 1M、page64、
NEXTN **steps2 / draft3**（draft4 在本树深上下文塌方）、mem 0.90、graphs 1-8。

**实测**：短代码 31.5-32.8 / 长文 26.2 / 120K 26.5 冷 29.5 暖（较 v1 全面 +30~60%）；
2026-09-09 干净机器基准：单流 **28.95** / 并发16 聚合 **65.21**。
八卡版见 **13 线**（单流 45.45 / 聚合 93.05，+57%/+43%）。

### 02 · DP2×TP4 + 粘性网关（8 卡）★

**参数**：01 配置 ×2（卡0-3→8101 q38-a，卡4-7→8102 q38-b）+ `router.py` 网关 8100
（4 级粘性键 + 按在飞数均衡）。包内嵌 01 目录，一键 `up.sh` 串起三件。

**实测**：conc8 对 TP8 prefill **2.91×** / decode **2.55×**；会话粘性 3/3（含零配置前缀粘性）；
循环画像与 01 直连逐项同值（**网关开销≈0，粘性不破坏前缀缓存**）：@220K ~19s、@350K 23.7s。

## 六、十步契约（方法论骨架）

**契约定义**在 [`../docs/方案脚本规范-设计文档.md`](../docs/方案脚本规范-设计文档.md)——
每阶段回答：前置条件 / 保证什么 / 怎么验 / 失败怎么办。执行载体（docker/bare/systemd）
是正交维度（`lib/carrier.sh`）。

| 阶段 | 干什么 | 实现位置 |
|---|---|---|
| S1 | 机器级准备（内核参数 + ACS 清除自愈）| `up.sh` 安装 `common/machine_prep.sh` + systemd 自愈 |
| S2 | 宿主环境自举（镜像 docker pull；权重 hf-mirror/`@weights-ms` ModelScope；1M 农场）| `common/ensure_ready.sh` |
| **S3** | ★ 起跑前提校验（numa_balancing、ACS、卡数、模型、孤儿 KFD）| `lib/stages.sh` `stage3_gate()` |
| S4 | 运行环境构造（挂载/补丁/环境变量）| 各线 `serve.sh` |
| S5 | 清理上一实例 | `stage5_cleanup()` |
| S6 | 启动命令 | 各线 `serve.sh` |
| **S7** | ★ 就绪等待（RestartCount 监视；**退出码 0 也算失败**；失败提根因）| `stage7_wait_ready()` |
| **S8** | ★ 运行时自证（KV 池实读、上下文、投机、内核回落；`# @attest` 声明）| `stage8_attest()` |
| **S9** | ★ 收尾与资源归还（停实例 + KFD 归零校验）| `stage9_*()` / `up.sh stop` |
| S10 | 网关（粘性 + 按在飞数均衡）| `common/router.py`（02）|

**编排**：`up.sh` = 体检 + S1 + 调 `common/launch.sh`（S3→S5→S4+S6→S7→S8，
任一步失败自动 S9）+ 真实请求冒烟。S1 由 systemd 开机自愈，S3 每次启动重新断言
（S1 做过 ≠ 现在还成立）。

## 七、完整性契约：包含什么、不含什么

**含**：本线拉起所需的全部我方内容（脚本/补丁/二进制含 SHA256/契约文档；
09/10 附量化方案 `quant/` 供追溯）。

**不含**（获取方式已写进脚本或各线 README）：
1. **Docker 镜像** —— S2 自动 `docker pull`（tag 全 URL 见各 serve.sh 头部 `@image`）；
2. **模型权重** —— S2 自动拉（HF 走 hf-mirror；官方 INT8 走 ModelScope git clone）；
3. 大文件下载受限于目标机 WAN 管径（验证机 系 ≈9.5MB/s，360G≈11h，提前规划）。

## 八、三条铁律（详见仓库 CLAUDE.md 护栏）

1. **投机验证批长 ≤4**：gfx928 厂商 paged-varlen 内核 q≥5 no-write（护栏 1·十二）。
   01-04 的 NEXTN 锁死 `steps3/topk1/draft4`；09 的 draft8 之所以合法，
   是 minichain5 对 q=8 verify 直调 raw paged_attention（260602 legacy ABI 版本锁，几何不符回退 2×q4）+ 其余走修复后的 Triton；2026-09-02 吸收 DocPang v1.2.1 后 245K decode +26.5%。
2. **性能对比必须串行**：同机并发实例互相污染测量（护栏 1·七）。
3. **单流 decode 对比看 步/s 与 tok/步**，不看裸 tok/s（护栏 1·五）。

## 九、目录结构（仓库侧）

```
scripts/
├── README.md            ← 本文件（总索引）
├── up.sh                ★ 拉起包统一入口（打包时置于包根）
├── lib/                 阶段契约 + 载体适配
├── common/              launch.sh / ensure_ready.sh / machine_prep.sh / numa_bind.sh
│                        preflight_acs.sh / router.py / serve_router.sh / fetch_hf_model.sh
├── 01-…/ … 10-…/        编号即配置：serve.sh + README.md（参数依据与实测）+ 附件
└── （判死配置在仓库 lab/configs-archive/，不随包）
```

旧式用法（仓库内直接跑，仍可用）：`bash common/launch.sh <配置目录名> [KEY=VAL…]`；
`TEARDOWN=1` 冒烟模式跑通即收尾。
