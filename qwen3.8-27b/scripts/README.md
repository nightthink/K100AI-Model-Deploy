# 部署配置总索引（拉起包 · 一线一包）

Qwen3.8-27B on 海光 K100-AI ×8。十条配置线，每条线打成一个独立拉起包；
现场解包即用，除 **镜像** 与 **模型权重**（首次拉起自动获取）外全自包含。
方法论见 [`../docs/方案脚本规范-设计文档.md`](../docs/方案脚本规范-设计文档.md)。

> 2026-09-02 状态：**推荐主力是 11（INT8 深上下文冠军，AR 开）+ 10（INT8 高QPS）双姿态**，09 退居后备，
> 已在 验证机B 以拉起包形态驻场运行并通过 S8 自证与真实请求验收。

---

## 一、打包脚本怎么用（仓库侧）

```bash
bash deploy-kit/build_package.sh 09        # 只打 09 号线一个包
bash deploy-kit/build_package.sh all       # 十条线各打一包
bash deploy-kit/build_package.sh 09 /out   # 指定输出目录
```

产出 `q38-kit-<NN>-<日期>.tar.gz`（28K–152K/包）。每包内容：
统一入口 `up.sh` + 本线目录（serve.sh/README/补丁件）+ 公共骨架
（launch/ensure_ready/机器准备/门禁/契约库）+ S1 的 ACS 自愈件 + 契约文档
+ 按线附件（01/03: dlhook2+1M农场配方；02: 内嵌 01 副本配置与网关件；
05-08: vLLM 补丁组；09/10: minichain5 补丁链（09 另含 v122 非贪婪修复）与量化方案）。

## 二、打好的包怎么用（现场侧）

```bash
# 传包到目标主机任意目录，然后：
tar xzf q38-kit-09-*.tar.gz && cd q38-kit-09

bash up.sh                     # 拉起本线：①体检 ②S1机器准备 ③④S2-S8 ⑤真实请求冒烟
bash up.sh GPUS=4,5,6,7        # 带参拉起（KEY=VAL 透传，如换卡组/端口）
bash up.sh status              # 观察：实例/端口/接口探活/容器态/重启数/KFD
bash up.sh stop                # 停止（S9 清场 + KFD 校验）
```

* **可观察**：`status` 显示三段——①阶段进展清单（体检/S1/S2/S3/S5/S4S6/S7/S8/冒烟，
  解析自最近一次拉起日志）②实例表（名称/端口/接口探活/容器态/重启数）③各在跑容器的
  docker 日志尾。全程拉起输出落 `logs/<线>-up-<时刻>.log`，实时跟随
  `tail -f logs/latest-up.log`；容器日志 `bash up.sh logs [-f]`。
* **幂等**：`up.sh` 可任意重复执行——S1 幂等安装，拉起前自动预清本线旧实例再起新。
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

| 包 | 线 | 卡 | 引擎 | 端口 | 上下文 | 一句话定位与代表数据 |
|---|---|---|---|---|---|---|
| **09** | A | 4 | sglang(INT8) | 8109 | 262K/1M | 低延迟（后备）：单流 d86 p2300，8路 148 |
| **11** ★ | A | 4 | sglang(INT8+AR) | 8111 | 1M | **主力**：120K 暖 53-77（峰 108），短 58-67，8路 146 |
| **12** | A | 4 | v1.3.1整包 | 8112 | 458K | **深prefill冠军**：120K冷TTFT 59s（11=85-102s），350K decode 34.5；离线镜像 |
| **10** ★ | A | 4 | sglang(INT8) | 8110 | 262K | **高QPS主力**：8路 243，p3990 |
| **01** ★ | A | 4 | sglang(bf16) | 8101 | **1M** | bf16 全能保守线：d19-25，prefill 侧最强 |
| **02** ★ | A | 8 | sglang×2+网关 | 8100 | 1M×2 | 双副本粘性网关：≈01×2，conc8 碾压 TP8 |
| 03 | A | 8 | sglang | 8100 | 1M | TP8 混合传输，仅单用户长输出 |
| 04 | A | 4 | sglang | 8100 | 1M | 基线对照 / 最简回退 |
| 05 | C | 4 | vLLM | 8100 | 512K | C 线生产（短输入强，长输入崩塌）|
| 06 | C | 4 | vLLM | 8100 | 可传 | 05 的参数化副本构件（DP 拼装用）|
| 07 | C | 8 | vLLM | 8108 | — | C 线 TP8 混合传输 |
| 08 | C | 8 | vLLM | 8108 | — | 07 的 Tree 取舍版（prefill 快/并发慢）|

（d=decode tok/s，p=prefill tok/s；09/10 目录名内嵌单流与聚合实测值，看名即选型）

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

### 10 · INT8-W8A8 无投机 · TP4（高QPS主力）★

**参数**：同 09 镜像/权重（无草稿模型）；cuda-graph bs **1/2/4/8/16**、
mamba cache **48**（每请求≈5 槽 → 9 路上限）、`--max-running-requests 16`；其余同 09。

**实测**：**8路聚合 242.6 tok/s**（比 09 高 64%：批量下 draft 算力反成负担）；
单流 33.1；96K prefill **3990 tok/s**（服务端 25s，全场最快）；并发上限 9 路。

### 01 · bf16 + NEXTN · TP4（1M 保守线）★

**参数**：镜像 `…custom:sglang0.5.12-…-deepseekV4-0811`；权重 `Qwen/Qwen3.8-27B` +
1M 软链农场（S2 自动生成）。TP4 同 socket 卡组、ctx 1M、chunk 32768、max-prefill 45000、
page64、NEXTN **steps3/topk1/draft4（锁死，铁律）**、`--disable-custom-all-reduce`（必须）、
`SGLANG_USE_CUDA_IPC_TRANSPORT=1`（仅同 socket 安全）、AITER_LINEAR_ATTN=1、mem 0.90。
依赖包内 `dlhook2-sg.so`。

**实测**：essay decode 19.2（步/s 7.62；无投机 16.68）；历史天花板 20.7 tok/s@64k；
循环画像：冷启 23K **7.5s**、增长段 **1028-2739 tok/s**（prefill 侧全场最强）、
每轮 E2E 17.8-25.4s；bf16 数值无量化顾虑；41 天生产验证史。

### 02 · DP2×TP4 + 粘性网关（8 卡）★

**参数**：01 配置 ×2（卡0-3→8101 q38-a，卡4-7→8102 q38-b）+ `router.py` 网关 8100
（4 级粘性键 + 按在飞数均衡）。包内嵌 01 目录，一键 `up.sh` 串起三件。

**实测**：conc8 对 TP8 prefill **2.91×** / decode **2.55×**；会话粘性 3/3（含零配置前缀粘性）；
循环画像与 01 直连逐项同值（**网关开销≈0，粘性不破坏前缀缓存**）：@220K ~19s、@350K 23.7s。

### 03 · bf16 TP8 混合传输

**参数**：同 01 树；TP8 全 8 卡；`NCCL_P2P_LEVEL=PXB + NCCL_ALGO=Ring`（同 socket 走 VRAM P2P、
跨 socket 走 SHM）+ `dlhook2-sg.so`（按 socket 过滤 agent）；NEXTN 同锁定值；ctx 1M。

**实测**：定位=单用户长输出独占 8 卡的例外场景；并发 ≥4 时被 02 碾压（见 02 数据）。
混合传输依据见 `docs/复盘三-52ms活锁根因与TP8混合传输.md`。

### 04 · bf16 TP4 基线对照

**参数**：01 的朴素版（无调优 flag）：TP4、ctx 1M、NEXTN 锁定值、`NCCL_P2P_LEVEL=PHB`、mem 0.90。

**实测**：作为 A/B 的对照基线与最简回退；无独立性能榜（对照用途）。

### 05 · C 线 vLLM TP4（GDN all 模式）

**参数**：镜像 `…custom:vllm-ubuntu22.04-dtk26.04-hy3-0706`（vLLM 0.21）；卡4-7；
YaRN factor 2.0（512K）、MTP 投机 num=3、PIECEWISE cudagraph、GDN all 模式、
mamba-block 8192、custom AR **启用**（故 serve 前强制 ACS 体检——没清干净会以 0.05 tok/s
静默起来）；挂包内 8 个 vLLM 补丁 .py（GDN 缓存/统一注意力）。

**实测**：C 线生产位：短输入强、长输入崩塌；512K 上限（缺 1M）。历史数据见线内 NOTES。

### 06 · C 线参数化副本构件

**参数**：05 的可参数化版（GPUS/PORT/NAME/MAXLEN/YARN_FACTOR/MAMBA_ALL 全可传）；
1M 需 `YARN_FACTOR=4.0 MAMBA_ALL=0`（TP4 上 1M 与 all 模式缓存显存互斥）。

**实测**：构件性质，用于拼 C 线 DP2；性能同 05 参数化外推。

### 07 · C 线 vLLM TP8 混合传输

**参数**：05 同树；TP8；`NCCL_P2P_LEVEL=PXB + ALGO=Ring` + `dlhook2.so`；
`--disable-custom-all-reduce` 必须（自研 AR 跨 8 rank IPC 绕不过活锁）；MTP 投机。

**实测**：集合带宽修复的受益者（见 08）；整机单实例场景。

### 08 · 07 的 Tree 取舍版

**参数**：07 基础上的 Tree 注意力取舍（prefill 快 / 并发慢）。

**实测**：集合带宽 1.5 → **5.7 GB/s（3.75×）**，正确性 27/27。取舍依据见线内 README。

---

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
