# 01 · A 线 · 4 卡 · TP4（★ 4 卡推荐）

| 项 | 值 |
|---|---|
| 卡数 | **4**（同 socket：0-3 或 4-7）|
| 引擎 | **sglang**（0811 镜像）|
| 网关 | 无（单实例直连）|
| 上下文 | **1M**（软链农场 + YaRN）|
| 投机 | NEXTN `steps3/topk1/draft4`（**锁死**，见护栏 1·六/1·十二：q≥5 内核 no-write）|
| 四项必须 | 1M ✓ / Think ✓ / Tool Call ✓ / 温度 ✓ |

**实测**（64K 输入、输出 2048、固定 prompt 口径）：
decode **25.5 tok/s**（验证机A·卡0-3，快态）/ 20.7（验证机B），TTFT ~25-29s，接受长度 2.51。

**机器选择**：优先 **验证机A 的卡 0-3** —— 全场唯一「同 socket + 四张全 400W（0x6210）」
的卡组，比 350W 卡组快约 20%（护栏 1·十）。

```bash
# 拉起（S3 前提校验 → 拉起 → S7 就绪 → S8 自证）
bash ../common/launch.sh 01-A-sglang-tp4
# 换卡组/端口
GPUS=4,5,6,7 PORT=8102 NAME=q38-b bash ../common/launch.sh 01-A-sglang-tp4
# 停止
bash -c ". ../../lib/stages.sh; stage9_teardown q38-sgA"
```

可调旋钮（探索用，默认勿动）：`CTX CHUNK MAXPRE MEM_FRAC SPEC_ALGO SPEC_STEPS/TOPK/DRAFT
MODEL_PATH EXTRA_ARGS EXTRA_ENV_IN EXTRA_MOUNT NUMA_OVERRIDE ENTRY_WRAPPER DFLASH_*`。

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
农场  bash ../patches/model-1m-farm/mk_1m_farm.sh   # launch 会自动做
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。

## 循环画像实测（2026-08-30，同 09 号线的成套测试：23K冷启→+1-2K/轮→350K→压缩回40K，出300）

| 深度 | 尾部轮 TTFT | decode | 每轮 E2E |
|---|---|---|---|
| 100K | 1.07s | 17.9 | 17.8s |
| 220K | 2.1s | 17.0 | 19.7s |
| 350K | 4.01s | 14.0 | 25.4s |

冷启 23K ≈ 7.5-7.9s；40K 重建 ≈ 6.2-6.4s；批量增长段 1001-2739 tok/s（**bf16 线强项**）；
峰值 3K 轮 170-185s。
**对比 09 号线（同画像）：每轮 E2E 是其 3-4 倍（decode 14-19 vs 35-90）；
prefill 侧（冷启/重建/批量段）反超 09 达 2-5 倍。逐轮交互型负载选 09，prefill 密集型选本线。**

**口径修正（常态输出=500 token）**：本线 decode 稳定 14-19（无投机波动），
常态轮 E2E@500tok ≈ TTFT + 500/17 ≈ **31-33s @220K**（换算值，decode 实测稳态）。

## v2（2026-09-03 迁树升级）

底座迁至 0828 树 + custom-AR 开启（11 号线优化迁移）；minichain5n（q≤4 直走原生）；
NEXTN steps2/draft3（draft4 在本树深上下文塌方）。**不再需要 dlhook2-sg 与 1M 农场入口包装。**

| 口径 | v1（0811 树） | **v2** |
|---|---|---|
| 短代码 decode | 20.7-25.5 | **31.5-32.8** |
| 长文 decode | 19.2 | **26.2** |
| 120K decode 冷/暖 | 深段 ~15-16 | **26.5 / 29.5** |

边界：1M 农场需同时挂 $MODELS_ROOT 根（符号链接）；overrides/ 两个 preprocessor json
与 hygon 目录 tokenizer 两件为 0828 树必需；02 号线内嵌本配置：已用 v2 内嵌重新验收通过（2026-09-03）。
旧版脚本保留为 serve.v1-0811tree.sh.bak。
