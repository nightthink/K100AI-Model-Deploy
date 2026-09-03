# 09 · INT8-W8A8 + DFlash2 投机 · TP4（低延迟档）

**目录名里的数字**：单流 decode 86 tok/s / prefill 2300 tok/s；聚合最高 decode 148 tok/s（8 路）/ prefill 同 2300（分块 prefill 串行流水，并发下不叠加）。

| 项 | 值 |
|---|---|
| 卡数 | **4**（同 socket 卡组：0,1,2,3 或 4,5,6,7）|
| 引擎 | sglang 0.5.12+das（K100AI 专属树）|
| 网关 | 无（单实例；与 10 号线双姿态部署时由外部网关分流）|
| 镜像 | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828`（serve.sh 会自动 docker pull）|
| 权重① | ModelScope `hygon/Qwen3.8-27B-Channel-INT8-w8a8`（58G，S2 自动 git clone）|
| 权重② | HF `z-lab/Qwen3.8-27B-DFlash2` 草稿模型（3.85G，S2 走 hf-mirror）|
| 上下文 | 262144 默认；**1M/请求已验证可达**（decode@996K = 10.3 tok/s，深前缀 prefill 慢，见边界）|

## 实测（验证机B，scan3 口径，temperature=0）

| 场景 | decode | 备注 |
|---|---|---|
| 单流·代码 | **85.9 tok/s**（瞬时 105-113，accept 5.3-5.7）| |
| 单流·长文 essay | 47.0 tok/s（步/s 18.8 ±0.1%）| 接受长随内容可预测性波动 |
| 96K 上下文 decode | 33.4-38.7 tok/s（TPOT 26-30ms 达标）| |
| 96K prefill | ~2300 tok/s（TTFT 43.1s，暖服务器）| |
| 4 路聚合 | 133.6 tok/s | 全部文本体检通过 |
| 8 路聚合 | **148.3 tok/s** | 并发甜点 8 路 |

## 一键

```bash
bash common/launch.sh 09-A-int8-tp4-spec-单流d86p2300-聚合d148p2300
```

## 硬性前提 / 边界（都已实证，别踩）

1. DCU 驱动已装且宿主有 `/opt/hyhal`（挂进容器，缺它 torch 报 No HIP GPUs）。
2. 就绪探测只能用 `/v1/models` 或 `/model_info`；**禁用 `/health`**（该树的 /health 会注入生成请求，冷启动期轮询会堆积成自杀链）。
3. **测 TTFT 前先发一个小请求暖场**（慢速 tokenizer 首载需数分钟）。
4. cuda-graph 只到 bs8（bs16×投机=调度器崩溃）；并发甜点 8 路。
5. **切勿挂厂商完整 runtime_patch 链**（TP-gather 钩子使 tp>1 每步分钟级）。本线只用 `minichain5/`（varlen 修复 + raw-q8 verifier〔2026-09-02 起，245K decode +26.5%〕 + q16k 桶 + 厂商 GEMM JSON），另挂 `v122/` 三文件修复 temperature>0 打死服务的缺陷（零贪婪性能代价）。
6. draft 的 config 必须是原版 `DFlash2DraftModel`（serve.sh 自动派生 `-v2` 目录处理）。

## 选择指引

交互/低延迟流量选本线；高 QPS 批量选 **10 号线**（聚合 242.6，单流仅 33）。两线同镜像同权重，可同机不同卡组并存。整案与证据链：`docs/2026-08-29-TP4攻克-病理定罪与外科配方.md`。

## 循环画像实测（2026-08-30，用户真实负载：23K冷启→+1-2K/轮→350K→压缩回40K循环，出300/峰值3K）

前提修正：`--pack-paged-kv-to-varlen-min-q-tokens 4096`（默认 2048 在尾巴 >2048 token 时
触发 pack 整池聚拢，TTFT 9s；提到 4096 后尾部轮全走快速分页路径——本目录 serve.sh 已内置）。

| 会话深度 | 尾部轮 TTFT | decode tok/s | 每轮 E2E（出300）|
|---|---|---|---|
| 40K（压缩重建后）| **0.39s** | 35.8 | 7.0s |
| 100K | 0.67s | 66.8 | ~4-6s |
| 160K | 1.05s | 90.4 | 4.4s |
| 220K | 1.68s | 81.9 | **5.3s** |
| 280K | 1.69s | —（样本输出过短）| ~2-6s |
| 350K（周期顶）| **2.31s** | 34.7 | 10.9s |

一次性成本：冷启 23K prefill 11.0s；压缩重建 40K ≈ 10-25s；
批量增长段速率 541（→160K）→ 212 tok/s（→350K，超出审计桶后衰减）。
峰值 3K 输出轮（ignore_eos 强制满额）：E2E 105s（decode 31.8）。
**全周期 TTFT 恒在 0.4-2.3s；350K 无悬崖（decode 34.7）。**

**口径修正（2026-08-30，用户确认常态输出=500 token、峰值=3K token）**：
常态轮 E2E = TTFT + 500/decode。decode 在该类内容上随接受率波动 28-90，
实测抽查 @220K 满额 500 轮：TTFT 1.61s / decode 28.3 / **E2E 19.2s**（波动区间 8-19s，
接受率高的轮次 ~8s）。峰值 3K 轮 105s 口径本就按 token，维持不变。
