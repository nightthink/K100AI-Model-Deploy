# A 线 sglang · TP4 · 1M · **面向单请求 prefill+decode 调优**

> 本文件是配置目录化时期的**详细依据**（参数理由/实测/踩坑），保持原样。
> 简明说明见同目录 README.md。

卡 0–3（全机唯一同质快卡组）。相对 `A-sglang-tp4-1m` 的改动：
`--chunked-prefill-size 32768`、`--max-prefill-tokens 45000`（海光官方配方）、
`SGLANG_USE_CUDA_IPC_TRANSPORT=1`（★仅同 socket 安全）、`SGLANG_USE_AITER_LINEAR_ATTN=1`。

**实测（输出 6144 token，对照 A-sglang-tp4-1m）**：80k 上 prefill **1,270→2,548 tok/s（2.01×）**、
TPOT **63.40→48.2 ms（1.31×）**；64k 上两者基本持平。**赢在长输入不塌**。

⚠ 去掉 `--disable-custom-all-reduce` 会让 rank0 SIGSEGV（aiter 的 custom AR 在 K100-AI 上崩）。

## 元数据（`launch.sh` 据此做 S3 校验与 S8 自证）

```
# @name  q38-sgA
# @port  8101
# @gpus  0,1,2,3
# @expect-gpu 8
# @requires /data/models/Qwen3.8-27B-1M   （v2 起不再需要 dlhook2-sg；以下为 v1/0811 树时期的历史笔记）
# @attest 上下文长度:context_length
# @attest 投机解码NEXTN:speculative
```

## 组成 DP2×TP4（八卡推荐方案）

```bash
bash common/launch.sh A-sglang-tp4-tuned GPUS=0,1,2,3 PORT=8101 NAME=q38-sgA
bash common/launch.sh A-sglang-tp4-tuned GPUS=4,5,6,7 PORT=8102 NAME=q38-sgB
bash common/serve_router.sh          # 单一入口 8100
```

**实测（2026-08-27，numa_balancing=0，64k 输入 / 6144 输出）**：并发 8 时
prefill 聚合 **6,298 tok/s**、decode 聚合 **127.35 tok/s**，分别是 TP8 混合的
**2.91×** 与 **2.55×**；并发 12 仍在涨（7,309 / 153.78）。
详见 [`docs/sla-2026-08-27/DP2xTP4-vs-TP8.md`](../../../docs/sla-2026-08-27/DP2xTP4-vs-TP8.md)。

⚠ **router 必须单 worker** —— 会话表与在飞计数是进程内状态，加 `--workers` 粘性立即失效。
会话粘性已实测验证（`tests/sticky_test.py`）：带 `X-Session-Id` 6/6 同副本、
不带任何标识仅靠 prompt 前缀 6/6 同副本、8 个不同会话 4:4 均衡。

## 用法

```bash
bash common/launch.sh A-sglang-tp4-tuned
TEARDOWN=1 bash common/launch.sh A-sglang-tp4-tuned     # 冒烟
```

> 本目录自给自足：除 `lib/`（阶段契约）、`common/`（跨配置工具）、
> `patches/`（补丁）外，本配置用到的一切都在这里。
> 规范见 [`docs/方案脚本规范-设计文档.md`](../../../docs/方案脚本规范-设计文档.md)。
