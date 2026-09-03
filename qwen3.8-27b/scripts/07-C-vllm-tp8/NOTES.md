# C 线 vLLM · TP8 · 混合传输

> 本文件是配置目录化时期的**详细依据**（参数理由/实测/踩坑），保持原样。
> 简明说明见同目录 README.md。

八卡全用，`dlhook2.so` agent 过滤。`--disable-custom-all-reduce` 必须保留
（去掉会卡死在 CUDA graph 捕获）。

**实测**：短 prompt 单流 54.82 tok/s、16 并发 288.87（与历史一致，健康）；
但 **64k 输入单流只有 6.65 tok/s，比 A 线慢 8.4×**。根因是 aiter 的 GDN 内核
缺 gfx928 配置（回落 `num_warps=1`）。见 `docs/sla-2026-08-25/c-tp8.md`。

## 元数据（`launch.sh` 据此做 S3 校验与 S8 自证）

```
# @name  q38-tp8h
# @port  8108
# @gpus  0,1,2,3,4,5,6,7
# @expect-gpu 8
# @requires /data/models/Qwen3.8-27B /data/q38-work/dlhook2.so
# @attest agent过滤（活锁绕法）:→保留4个
# @attest 投机解码MTP:speculative_config
```

## 用法

```bash
bash common/launch.sh C-vllm-tp8-hybrid
TEARDOWN=1 bash common/launch.sh C-vllm-tp8-hybrid     # 冒烟
```

> 本目录自给自足：除 `lib/`（阶段契约）、`common/`（跨配置工具）、
> `patches/`（补丁）外，本配置用到的一切都在这里。
> 规范见 [`docs/方案脚本规范-设计文档.md`](../../../docs/方案脚本规范-设计文档.md)。
