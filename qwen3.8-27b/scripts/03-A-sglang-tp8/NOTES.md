# A 线 sglang · TP8 · 1M · **混合传输**

> 本文件是配置目录化时期的**详细依据**（参数理由/实测/踩坑），保持原样。
> 简明说明见同目录 README.md。

八卡全用。同 socket 走 VRAM P2P、跨 socket 走 SHM，靠 `dlhook2-sg.so` 按 socket 过滤
agent 列表绕开 hycu 的 52ms evict↔restore 活锁。KV 池 2,125,568。

**自证必查** `→保留4个`：`LD_PRELOAD` 失败是静默的，不查就会把活锁中毒路径
当成「我们能达到的最好水平」。

**实测**：单流比 TP4 快 1.82×（TPOT 41.78→17.86ms），但吞吐峰值反低 31%。
见 `docs/sla-2026-08-25/a-tp8.md`。

## 元数据（`launch.sh` 据此做 S3 校验与 S8 自证）

```
# @name  q38-sg8h
# @port  8100
# @gpus  0,1,2,3,4,5,6,7
# @expect-gpu 8
# @requires /data/models/Qwen3.8-27B-1M /data/q38-work/dlhook2-sg.so
# @attest agent过滤（活锁绕法）:→保留4个
# @attest 投机解码NEXTN:speculative
```

## 用法

```bash
bash common/launch.sh A-sglang-tp8-hybrid
TEARDOWN=1 bash common/launch.sh A-sglang-tp8-hybrid     # 冒烟
```

> 本目录自给自足：除 `lib/`（阶段契约）、`common/`（跨配置工具）、
> `patches/`（补丁）外，本配置用到的一切都在这里。
> 规范见 [`docs/方案脚本规范-设计文档.md`](../../../docs/方案脚本规范-设计文档.md)。
