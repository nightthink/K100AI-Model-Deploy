# A 线 sglang · TP4 · 1M · **基线**

> 本文件是配置目录化时期的**详细依据**（参数理由/实测/踩坑），保持原样。
> 简明说明见同目录 README.md。

未调优的基准配置，`a-tp4` 那轮 SLA 扫描用的就是它。
64k/80k 各 5 格数据见 `docs/sla-2026-08-25/a-tp4.md`。

## 元数据（`launch.sh` 据此做 S3 校验与 S8 自证）

```
（尚未补 @meta，launch.sh 无法校验）
```

## 用法

```bash
bash common/launch.sh A-sglang-tp4-1m
TEARDOWN=1 bash common/launch.sh A-sglang-tp4-1m     # 冒烟
```

> 本目录自给自足：除 `lib/`（阶段契约）、`common/`（跨配置工具）、
> `patches/`（补丁）外，本配置用到的一切都在这里。
> 规范见 [`docs/方案脚本规范-设计文档.md`](../../../docs/方案脚本规范-设计文档.md)。
