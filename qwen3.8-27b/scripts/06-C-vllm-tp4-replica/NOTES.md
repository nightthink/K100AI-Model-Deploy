# C 线 vLLM · TP4 · 参数化副本（DP2×TP4 的构件）

> 本文件是配置目录化时期的**详细依据**（参数理由/实测/踩坑），保持原样。
> 简明说明见同目录 README.md。

`GPUS`/`PORT`/`NAME`/`MAXLEN`/`YARN_FACTOR`/`MAMBA_ALL` 均可传。
配 `common/serve_router.sh` 组成 DP2×TP4 单一入口。

⚠ **TP4 装不下 1M**：单请求需 61.06 GiB KV/卡而仅 41.01 可用，vLLM 算出上限 671,744 token。

## 元数据（`launch.sh` 据此做 S3 校验与 S8 自证）

```
（尚未补 @meta，launch.sh 无法校验）
```

## 用法

```bash
bash common/launch.sh C-vllm-tp4-replica
TEARDOWN=1 bash common/launch.sh C-vllm-tp4-replica     # 冒烟
```

> 本目录自给自足：除 `lib/`（阶段契约）、`common/`（跨配置工具）、
> `patches/`（补丁）外，本配置用到的一切都在这里。
> 规范见 [`docs/方案脚本规范-设计文档.md`](../../../docs/方案脚本规范-设计文档.md)。
