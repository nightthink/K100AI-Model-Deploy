# 运行时补丁（挂载/注入进容器，不改镜像）

| 文件 | 作用 | 用法 |
|---|---|---|
| `drco_idempotent.py` | lightop 与 lmslim 重复注册算子 `lmslim::gptq_gemm1` 的对策：注册幂等化（护栏 1·十一 坎②）| `-v` 同时挂到两处 `direct_register_custom_op.py` |
| `fix_actorder_int8_config.py` | INT8 模型 config.json 里无效 `actorder` 字段的清除（坎①）| 对模型目录跑一次，自动留 `.orig_hf` 备份 |

DFLASH 探索件（q8split 拆分 + 注入包装器）在仓库 `lab/runtime-extras/`，与现场拉起无关，不随包。
