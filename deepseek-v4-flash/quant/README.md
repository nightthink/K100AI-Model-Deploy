# DSv4-Flash 0731 W8A8 量化配方（自给）

本包两条线的权重均为自量化产物，从公开 bf16 原模型生成：

1. 取 DeepSeek-V4-Flash bf16 原模型（HF/ModelScope 公开，~149GB）
2. `python3 quant_w8a8_0731.py`（逐通道 W8A8，含 DSpark 张量保留版与剔除版双输出）
3. `bash run_0731_pipeline.sh` 一键串联（量化 + 校验）

产物目录：`dsv4-0731-w8a8-dspark`（01 线用）与 `dsv4-0731-w8a8`（02 线用）。
注意：ModelScope 上 `hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8`（2026-05-15）是更老的
量化谱系，与本包补丁未联测——请用本配方产物。
