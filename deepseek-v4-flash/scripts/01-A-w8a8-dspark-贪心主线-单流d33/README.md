# DSv4-01 · 0731-w8a8 + DSpark · TP8（贪心主线）

8 卡 K100-AI 上 DeepSeek-V4-Flash（284B MoE，激活 13B）的单流最快形态。
2026-08-14 定稿的生产线固化包。

## 实测

| 指标 | 值 |
|---|---|
| 单流 decode（编程类） | **33.2 tok/s**（DSpark accept 4.38 / 0.68） |
| 23K prompt prefill | 437 tok/s |
| 并发 | 8/8、10/10 稳定（贪心） |
| KV 池 | 1,026,816 token |

## 硬边界（必读）

1. **只在 temperature=0 稳定**。temp>0 且并发≥8 → GPU 硬件异常 0x1016，服务挂死需重启。
   缓解：网关对 temp>0 强制 `top_k=1`（数学等价贪心且实测稳定），或路由到 02 线。
2. **勿改 `--kv-cache-dtype`**：本线按 fp8 布局读 KV；bf16 会静默乱码且 accept 恒 1.00。
3. Think 需请求显式携带 `{"chat_template_kwargs": {"thinking": true}}`。
4. 就绪约 14 分钟；就绪后**先热身两次**再测（首次生成受 JIT 污染必偏慢）。

## 镜像与权重

- 镜像：`custom:sglang0.5.12-…-20260804-0006-deepseekV4-0811`（harbor，S2 自动拉取）
- 权重：`$DSV4_MODELS_ROOT/dsv4-0731-w8a8-dspark`（默认 /data1/models）——**自量化产物**，
  无法自动下载；配方在包内 `quant/`（bf16 原模型 → W8A8 + DSpark 张量）
- 四个运行时补丁（inner/）：triton 后端路由、dflash renorm、DSpark torch accept、
  **moe_align 根修**（gfx928 上预编译 sgl_kernel 输出垃圾 → VMFault 的根因修复）
