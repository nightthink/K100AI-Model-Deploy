# GDN 比值 3 融合 —— 13 线 A/B 装置（原样收录）

产出 2026-09-16 那次 A/B 结论的就是这套文件，此前只存在于机器
`~/patchwork/`，不在版本控制内。**全部原样收录，未作改动**，sha256 与机上一致。

## 为什么 0.5.12 上需要**两个**补丁

| | 0.5.12（13 线基版） | 0.5.18 |
|---|---|---|
| 比值闸 | `qwen3_5.py:473` 硬编码字面量 `in [1, 2, 4]`，无具名常量、无 CUDA/HIP 分支 | 具名常量 `_GDN_FUSED_QKVZBA_RATIOS`，按后端分档 |
| 内核非 2 的幂支持 | `jit_kernel/triton/gdn_fused_proj.py`（310 行），**无 `V_POW2`** | `kernels/ops/attention/triton_gdn_fused_proj.py`（429 行），`V_POW2` 4 处 |

⇒ 0.5.12 上放开比值 3 必须**同时**改模型文件与内核，故本目录有两件：

- `qwen3_5_armA_baseline.py` / `qwen3_5_armB_widened.py` —— A/B 两臂，唯一差异是白名单
- `gdn_fused_proj_patched.py` —— 自写内核，补上按 `V_PER_GROUP` 逐个搬运的非 2 的幂路径
- `serve_gdnfuse.sh` / `armA2_chain.sh` —— 起服与链脚本
- `fixed_armA.txt` / `fixed_armB.txt` —— 两臂的固定输出，用于逐字节比对

## 结论（已在 docs/GDN融合-比值3-预登记-2026-09-16.md 落档）

张量级逐位相同（max abs diff 0.000e+00）、端到端输出 sha256 相同；
增益约 **+3%（8K）/ +2%（32K）**，重起漂移 ±1%。
