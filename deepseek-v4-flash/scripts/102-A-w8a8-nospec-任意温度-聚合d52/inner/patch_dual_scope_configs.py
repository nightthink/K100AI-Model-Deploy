"""给 dual_scope 的 autotune 候选表补两个旋钮：软件流水与更大的 N 分块。

## 为什么（docs/攻坚路线图-prefill4000-decode80-2026-09-13.md 候选⑨）

dual_scope 占 prefill 单卡忙时 44.8%，而它**既非算力瓶颈也非带宽瓶颈**：

    算力 0.602 TFLOP/s ÷ bf16 峰值 85.6 TFLOPS = 0.70%
    带宽 22.5 GB/s     ÷ HBM 约 1 TB/s        = 约 2%

两头都只有百分之一二 ⇒ 瓶颈是**访存延迟与块内占用率**。

原本对症的解法是分离式 prefill（flash_mla_sparse_fwd），但它自称
"DSA MLS sparse prefill"，而 gfx928 的默认特性集里**一个 MLS（矩阵加载）特性都没有**
（gfx936 有 bowen-mls，gfx938 有 bowen-mls + hcu-mls，本机两者皆无），
极可能硬件级封死。本补丁是不依赖 MLS 的替代打法。

## 现有候选表的两处空白

十个配置经 `_prune_dual_scope_configs`（h_q=8 ≤ 64 ⇒ 只留 BLOCK_H ≤ 16）后，
**实际可选的只剩三个**：(16,64,w4)、(16,128,w4)、(16,64,w8)。而：

  · num_stages 全表恒为 1 —— 软件流水完全没启用。对延迟受限的 gather 循环，
    num_stages ≥ 2 让下一批 KV 的取数与当前批的计算重叠，这是第一解法。
  · BLOCK_N 上限 128 —— 640 条 topk 要 5 轮；256 只要 2.5 轮，访存并行度翻倍。

剪枝函数**只过滤 BLOCK_H**，不看 BLOCK_N 与 num_stages，故新增的
BLOCK_H=16 配置一律存活。

## 为什么这比改内核逻辑先做

autotune 会自行择优：**新配置不占优就不会被选中，等于零损失**。
风险在另一头——num_stages=2 与 BLOCK_N=256 都会抬高 LDS 与寄存器占用，
可能反而压低占用率。**所以必须实测，不能推断。**

与已判死的候选⑧（BLOCK_H=8）的区别：⑧死于硬件——gfx928 的矩阵指令族是 MMAC，
`mmop2-insts` 全族清一色 16x16x*，M 维恒为 16，往下没有指令。
本补丁动的是 **N 维与流水深度**，与指令粒度无关，不受 MMAC 约束。

## 开关

未设 `DSV4_DUAL_SCOPE_EXTRA_CFG` 或设为 0 时，附加表为空，
**候选表与原版逐字相同，行为不变**。设为 1 时追加四条，并在导入时打一行
`[DUAL_SCOPE_CFG]` 探针——只有这行出现才算真的加上了。
"""

MARKER = "PATCH_DUAL_SCOPE_CFG"
F = ("/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/"
     "nsa/triton_decode/triton_mla_kernels_decode_fused.py")

OLD_DEF = "def _prune_dual_scope_configs(configs, named_args, **kwargs):\n"
NEW_DEF = '''_PATCH_DUAL_SCOPE_CFG = __import__("os").environ.get(
    "DSV4_DUAL_SCOPE_EXTRA_CFG", "0"
) not in ("0", "", "false", "False")  # PATCH_DUAL_SCOPE_CFG
if _PATCH_DUAL_SCOPE_CFG:
    # 延迟受限的 gather 循环该动的两个旋钮：软件流水（num_stages）与 N 分块。
    _PATCH_EXTRA_CFGS = [
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]
    import logging as _pl
    _pl.getLogger(__name__).info(
        "[DUAL_SCOPE_CFG] PATCH_DUAL_SCOPE_CFG active: +%d configs (BLOCK_N=256 / num_stages=2)",
        len(_PATCH_EXTRA_CFGS),
    )
else:
    _PATCH_EXTRA_CFGS = []


def _prune_dual_scope_configs(configs, named_args, **kwargs):
'''

OLD_LIST = """        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "topk_main", "topk_extra"],
"""
NEW_LIST = """        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
    ] + _PATCH_EXTRA_CFGS,  # PATCH_DUAL_SCOPE_CFG
    key=["total_tokens_bucket", "h_q", "topk_main", "topk_extra"],
"""

HUNKS = [("开关与附加表", OLD_DEF, NEW_DEF), ("候选表收尾", OLD_LIST, NEW_LIST)]

if __name__ == "__main__":
    s = open(F, encoding="utf-8").read()
    if MARKER in s:
        print(f"[{MARKER}] 已打过，跳过"); raise SystemExit(0)
    for name, old, new in HUNKS:
        n = s.count(old)
        assert n == 1, f"[{MARKER}] 锚点「{name}」匹配 {n} 次（应为 1），拒绝改动"
        s = s.replace(old, new, 1)
    open(F, "w", encoding="utf-8").write(s)
    print(f"[{MARKER}] ✓ 已打 {len(HUNKS)} 处（未设 DSV4_DUAL_SCOPE_EXTRA_CFG=1 时候选表与原版逐字相同）")
