"""hc 的两次加权求和改用 bmm：消掉 prefill 里最大的一块中间张量。

## 现象（2026-09-13 prefill 剖析，见 docs/剖析-101线prefill-2026-09-13.md §六·六 / §六·七）

66K prefill、TP8，单卡 kernel 忙时 117.2 s。按内核名分家族时，`other` 桶记了 28.09 s（24%），
当时被当成杂项略过。逐项打印后发现 **其中 93%（26.1 s）是 hc 与 RMSNorm**：

      16758 ms  2964 次  reduce_kernel<128,4,ReduceOp<float,func_wrapper_t<float,…>>>
       3368 ms  2924 次  elementwise_kernel_manual_unroll<128,4,type_…>
       1590 ms  1462 次  vectorized_elementwise_kernel<4,CUDAFunctor_add<float>>
       1043 ms  3307 次  bfloat16tofloat32_copy_kernel_cuda
       1029 ms  2964 次  bfloat16_copy_kernel_cuda
        925 ms  1463 次  pow_tensor_…
        491 ms  1463 次  reduce_kernel<512,1,ReduceOp<float,MeanOps<float>>>
         20 ms  1463 次  rsqrt_kernel_cuda

调用次数即指纹：18 个 run_batch 步 × 43 层 = 774，hc_pre 与 hc_post 各一次 ⇒ 1548；
实测 reduce 与 bf16 拷贝各 2964（≈2×1482），pow/mean/rsqrt/add 各 1463（≈每次 hc_pre 一遍）。
与 `models/deepseek_v4.py` 的源码逐项吻合，无一落空。

## 病灶

    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)

先把 pre（[T, hc_mult]）广播乘出 **[T, hc_mult, hidden] 的 fp32 中间张量**、落回显存，
再读回来沿 hc_mult 维归约。chunk 4096 / hidden 4096 / hc_mult 4 时，
**单次单层就要写入并读回 268 MB**。hc_post 更糟：中间张量是 [T, hc_mult, hc_mult, hidden]，
是输出的 hc_mult 倍。

## 改法：严格等价的批量矩阵乘

hc 混合与 hc_pre：  y[t,h]    = Σ_m pre[t,m] · x[t,m,h]        ⇒ bmm([T,1,M], [T,M,H])
hc_post：          out[t,j,h] = Σ_i comb[t,i,j] · residual[t,i,h] ⇒ bmm(comb^T, residual)

两处都不是近似——同样的乘加集合，只是交给 GEMM 去做，不再 materialize 中间张量。
（浮点累加次序可能变化。这与已判死的 tilelang MHC 所担心的「MFMA reduction orders」不同：
bmm 走标准 GEMM 累加路径，不是手写 MFMA 归约。）

为什么不改走「开 prefill cuda graph 让 inductor 融合」：`compile_in_capture_mode` 只在
捕获图时才套 torch.compile，prefill 无图故全部散开；但开图要动 PhaseConfig、
且 `cuda_graph_tc_compiler` 默认 eager，链路长、起服风险高。bmm 改写是纯 PyTorch
等价变换，可 env 门控、可数值比对，与已实测有效的 decode bf16→fp32 补丁是同一手法。

## 开关

未设 `DSV4_HC_BMM` 或设为 0 时**完全不改变原行为**（三处都保留原式在 else 分支）。
设 `DSV4_HC_BMM=1` 时生效，并在首次 hc_pre 调用时打一行 `[HC_BMM]` 运行时探针——
只有这行出现才算真的走上了新路（本项目已三次栽在「看起来该生效、实际没执行」上）。
"""

MARKER = "PATCH_HC_BMM"
F = "/usr/local/lib/python3.10/dist-packages/sglang/srt/models/deepseek_v4.py"

# ── ① 模块级开关（挂在同族 env 旁边，get_bool_env_var 该模块已导入）────────────
OLD_FLAG = '_use_dpskv4_lightop_rmsnorm = get_bool_env_var("SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM")\n'
NEW_FLAG = (
    '_use_dpskv4_lightop_rmsnorm = get_bool_env_var("SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM")\n'
    '_PATCH_HC_BMM = get_bool_env_var("DSV4_HC_BMM")  # PATCH_HC_BMM\n'
    "_PATCH_HC_BMM_SEEN = False  # PATCH_HC_BMM\n"
)

# ── ② hc_head_torch 的加权求和 ────────────────────────────────────────────────
OLD_HEAD = """    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype)
"""
NEW_HEAD = """    if _PATCH_HC_BMM:
        # PATCH_HC_BMM: y[t,h] = Σ_m pre[t,m]·x[t,m,h]，等价于 bmm，免 [T,M,H] 中间张量
        _x3 = x.view(shape)
        _m, _h = _x3.shape[-2], _x3.shape[-1]
        y = torch.bmm(
            pre.reshape(-1, 1, _m).to(_x3.dtype), _x3.reshape(-1, _m, _h)
        ).reshape(*shape[:-2], _h)
    else:
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype)
"""

# ── ③ hc_pre 的加权求和（HIP 上 DEEPGEMM_HC_PRENORM 被强制关，这里是活跃现场）──
OLD_PRE = """        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False
"""
NEW_PRE = """        if _PATCH_HC_BMM:
            # PATCH_HC_BMM: 同 hc_head_torch，[T,1,M] @ [T,M,H] → [T,1,H]
            global _PATCH_HC_BMM_SEEN
            _x3 = x_flat.view(shape)
            y = torch.bmm(
                pre.squeeze(1).unsqueeze(1).to(_x3.dtype), _x3
            ).squeeze(1)
            if not _PATCH_HC_BMM_SEEN:
                _PATCH_HC_BMM_SEEN = True
                logger.info(
                    "[HC_BMM] PATCH_HC_BMM active: pre=%s x=%s y=%s",
                    tuple(pre.shape), tuple(_x3.shape), tuple(y.shape),
                )
        else:
            y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False
"""

# ── ④ hc_post（中间张量是输出的 hc_mult 倍，收益最大的一处）───────────────────
OLD_POST = """        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)
"""
NEW_POST = """        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            if _PATCH_HC_BMM:
                # PATCH_HC_BMM: out[t,j,h] = Σ_i comb[t,i,j]·residual[t,i,h] ⇒ bmm(comb^T, residual)
                # 原式的中间张量是 [T,M,M,H]，即输出的 M 倍。
                return (
                    post.unsqueeze(-1) * x.unsqueeze(1)
                    + torch.bmm(
                        comb.transpose(1, 2).contiguous().to(residual.dtype), residual
                    )
                ).type_as(x)
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)
"""

HUNKS = [
    ("模块级开关", OLD_FLAG, NEW_FLAG),
    ("hc_head_torch 求和", OLD_HEAD, NEW_HEAD),
    ("hc_pre 求和", OLD_PRE, NEW_PRE),
    ("hc_post 求和", OLD_POST, NEW_POST),
]

if __name__ == "__main__":
    s = open(F, encoding="utf-8").read()
    if MARKER in s:
        print(f"[{MARKER}] 已打过，跳过")
        raise SystemExit(0)
    for name, old, new in HUNKS:
        n = s.count(old)
        assert n == 1, f"[{MARKER}] 锚点「{name}」匹配 {n} 次（应为 1），拒绝改动"
        s = s.replace(old, new, 1)
    open(F, "w", encoding="utf-8").write(s)
    print(f"[{MARKER}] ✓ 已打 {len(HUNKS)} 处（未设 DSV4_HC_BMM=1 时行为不变）")
