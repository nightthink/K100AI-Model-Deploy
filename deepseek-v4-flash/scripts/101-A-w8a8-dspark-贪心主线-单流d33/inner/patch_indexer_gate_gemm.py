"""DSA 索引器头门 GEMM：绕开 rocBLAS 在 gfx928 上没有调优内核的「bf16 入 fp32 出」组合。

## 现象（2026-09-13，profiler 实测，见 docs/剖析-101线decode一步114ms-2026-09-13.md）

单流 decode 一步 114 ms，GPU 忙 97%，其中 **40%（45 ms）是同一个 GEMM**：

    Cijk_Alik_Bljk_BSS_BH_MT128x32x16_…_GSU3_ISA000     105 次/步，428 µs/次

`ISA000` = rocBLAS 通用回退内核，不是 gfx928 调优内核。1 卡微基准（tests/prof/blas_probe.py）
把它精确对到 `torch.mm(bf16, bf16, out_dtype=torch.float32)`：rocBLAS 对 gfx928 有
bf16→bf16（BBH）、fp32→fp32（SB）、int8（I8II）的 ISA928 调优内核，**唯独没有 bf16→fp32**，
于是落到 MT128x32x16 + K 分 3 片的通用内核，M=7 时慢 3–9 倍。

## 出处

`layers/attention/dsa/dsa_indexer.py` 的 `logits_head_gate_graph`（图内非 prefill 路径，
decode 每层每趟调用一次）：

    out = torch.mm(x, weight.t(), out_dtype=torch.float32)      # x: [M, 4096] bf16, weight: [64, 4096] bf16

N 只有 64（index_n_heads），却每次 428 µs。

## 改法

    out = torch.mm(x, weight.t()).float()                        # BBH ISA928 内核 + 微小升精度拷贝

精度依据：sglang 自己在 HIP 的 eager 路径（同文件 `_weights_proj_bf16_in_fp32_out`）就是
`weights, _ = self.weights_proj(x)` 返回 bf16，注释写明 "Return bf16; multiplying with
q_scale promotes back to fp32"。本改法让图路径与 eager 路径精度一致，而不是引入新的近似。

预期：428 → ~40 µs/次，每步省 ~40 ms（114 → ~75 ms，单流 40 → ~55 tok/s）。
须由 correctness 四项 + needle + dsv4_measure 验证（bf16 头门权重是否改变 top-k 选择）。
"""
MARKER = "PATCH_INDEXER_GATE_GEMM"
F = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/dsa/dsa_indexer.py"

OLD = """        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
        weights = out * n_heads_inv_sqrt
"""
NEW = """        # PATCH_INDEXER_GATE_GEMM: gfx928 的 rocBLAS 没有 bf16→fp32 调优内核（落到 ISA000 通用核，
        # 428 µs/次、占 decode 一步 40%）。改走 bf16→bf16 的 ISA928 内核再升精度，
        # 与本文件 HIP eager 路径 _weights_proj_bf16_in_fp32_out 的精度一致。
        out = torch.mm(x, weight.t()).float()
        weights = out * n_heads_inv_sqrt
"""

if __name__ == "__main__":
    s = open(F, encoding="utf-8").read()
    if MARKER in s:
        print(f"[{MARKER}] 已打过，跳过")
    else:
        assert s.count(OLD) == 1, f"[{MARKER}] 锚点命中 {s.count(OLD)} 处（应为 1）——镜像版本可能已变"
        open(F, "w", encoding="utf-8").write(s.replace(OLD, NEW, 1))
        print(f"[{MARKER}] ✓ logits_head_gate_graph 改走 bf16 GEMM + .float()")
    assert MARKER in open(F, encoding="utf-8").read()
