"""linear_bf16_fp32：绕开 rocBLAS 在 gfx928 上没有调优内核的「bf16 入 fp32 出」GEMM。

## 现象（2026-09-13 profiler + eager 归因，见 docs/剖析-101线decode一步114ms-2026-09-13.md）

单流 decode 一步 114 ms，**40%（45 ms）是同一个 rocBLAS 通用回退内核**
`Cijk_Alik_Bljk_BSS_BH_…_GSU3_ISA000`（105 次/步，428 µs/次）。eager 归因把它钉在
`sglang/jit_kernel/dsv4/gemm.py: linear_bf16_fp32` 的最后一个分支：

    return torch.mm(x, y.t(), out_dtype=torch.float32)

调用者：
  · `layers/attention/dsv4/compressor.py: compute_kv_score` —— `wkv_gate`，N = 256 / 1024 / 2048
  · `models/deepseek_v2.py: MoEGate.forward`（HCU + DSv4 分支，"HCU Hash-MoE JIT requires FP32 router logits"）—— N = 256

1 卡微基准（tests/prof/blas_probe.py）：rocBLAS 对 gfx928 有 bf16→bf16（BBH）、fp32→fp32（SB）
的 ISA928 调优内核，唯独 bf16→fp32 落到通用核，M=7 时慢 3–9 倍。hipBLASLt 全局偏好更慢。

## 两种改法，由 DSV4_BF16FP32_ALGO 选

  bf16      : torch.mm(x, y.t()).float()          —— BBH ISA928，最快；输出经 bf16 舍入再升精度
  fp32cache : 首次见到某权重时缓存一份 fp32 副本；之后 torch.mm(x.float(), w32.t()) —— SB ISA928，
              结果与 out_dtype=fp32 等价（fp32 累加、fp32 输出）；代价是权重多占一份 fp32 显存
              （ReplicatedLinear 不分片：路由门 256×4096×4B×43 层 ≈ 172 MB，压缩器 1024/2048 宽的更大）

默认 fp32cache（路由 logits 与压缩分数都要做 top-k，精确优先）；速度差距见 docs 里的微基准表。
缓存条目在 warmup 期间建立（sglang 图捕获前必有 warmup），不会把分配捕进 cuda graph。

未设 DSV4_BF16FP32_ALGO 或设为 off 时不改变原行为。
"""
MARKER = "PATCH_BF16FP32_GEMM"
F = "/usr/local/lib/python3.10/dist-packages/sglang/jit_kernel/dsv4/gemm.py"

OLD_HEAD = "import torch\n"
NEW_HEAD = "import torch\nimport os  # PATCH_BF16FP32_GEMM\n"

OLD = """    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
"""
NEW = """    elif _PATCH_ALGO == "bf16":
        # PATCH_BF16FP32_GEMM: gfx928 的 rocBLAS 无 bf16→fp32 调优核（ISA000 通用核 428 µs/次），
        # 改走 bf16→bf16 的 ISA928 内核再升精度（输出经 bf16 舍入）。
        return torch.mm(x, y.t()).float()
    elif _PATCH_ALGO == "fp32cache":
        # PATCH_BF16FP32_GEMM: 缓存 fp32 权重，走 fp32→fp32 的 ISA928 内核，结果与 out_dtype=fp32 等价。
        key = (y.data_ptr(), tuple(y.shape))
        w32 = _PATCH_W32.get(key)
        if w32 is None:
            w32 = y.detach().float().contiguous()
            _PATCH_W32[key] = w32
        return torch.mm(x.float(), w32.t())
    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)


_PATCH_ALGO = os.environ.get("DSV4_BF16FP32_ALGO", "off").lower()  # PATCH_BF16FP32_GEMM
_PATCH_W32 = {}
"""

if __name__ == "__main__":
    s = open(F, encoding="utf-8").read()
    if MARKER in s:
        print(f"[{MARKER}] 已打过，跳过")
    else:
        assert s.count(OLD_HEAD) >= 1, f"[{MARKER}] 文件头锚点未命中"
        assert s.count(OLD) == 1, f"[{MARKER}] 函数尾锚点命中 {s.count(OLD)} 处（应为 1）——镜像版本可能已变"
        s = s.replace(OLD_HEAD, NEW_HEAD, 1).replace(OLD, NEW, 1)
        open(F, "w", encoding="utf-8").write(s)
        print(f"[{MARKER}] ✓ linear_bf16_fp32 新增 bf16 / fp32cache 分支（DSV4_BF16FP32_ALGO 选，默认 off=原行为）")
    import py_compile
    py_compile.compile(F, doraise=True)
    print(f"[{MARKER}] 当前 DSV4_BF16FP32_ALGO={__import__('os').environ.get('DSV4_BF16FP32_ALGO', 'off')}")
