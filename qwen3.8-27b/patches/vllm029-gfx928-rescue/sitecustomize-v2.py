"""vllm0.29 flashnext 镜像 gfx928 救援补丁 v2（2026-09-11）

背景（2026-09-02 已用哨兵实验坐实，2026-09-11 复验并扩大范围）：
镜像的 vLLM 编译扩展只含 gfx936（打包人 PYTORCH_ROCM_ARCH=gfx936）：
    _C.abi3.so                     : gfx936
    _moe_C_stable_libtorch.abi3.so : gfx936
    _C_stable_libtorch.abi3.so     : gfx936 gfx950
在 gfx928（K100-AI）上这些 kernel **不报错、不写出**（静默空跑），
输出 buffer 保持 torch.empty 的未初始化内容 → 垃圾索引 → 越界读显存 → VMFault。

2026-09-11 哨兵实测（单卡，产线挂载），确认静默空跑的算子共 11 个：

  rms_norm / fused_add_rms_norm / rotary_embedding              → CustomOp 家族
  silu_and_mul / gelu_and_mul / gelu_tanh_and_mul / mul_and_silu → CustomOp 家族
  topk_softmax / moe_align_block_size / moe_sum                 → 本补丁 v1 已替换
  top_k_per_row_decode                                          → ★ v2 新增

分工：
  · CustomOp 家族（前 7 个）由启动参数 --compilation-config '{"custom_ops":["none"]}'
    走 forward_native（已核实是纯 torch：F.silu(x[..,:d])*x[..,d:]、
    x.pow(2).mean()+torch.rsqrt，不会绕回 _C）。
  · MoE 主体走 triton（fused_moe.py 有 4 个 @triton.jit，按实际 arch 即时编译，
    不受 gfx936 限制），但其 align/topk/sum 三个辅助算子在 _C/_moe_C 里 → v1 替换。
  · QSA（Flash-Next 的稀疏注意力）在 ROCm 上走 ops.top_k_per_row_decode → v2 替换。
    注：qsa.py 里 cooperative_topk / persistent_topk 都在 current_platform.is_cuda()
    分支内，ROCm 走不到；且 cooperative_topk 在本镜像中根本不存在。
    top_k_per_row_prefill 在 qsa.py 中无调用点（prefill 走 triton 的 qsa_mqa_paged）。

开关 FIX928=0 可整体停用。
"""
from __future__ import annotations

import os

_on = os.getenv("FIX928", "1").strip().lower() in ("1", "true", "yes", "on")

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
    or "cdll.LoadLibrary" in _cmd
)


def _install() -> None:
    import torch
    import vllm._custom_ops as C

    # ---------------- v1：MoE 三算子 ----------------

    def moe_align_torch(topk_ids, num_experts, block_size, sorted_token_ids,
                        experts_ids, num_tokens_post_pad, expert_map=None):
        numel = topk_ids.numel()
        dev = topk_ids.device
        flat = topk_ids.reshape(-1).to(torch.int64)
        valid = (flat >= 0) & (flat < num_experts)
        e = flat[valid]
        tok = torch.nonzero(valid).squeeze(1).to(torch.int32)
        cnt = torch.bincount(e, minlength=num_experts)
        padded = ((cnt + block_size - 1) // block_size) * block_size
        cum = torch.cumsum(padded, 0)
        starts = cum - padded
        total = int(cum[-1].item()) if num_experts > 0 else 0
        # 哨兵预填充：kernel 以 id==numel 识别 pad 槽（DSv4 轮次35 教训）
        sorted_token_ids.fill_(numel)
        if e.numel():
            order = torch.argsort(e, stable=True)
            e_s = e[order]
            run_start = torch.searchsorted(e_s, torch.arange(num_experts, device=dev))
            pos = torch.arange(e_s.numel(), device=dev) - run_start[e_s]
            dest = (starts[e_s] + pos).to(torch.int64)
            sorted_token_ids[dest] = tok[order]
        nblk = total // block_size
        experts_ids.zero_()
        if nblk:
            blk_off = torch.arange(nblk, device=dev, dtype=torch.int64) * block_size
            experts_ids[:nblk] = torch.searchsorted(cum, blk_off, right=True).to(experts_ids.dtype)
        num_tokens_post_pad.fill_(total)

    def topk_softmax_torch(topk_weights, topk_ids, token_expert_indices,
                           gating_output, renormalize=False,
                           e_score_correction_bias=None, is_padding=None):
        scores = torch.softmax(gating_output.float(), dim=-1)
        k = topk_ids.shape[-1]
        if e_score_correction_bias is not None:
            sel = scores + e_score_correction_bias.float()
            _, ids = torch.topk(sel, k, dim=-1)
            w = scores.gather(-1, ids)
        else:
            w, ids = torch.topk(scores, k, dim=-1)
        if renormalize:
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
        topk_weights.copy_(w.to(topk_weights.dtype))
        topk_ids.copy_(ids.to(topk_ids.dtype))
        if token_expert_indices is not None:
            token_expert_indices.zero_()

    def moe_sum_torch(input, output, topk_ids=None, expert_map=None):
        x = input
        if topk_ids is not None and expert_map is not None:
            mask = expert_map[topk_ids.to(torch.int64)] >= 0
            x = x * mask.unsqueeze(-1).to(x.dtype)
        output.copy_(x.sum(dim=1).to(output.dtype))

    # ---------------- v2 新增：QSA 的 top_k_per_row_decode ----------------
    #
    # 语义取自 BAAI 0.26.1 源码树的权威实现（csrc/libtorch_stable/sampler.cu）：
    #
    #   topKPerRowDecode（:569-614）外层算边界：
    #     batch_idx   = rowIdx / next_n
    #     next_n_idx  = rowIdx % next_n
    #     seqLensIs2D = (seqLens.dim() == 2)
    #     seq_len     = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch_idx]
    #     rowEnd      = seqLensIs2D ? max(0, seq_len)
    #                               : max(0, seq_len - next_n + next_n_idx + 1)
    #     rowStart    = 0
    #     logits     += rowIdx * stride0      （行内按 stride1 步进）
    #     outIndices += rowIdx * topK
    #
    #   topKPerRowJob（:387-404）取值：
    #     rowLen = rowEnd - rowStart
    #     · rowLen <= topK  → outIndices[i] = i（0..rowLen-1），
    #       **注释明写 "Indices are not sorted by their corresponding logit"**，
    #       尾部 outIndices[i] = -1
    #     · rowLen >  topK  → 直方图+基数排序取 logits **最大**的 topK 个索引
    #
    # 调用点（qwen4_exp/amd/ops/qsa.py:800）：
    #     ops.top_k_per_row_decode(logits, 1, visible_blocks, blocks,
    #                              blocks.shape[0], logits.stride(0),
    #                              logits.stride(1), block_topk)
    #   logits 为 float32 二维；visible_blocks 为 1D int32；blocks 为 int32 (rows, topK)

    def top_k_per_row_decode_torch(logits, next_n, seq_lens, raw_topk_indices,
                                   num_rows, stride0, stride1, topk_tokens):
        dev = logits.device
        rows = int(num_rows)
        k = int(topk_tokens)
        if rows <= 0 or k <= 0:
            return

        rid = torch.arange(rows, device=dev, dtype=torch.int64)

        # 复刻 rowEnd 的两种形态
        if seq_lens.dim() == 2:
            row_end = seq_lens.reshape(-1)[:rows].to(torch.int64)
        else:
            batch_idx = rid // int(next_n)
            next_n_idx = rid % int(next_n)
            row_end = (seq_lens.to(torch.int64)[batch_idx]
                       - int(next_n) + next_n_idx + 1)
        row_end = row_end.clamp_min(0)

        # 按 stride 还原每行的有效切片；列数上限取 logits 实际宽度
        ncols = int(logits.shape[1]) if logits.dim() >= 2 else int(logits.numel())
        row_end = row_end.clamp_max(ncols)

        # as_strided 精确复刻 C++ 的 logits += rowIdx*stride0，行内 stride1 步进
        flat = logits.reshape(-1)
        base = rid * int(stride0)
        col = torch.arange(ncols, device=dev, dtype=torch.int64) * int(stride1)
        gathered = flat[(base.unsqueeze(1) + col.unsqueeze(0))].float()  # (rows, ncols)

        # 超出 rowEnd 的列不参与竞争
        col_idx = torch.arange(ncols, device=dev, dtype=torch.int64).unsqueeze(0)
        valid = col_idx < row_end.unsqueeze(1)
        scores = torch.where(valid, gathered,
                             torch.full_like(gathered, float("-inf")))

        kk = min(k, ncols)
        _, idx = torch.topk(scores, kk, dim=-1)          # 取最大的 topK
        out = torch.full((rows, k), -1, device=dev, dtype=torch.int32)
        out[:, :kk] = idx.to(torch.int32)

        # rowLen <= topK 的行：C++ 走捷径，填 0..rowLen-1（不按 logit 排序），其余 -1
        short = row_end <= k
        if bool(short.any()):
            ar = torch.arange(k, device=dev, dtype=torch.int64).unsqueeze(0)
            short_fill = torch.where(ar < row_end.unsqueeze(1),
                                     ar, torch.full_like(ar, -1)).to(torch.int32)
            out = torch.where(short.unsqueeze(1), short_fill, out)
        else:
            # 长行也要把超出有效范围的位置置 -1（kk < k 时）
            if kk < k:
                out[:, kk:] = -1

        raw_topk_indices.copy_(out.reshape(raw_topk_indices.shape))

    C.moe_align_block_size = moe_align_torch
    C.topk_softmax = topk_softmax_torch
    C.moe_sum = moe_sum_torch
    C.top_k_per_row_decode = top_k_per_row_decode_torch
    print("[fix928-v2] torch 版 moe_align/topk_softmax/moe_sum/top_k_per_row_decode "
          "已替换 gfx936-only C 算子", flush=True)


if _on and not _helper:
    try:
        _install()
    except Exception as _e:  # 失败宁可裸跑并留痕，不无声吞掉
        print(f"[fix928-v2] 安装失败: {_e!r}", flush=True)
elif not _on:
    print("[fix928-v2] FIX928=0，补丁停用", flush=True)
