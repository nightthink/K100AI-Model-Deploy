"""vllm0.29 flashnext 镜像 gfx928 救援补丁（2026-09-02）

镜像的 _C / _moe_C 只编了 gfx936（打包人 PYTORCH_ROCM_ARCH=gfx936），
在 gfx928 上 C 算子静默产垃圾：moe_align 输出未初始化 buffer →
fused_moe token_mask 被负值骗过 → 按负偏移读显存 → XID:81 VMFault
（与 DSv4 项目轮次35 定位的 sgl_kernel moe_align 垃圾同型）。

本补丁把 qwen3_next 路径上直调 _moe_C 的三个算子换成纯 torch 等价实现
（moe_align 含哨兵预填充），CustomOp 家族（rms_norm 等）由启动参数
--compilation-config '{"custom_ops":["none"]}' 走 forward_native。
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

    C.moe_align_block_size = moe_align_torch
    C.topk_softmax = topk_softmax_torch
    C.moe_sum = moe_sum_torch
    print("[fix928] torch 版 moe_align/topk_softmax/moe_sum 已替换 gfx936-only C 算子", flush=True)


if _on and not _helper:
    try:
        _install()
    except Exception as _e:  # 失败宁可裸跑并留痕，不无声吞掉
        print(f"[fix928] 安装失败: {_e!r}", flush=True)
elif not _on:
    print("[fix928] FIX928=0，补丁停用", flush=True)
