"""迷你链 v2：① gfx928 paged-varlen 修复（官方 repair.py 原件）
② 去毒版 q8split —— 官方 runtime_patch_dflash_q8split_v1 的等价改写：
   去掉 runpy 父链装载（父链有 TP-gather 毒钩子），父路由直接指向 ① 的 triton 修复。
   行为：batch1 causal q=8 verify → 2× 原生 q=4（铁律安全区，原生速度）；
   其余多 token 形状 → triton 修复（正确性兜底）。
"""
import runpy

runpy.run_path(
    "/data/qwen38-27b-k100ai-int8-opt/runtime_patch_sglang_gfx928_paged_varlen_repair_v1/repair.py",
    run_name="__repair_parent__",
)

import torch
from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai

_parent = _fab.vllm_flash_attn_varlen_func            # = triton 修复
_native = _fai.vllm_flash_attn_varlen_func_interface  # = 原生 raw
_seen = False
_cuq_cache = {}


def _cuq(device, qlen):
    key = (device.index, qlen)
    t = _cuq_cache.get(key)
    if t is None:
        t = torch.tensor([0, qlen], device=device, dtype=torch.int32)
        _cuq_cache[key] = t
    return t


def _native_call(*, q, k, v, qlen, seqused_k, max_seqlen_k, softmax_scale, causal,
                 window_size, block_table, fa_version, q_descale, k_descale, v_descale):
    return _native(
        q=q, k=k, v=v,
        cu_seqlens_q=_cuq(q.device, qlen), max_seqlen_q=qlen,
        seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        block_table=block_table, fa_version=fa_version,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )


def _q8_split_native_else_parent(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k,
                                 max_seqlen_k, softmax_scale, causal, window_size,
                                 block_table, fa_version, q_descale, k_descale, v_descale):
    global _seen
    qlen = int(max_seqlen_q)
    batch = int(cu_seqlens_q.numel() - 1)
    eligible = (
        qlen == 8
        and int(q.shape[0]) == 8
        and batch == 1
        and int(seqused_k.numel()) == 1
        and bool(causal)
        and window_size == (-1, -1)
        and block_table is not None
    )
    if not eligible:
        if qlen > 512:
            # prefill 主体/长尾：原生路径历来正确（q>=873 有逐位证据），且比 triton 快 ~1.6-2.4x
            return _native(
                q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
                causal=causal, window_size=window_size, block_table=block_table,
                fa_version=fa_version, q_descale=q_descale, k_descale=k_descale,
                v_descale=v_descale,
            )
        return _parent(
            q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size, block_table=block_table,
            fa_version=fa_version, q_descale=q_descale, k_descale=k_descale,
            v_descale=v_descale,
        )
    if not _seen:
        _seen = True
        print("[minichain2 q8split] ACTIVE batch1 causal q=8 -> 2x native q=4", flush=True)
    seq_k_first = seqused_k - 4
    out0 = _native_call(
        q=q[:4], k=k, v=v, qlen=4, seqused_k=seq_k_first,
        max_seqlen_k=max(1, int(max_seqlen_k) - 4),
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        block_table=block_table, fa_version=fa_version,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    out1 = _native_call(
        q=q[4:], k=k, v=v, qlen=4, seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        block_table=block_table, fa_version=fa_version,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    if isinstance(out0, (tuple, list)) or isinstance(out1, (tuple, list)):
        raise RuntimeError("minichain2 q8split expected tensor outputs")
    return torch.cat((out0, out1), dim=0)


_fai.vllm_flash_attn_varlen_func = _q8_split_native_else_parent
_fab.vllm_flash_attn_varlen_func = _q8_split_native_else_parent
print("[minichain3] repair + depoisoned q8split installed", flush=True)

# ── 第三层：q16k 长 prefill 加速（官方原件，自含无父链）──
runpy.run_path(
    "/data/qwen38-27b-k100ai-int8-opt/runtime_patch_sglang_tp4_u036_q16k_v1/sitecustomize.py",
    run_name="__q16k_member__",
)
print("[minichain4] repair + q8split + q16k(prefill buckets) installed", flush=True)
