"""迷你链 v5n（bf16/NEXTN 变体：q<=4 直走原生）：minichain4 + raw-q8 verifier（DocPang v1.2.1 吸收，2026-09-01）
① gfx928 paged-varlen 修复（官方 repair.py 原件）
② q=8 verify 三级路由：
   raw paged_attention 直调（精确几何 + flash-attn 260602 legacy ABI 版本锁）
   → 不满足则 2× 原生 q=4（q8split，minichain4 原路径）
   → 非 q8 形状走 triton 修复 / 原生长 prefill。
   出处：DocPang hotfixes/v1.2.1 Agent128K —— legacy ABI（我们这个 flash-attn
   构建）raw-q8 即「旧冠军」路径，长上下文 decode 比 2×q4 快 8-20%，
   isolated 16K/64K/128K/257.9K 与 2×q4 bitwise equal。
   版本锁：flash-attn != 260602 legacy 时自动禁用 raw，整链退回 minichain4 行为。
③ q16k 长 prefill 桶（官方原件）。
"""
import os
import runpy

# ── 第零层：0.5.20 回输（2026-09-19）────────────────────────────────
#   海光侧无 0.5.20（harbor 全站搜零命中），整版升级无版本可升，故走回输。
#   ★ 必须放在**最前**：镜像自带的第三层 q16k 原件会抛
#     `RuntimeError: TP4-q16 fail-closed: SGLANG_Q38_TP4_Q16_KV_LENGTHS must be
#     explicit`（已记为故障闸常数第 11 条，基线里本就有），sitecustomize 在那里
#     中断，排它之后的任何代码都执行不到。回输是平台无关的纯 Python 逻辑、
#     不依赖前面各层，放最前既绕开该既有断点，又不改变原链路行为。
#   两条均为平台无关（不碰算子/显存/通信），详见
#   deploy-kit/patches/sglang-hrrn-backport/README.md
#     · patch_radix：RadixKey galloping match + offset 游标（上游 #37324）
#         0.5.12 原生 match 是逐 token Python 循环；40 万 token 全命中
#         实测 24.1ms → 3.9ms（6×）。对拍 2000 例（含 bigram/ps∈{1,2,64}）零不符。
#         **默认启用**——纯加速、语义逐位复刻原生。
#     · patch_hrrn：HRRN 调度策略（上游 #32911，实测 TTFT −69%）
#         **默认不启用**，需 --schedule-policy hrrn 才生效，便于 A/B 对照。
try:
    import patch_radix as _pr
    print("[backport]", _pr.apply(), flush=True)
except Exception as _e:                     # 回输失败不得拖垮起服
    print(f"[backport] radix 回输跳过: {_e!r}", flush=True)

try:
    import patch_hrrn as _ph
    print("[backport]", _ph.apply(), flush=True)
except Exception as _e:
    print(f"[backport] hrrn 回输跳过: {_e!r}", flush=True)


runpy.run_path(
    "/data/qwen38-27b-k100ai-int8-opt/runtime_patch_sglang_gfx928_paged_varlen_repair_v1/repair.py",
    run_name="__repair_parent__",
)

import importlib.metadata as _metadata

import torch
from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai

_parent = _fab.vllm_flash_attn_varlen_func            # = triton 修复
_native = _fai.vllm_flash_attn_varlen_func_interface  # = 原生 raw varlen
_seen = None
_cuq_cache = {}

# ── raw-q8 开关：环境变量 + flash-attn 版本锁（legacy 260602 ABI 无 layout 参数）──
_LEGACY_ABI = "2.8.3+das.opt1.dtk2604.torch290.2606021702.ge93bd4"
_raw_q8 = os.getenv("MC5_RAW_Q8", "1").strip().lower() in ("1", "true", "yes", "on")
_fa2 = None
if _raw_q8:
    _fa_ver = _metadata.version("flash-attn")
    if _fa_ver == _LEGACY_ABI:
        import flash_attn_2_cuda as _fa2
    else:
        _raw_q8 = False
        print(f"[minichain5] flash-attn={_fa_ver} 非 legacy ABI，raw-q8 禁用，退回 q8split", flush=True)


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


def _q8_route(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k,
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
        if qlen <= 4 or qlen > 512:  # q<=4 铁律安全区直走原生（NEXTN verify 必经；bf16 线 2026-09-03 定界）
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

    if _raw_q8:
        # 审计过的精确几何：q(8,QH6,D256) bf16；k(pages,KVH1,page64,D256)；v(pages,KVH1,D256,page64)
        exact = (
            q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and tuple(q.shape) == (8, 6, 256)
            and int(k.ndim) == 4 and int(v.ndim) == 4
            and int(k.shape[1]) == 1 and int(v.shape[1]) == 1
            and int(k.shape[2]) == 64 and int(k.shape[3]) == 256
            and int(v.shape[2]) == 256 and int(v.shape[3]) == 64
            and int(fa_version) == 2
        )
        if exact:
            if _seen != "raw":
                _seen = "raw"
                print("[minichain5] ACTIVE q=8 verifier -> single raw paged_attention (legacy ABI)", flush=True)
            out = torch.empty_like(q)
            _fa2.paged_attention(
                out, q.reshape(1, 8, 6, 256), k, v, softmax_scale,
                block_table, seqused_k, None, "auto",
                q_descale, k_descale, v_descale, max_seqlen_k, None,
            )
            return out
        # 几何不符：不 fail-closed，安全回退 q8split

    if _seen != "split":
        _seen = "split"
        print("[minichain5] ACTIVE q=8 verifier -> 2x native q=4 (fallback)", flush=True)
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
        raise RuntimeError("minichain5 q8split expected tensor outputs")
    return torch.cat((out0, out1), dim=0)


_fai.vllm_flash_attn_varlen_func = _q8_route
_fab.vllm_flash_attn_varlen_func = _q8_route

# ── 第三层：q16k 长 prefill 加速（官方原件，自含无父链）──
runpy.run_path(
    "/data/qwen38-27b-k100ai-int8-opt/runtime_patch_sglang_tp4_u036_q16k_v1/sitecustomize.py",
    run_name="__q16k_member__",
)
print(f"[minichain5] repair + q8(raw={_raw_q8})/split + q16k installed", flush=True)
