# SPDX-License-Identifier: Apache-2.0
"""GDN (gated delta net) "all"-mode prefix caching helpers.

为 Qwen3.5 混合模型的 GDN 线性注意力层实现 mamba_cache_mode="all"：
- builder 侧：计算三块索引张量 + prefill 分段（CPU，避免新增 D2H 同步）
- mixer 侧：分段链式调用 chunk 内核 + 块边界状态/conv 快照存档 + decode 跨界搬移

设计对齐 vllm/v1/attention/backends/mamba_attn.py 与 mamba_mixer2.py 的
"all" 模式语义（block_idx_last_computed / first_scheduled / last_scheduled）。
Phase 1 约束：不支持投机解码（与 mamba2 上游一致）；建议 mamba_block_size ==
max_num_batched_tokens（则每步每序列最多 1 个内部边界，分段轮数 <= 2）。
"""
from dataclasses import dataclass

import torch


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


@dataclass
class GDNAllModeMetadata:
    """all 模式下 GDN 需要的附加元数据（builder 产出，mixer 消费）。

    仅 prefill 相关字段按 prefill 行序（batch 内 decode 在前、prefill 在后，
    与 non_spec_state_indices_tensor 的行序一致）。
    """
    # 全 batch（decode 行在前）
    block_idx_last_computed: torch.Tensor      # (num_reqs,) int32
    block_idx_last_scheduled: torch.Tensor     # (num_reqs,) int32
    # 仅 prefill
    block_idx_first_scheduled_p: torch.Tensor  # (num_prefills,) int32
    num_computed_tokens_p: torch.Tensor        # (num_prefills,) int32 (CPU 副本另存)
    # 分段调度（CPU 构建，H2D 一次上传）
    # 轮 r 的 varlen 调用：seg_cu_seqlens[r] (n_r+1,)，seg_token_index[r] (tokens_r,)
    # seg_src_rows[r] (n_r,) —— 参与轮 r 的 prefill 行号（用于取上一轮 final state）
    # seg_dst_slot[r] (n_r,) —— 轮 r 结束状态要写入的槽 id（-1 表示不写，仅传递）
    seg_cu_seqlens: list          # list[torch.Tensor(device)]
    seg_token_index: list         # list[torch.Tensor(device)]
    seg_src_rows: list            # list[torch.Tensor(device)]
    seg_dst_slot: list            # list[torch.Tensor(device)]
    # conv 边界快照：boundary_tok_pos[i] 是该快照取材的 token 终点位置（varlen 平铺坐标），
    # boundary_conv_slot[i] 是要写入的 conv 槽 id
    boundary_tok_pos: torch.Tensor            # (n_boundaries,) int64 device
    boundary_conv_slot: torch.Tensor          # (n_boundaries,) int64 device
    boundary_q_beg: torch.Tensor              # (n_boundaries,) int64 device
    boundary_src_slot: torch.Tensor           # (n_boundaries,) int64 device
    num_rounds: int


def compute_all_mode_metadata(
    *,
    block_size: int,
    block_table_tensor: torch.Tensor,          # (num_reqs, max_blocks) device
    num_computed_tokens_cpu: torch.Tensor,     # (num_reqs,) CPU
    seq_lens_cpu: torch.Tensor,                # (num_reqs,) CPU（含本步新 token）
    query_start_loc_cpu: torch.Tensor,         # (num_reqs+1,) CPU（varlen 平铺）
    num_decodes: int,
    num_prefills: int,
    device: torch.device,
) -> GDNAllModeMetadata:
    """CPU 上一次性构建 all 模式全部元数据（builder 内调用）。"""
    B = block_size
    nc = num_computed_tokens_cpu.to(torch.int64)
    sl = seq_lens_cpu.to(torch.int64)

    blk_last_computed = torch.clamp(cdiv_t(nc, B) - 1, min=0)
    blk_last_sched = torch.clamp(cdiv_t(sl, B) - 1, min=0)
    blk_first_sched = cdiv_t(nc + 1, B) - 1

    # ---- prefill 分段（只对 prefill 行；行序：decode 在前）----
    seg_cu, seg_tok, seg_src, seg_dst = [], [], [], []
    bnd_pos: list[int] = []
    bnd_slot: list[int] = []
    bnd_qbeg: list[int] = []
    bnd_src: list[int] = []
    # 每个 prefill 行的分段列表：[(tok_beg, tok_end, dst_slot_or_-1), ...]
    per_seq_segments: list[list[tuple[int, int, int]]] = []
    bt_cpu = None  # 惰性拉取块表（一次 D2H，builder 阶段可接受）
    for i in range(num_prefills):
        row = num_decodes + i
        q_beg = int(query_start_loc_cpu[row].item())
        q_end = int(query_start_loc_cpu[row + 1].item())
        start = int(nc[row].item())          # 本步第一个新 token 的绝对位置
        end = start + (q_end - q_beg)        # 本步结束后的绝对位置
        segs: list[tuple[int, int, int]] = []
        cur = start
        while cur < end:
            nxt_boundary = ((cur // B) + 1) * B
            seg_end = min(nxt_boundary, end)
            if bt_cpu is None:
                bt_cpu = block_table_tensor.cpu()
            if seg_end == nxt_boundary and seg_end <= end:
                # 段末正好是块边界：状态写入该块的槽
                blk_idx = seg_end // B - 1
                dst = int(bt_cpu[row, blk_idx].item())
                # conv 边界快照（取材终点=该边界在 varlen 平铺里的位置；
                # 记录本序列平铺起点与 src 槽，供不足 width-1 时接续旧状态）
                bnd_pos.append(q_beg + (seg_end - start))
                bnd_slot.append(dst)
                bnd_qbeg.append(q_beg)
                bnd_src.append(int(bt_cpu[
                    row, int(blk_last_computed[row].item())].item()))
            else:
                # partial 末段：写 last_scheduled 槽
                dst = int(bt_cpu[row, int(blk_last_sched[row].item())].item())
            segs.append((q_beg + (cur - start), q_beg + (seg_end - start), dst))
            cur = seg_end
        per_seq_segments.append(segs)

    num_rounds = max((len(s) for s in per_seq_segments), default=0)
    for r in range(num_rounds):
        cu = [0]
        toks: list[int] = []
        rows: list[int] = []
        dsts: list[int] = []
        for i, segs in enumerate(per_seq_segments):
            if r >= len(segs):
                continue
            beg, endp, dst = segs[r]
            toks.extend(range(beg, endp))
            cu.append(len(toks))
            rows.append(i)
            dsts.append(dst)
        seg_cu.append(torch.tensor(cu, dtype=torch.int32, device=device))
        seg_tok.append(torch.tensor(toks, dtype=torch.int64, device=device))
        seg_src.append(torch.tensor(rows, dtype=torch.int64, device=device))
        seg_dst.append(torch.tensor(dsts, dtype=torch.int64, device=device))

    return GDNAllModeMetadata(
        block_idx_last_computed=blk_last_computed.to(torch.int32).to(device),
        block_idx_last_scheduled=blk_last_sched.to(torch.int32).to(device),
        block_idx_first_scheduled_p=blk_first_sched[num_decodes:].to(
            torch.int32).to(device),
        num_computed_tokens_p=nc[num_decodes:].to(torch.int32).to(device),
        seg_cu_seqlens=seg_cu,
        seg_token_index=seg_tok,
        seg_src_rows=seg_src,
        seg_dst_slot=seg_dst,
        boundary_tok_pos=torch.tensor(bnd_pos, dtype=torch.int64, device=device),
        boundary_conv_slot=torch.tensor(bnd_slot, dtype=torch.int64,
                                        device=device),
        boundary_q_beg=torch.tensor(bnd_qbeg, dtype=torch.int64, device=device),
        boundary_src_slot=torch.tensor(bnd_src, dtype=torch.int64,
                                       device=device),
        num_rounds=num_rounds,
    )


def cdiv_t(a: torch.Tensor, b: int) -> torch.Tensor:
    return -(a // -b)


def gather_slots(block_table_row_major: torch.Tensor,
                 block_idx: torch.Tensor) -> torch.Tensor:
    """slots[i] = block_table[i, block_idx[i]]（1D gather）。"""
    return block_table_row_major.gather(
        1, block_idx.to(torch.int64).unsqueeze(1)).squeeze(1)


@torch.no_grad()
def prefill_chunked_with_checkpoints(
    *,
    chunk_gated_delta_rule,        # 可调用（层上的 self.chunk_gated_delta_rule）
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,   # (1, T, H, D)
    g: torch.Tensor, beta: torch.Tensor,                  # (1, T, H)
    ssm_state: torch.Tensor,                              # 槽池 (num_slots, H, Dk, Dv)
    init_slot: torch.Tensor,       # (num_prefills,) 初始状态槽 id
    has_initial: torch.Tensor,     # (num_prefills,) bool
    meta: GDNAllModeMetadata,
    core_out: torch.Tensor,        # (1, T, Hv, Dv) 预分配输出（varlen 平铺）
) -> None:
    """分段链式 prefill：逐轮调用 chunk 内核，边界状态写档。

    轮 0 的 initial 来自 init_slot（has_initial 掩零）；
    轮 r>0 的 initial 是上一轮对应序列的 final（链式传递）。
    每轮结束把 final 写入 seg_dst_slot[r]（partial 末段也写，语义与 mamba2 一致）。
    """
    num_p = init_slot.shape[0]
    if num_p == 0 or meta.num_rounds == 0:
        return
    # 活跃状态缓存：per prefill row 的当前链式状态（与缓存同 dtype，语义对齐原实现）
    st = ssm_state[init_slot.to(torch.int64)].clone()
    st[~has_initial] = 0
    for r in range(meta.num_rounds):
        tok = meta.seg_token_index[r]
        rows = meta.seg_src_rows[r]
        cu = meta.seg_cu_seqlens[r]
        q_r = q.index_select(1, tok)
        k_r = k.index_select(1, tok)
        v_r = v.index_select(1, tok)
        g_r = g.index_select(1, tok)
        b_r = beta.index_select(1, tok)
        init_r = st.index_select(0, rows).contiguous()
        out_r, final_r = chunk_gated_delta_rule(
            q=q_r, k=k_r, v=v_r, g=g_r, beta=b_r,
            initial_state=init_r,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False,
        )
        core_out.index_copy_(1, tok, out_r.to(core_out.dtype))
        # 状态写档 + 链式更新
        ssm_state[meta.seg_dst_slot[r]] = final_r.to(ssm_state.dtype)
        st.index_copy_(0, rows, final_r.to(st.dtype))


@torch.no_grad()
def conv_boundary_snapshots(
    conv_state: torch.Tensor,       # (num_slots, dim, width-1)  [DS 布局视角]
    mixed_qkv_flat: torch.Tensor,   # (T, dim) —— conv 之前的原始投影
    meta: GDNAllModeMetadata,
    width_minus_1: int,
) -> None:
    """块边界的 conv 缓存 = 边界前 width-1 个原始输入 token（causal conv 语义）。"""
    n = meta.boundary_tok_pos.shape[0]
    if n == 0:
        return
    for j in range(n):
        endp = int(meta.boundary_tok_pos[j].item())
        slot = int(meta.boundary_conv_slot[j].item())
        q_beg = int(meta.boundary_q_beg[j].item())
        lo = max(endp - width_minus_1, q_beg)   # 不越过本序列本步起点
        seg = mixed_qkv_flat[lo:endp].transpose(0, 1)         # (dim, navail)
        navail = endp - lo
        if navail < width_minus_1:
            # 边界距本步起点不足 width-1：缺的部分正是 src 槽里
            # 上一步末尾的原始输入（conv 缓存语义）
            src = int(meta.boundary_src_slot[j].item())
            prev = conv_state[src][:, navail - width_minus_1:]
            full = torch.cat([prev.to(seg.dtype), seg], dim=1)
        else:
            full = seg
        conv_state[slot] = full.to(conv_state.dtype)


@torch.no_grad()
def decode_cross_block_copy(
    *,
    kv_caches: tuple,               # (conv_state, ssm_state)
    state_indices_2d: torch.Tensor, # (num_decodes, max_blocks)
    blk_last_computed_d: torch.Tensor,
    blk_last_scheduled_d: torch.Tensor,
) -> torch.Tensor:
    """decode 跨块边界：把旧槽状态拷到新槽，返回本步应使用的槽 id（=新槽）。"""
    src = gather_slots(state_indices_2d, blk_last_computed_d).to(torch.int64)
    dst = gather_slots(state_indices_2d, blk_last_scheduled_d).to(torch.int64)
    # 无条件向量拷贝：src==dst 时为等值自拷（无害），
    # 避免 .any() 的每层每步 D2H 同步（48 层 = 48 次同步/步）
    conv_state, ssm_state = kv_caches
    ssm_state[dst] = ssm_state[src]
    conv_state[dst] = conv_state[src]
    return dst
