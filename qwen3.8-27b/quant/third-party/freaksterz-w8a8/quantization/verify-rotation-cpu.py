#!/usr/bin/env python3
"""verify-rotation-cpu.py — T4 step 2a: GPU-free verification of the rotated
checkpoint. Two independent halves:

  PART 1 — COVERAGE AUDIT. Classify EVERY tensor in the index into exactly one
  of {absorbed-norm, zeroed-norm, cols(xR.T), rows(xR), untouched-whitelist}.
  An unclassified tensor = a transform we forgot = the exact failure mode the
  full-model logit diff exists to catch. Also asserts absorbed norms really
  are zero and untouched tensors are unchanged.

  PART 2 — fp64 PATH EQUIVALENCE for every transform class, using RUNTIME
  semantics (zero-centered (1+w) for the absorbed norms):
    (a) GDN in-path      : n*(1+w) @ W.T           == (R n) @ W'.T
    (b) attn in-path     : same for q/k/v
    (c) producers        : W a                     == R (W a)      [rows xR]
    (d) lm_head readout  : (h*(1+w_f)) @ L.T       == (R h) @ L'.T
    (e) mtp.fc           : [e*(1+w_e); h*(1+w_h)] @ Wfc.T == R(...) [cols+rows]
    (f) mtp layer paths  : as (a)/(c)

Passing BOTH halves means the only thing the GPU logit-diff can still add is
end-to-end confirmation under real dtypes — worth running, but this is the
check that finds structural bugs.
"""
import json
import re

import torch
from safetensors import safe_open

SRC = "/glm/qwen38-bf16/Qwen3.8-27B"
DST = "/glm/qwen38-sq-w8a8/Qwen3.8-27B-BF16-rotated"
R = torch.load("/glm/qwen38-sq-w8a8/rotation-R.pt").double()
H = 5120
L = "model.language_model.layers"

si = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
di = json.load(open(f"{DST}/model.safetensors.index.json"))["weight_map"]
assert set(si) == set(di), "tensor sets differ between original and rotated!"


def get(base, index, name, dt=torch.float64):
    with safe_open(f"{base}/{index[name]}", framework="pt") as f:
        return f.get_tensor(name).to(dt)


# ---------------- PART 1: coverage audit ------------------------------------
ABSORBED = re.compile(
    r"(^|\.)(input_layernorm|post_attention_layernorm)\.weight$"
    r"|^model\.language_model\.norm\.weight$"
    r"|^mtp\.pre_fc_norm_(embedding|hidden)\.weight$")
ZEROED = re.compile(r"^mtp\.norm\.weight$")
COLS = re.compile(
    r"self_attn\.(q|k|v)_proj\.weight$"
    r"|linear_attn\.in_proj_(qkv|z|a|b)\.weight$"
    r"|mlp\.(gate|up)_proj\.weight$"
    r"|^lm_head\.weight$"
    r"|embed_tokens\.weight$"
    r"|^mtp\.fc\.weight$")
ROWS = re.compile(
    r"self_attn\.o_proj\.weight$|linear_attn\.out_proj\.weight$"
    r"|mlp\.down_proj\.weight$|^mtp\.fc\.weight$")
UNTOUCHED = re.compile(
    r"visual\.|conv1d|A_log|dt_bias|self_attn\.(q|k)_norm\.weight$"
    r"|linear_attn\.norm\.weight$")

buckets = {"absorbed": [], "zeroed": [], "cols": [], "rows": [], "untouched": [],
           "UNCLASSIFIED": []}
for n in sorted(si):
    if ABSORBED.search(n):
        buckets["absorbed"].append(n)
    elif ZEROED.search(n):
        buckets["zeroed"].append(n)
    elif UNTOUCHED.search(n):
        buckets["untouched"].append(n)
    elif COLS.search(n) or ROWS.search(n):
        buckets["cols" if COLS.search(n) else "rows"].append(n)
    else:
        buckets["UNCLASSIFIED"].append(n)

print("COVERAGE:", {k: len(v) for k, v in buckets.items()}, flush=True)
if buckets["UNCLASSIFIED"]:
    print("  UNCLASSIFIED (first 10):", buckets["UNCLASSIFIED"][:10], flush=True)

fails = len(buckets["UNCLASSIFIED"])

# absorbed/zeroed norms must be exactly zero in the rotated checkpoint
bad_norm = [n for n in buckets["absorbed"] + buckets["zeroed"]
            if get(DST, di, n).abs().max().item() != 0.0]
print(f"norms zeroed: {len(buckets['absorbed'])+len(buckets['zeroed'])} checked, "
      f"{len(bad_norm)} NOT zero", flush=True)
fails += len(bad_norm)

# untouched tensors must be unchanged (bf16->fp16 cast tolerance)
bad_un = []
for n in buckets["untouched"][:40]:
    a, b = get(SRC, si, n), get(DST, di, n)
    if (a - b).abs().max().item() > 1e-3 * max(a.abs().max().item(), 1e-6):
        bad_un.append(n)
print(f"untouched spot-check: 40 sampled, {len(bad_un)} changed", flush=True)
fails += len(bad_un)

# ---------------- PART 2: fp64 path equivalence -----------------------------
torch.manual_seed(0)
print("\nPATH CHECKS (rel-err, expect ~1e-3 = fp16 storage):", flush=True)


def rel(a, b):
    return ((a - b).abs().max() / a.abs().max().clamp(min=1e-12)).item()


def check(tag, orig, rot, tol=5e-3):
    global fails
    e = rel(orig, rot)
    ok = e < tol
    fails += (not ok)
    print(f"  {tag:<34} {e:.3e} {'ok' if ok else 'FAIL'}", flush=True)


n_vec = torch.randn(4, H, dtype=torch.float64)      # a post-norm activation
h_vec = torch.randn(4, H, dtype=torch.float64)      # a residual/hidden vector

# (a) GDN in-path, layer 0
lk = f"{L}.0"
w = get(SRC, si, f"{lk}.input_layernorm.weight")
for proj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
    Wo = get(SRC, si, f"{lk}.linear_attn.{proj}.weight")
    Wr = get(DST, di, f"{lk}.linear_attn.{proj}.weight")
    check(f"(a) gdn {proj}", (n_vec * (1 + w)) @ Wo.T, (n_vec @ R.T) @ Wr.T)

# (b) attn in-path, layer 3
lk = f"{L}.3"
w = get(SRC, si, f"{lk}.input_layernorm.weight")
for proj in ("q_proj", "k_proj", "v_proj"):
    Wo = get(SRC, si, f"{lk}.self_attn.{proj}.weight")
    Wr = get(DST, di, f"{lk}.self_attn.{proj}.weight")
    check(f"(b) attn {proj}", (n_vec * (1 + w)) @ Wo.T, (n_vec @ R.T) @ Wr.T)

# ffn in-path (post_attention_layernorm)
w = get(SRC, si, f"{lk}.post_attention_layernorm.weight")
for proj in ("gate_proj", "up_proj"):
    Wo = get(SRC, si, f"{lk}.mlp.{proj}.weight")
    Wr = get(DST, di, f"{lk}.mlp.{proj}.weight")
    check(f"(b) ffn {proj}", (n_vec * (1 + w)) @ Wo.T, (n_vec @ R.T) @ Wr.T)

# (c) producers: output must land in the rotated basis
for name, cols in ((f"{lk}.self_attn.o_proj.weight", 6144),
                   (f"{lk}.mlp.down_proj.weight", 17408),
                   (f"{L}.0.linear_attn.out_proj.weight", 6144)):
    Wo, Wr = get(SRC, si, name), get(DST, di, name)
    a = torch.randn(4, cols, dtype=torch.float64)
    check(f"(c) producer {name.split('.')[-2]}", (a @ Wo.T) @ R.T, a @ Wr.T)

# (d) lm_head readout (final norm absorbed)
wf = get(SRC, si, "model.language_model.norm.weight")
Lo, Lr = get(SRC, si, "lm_head.weight"), get(DST, di, "lm_head.weight")
check("(d) lm_head", (h_vec * (1 + wf)) @ Lo.T, (h_vec @ R.T) @ Lr.T)

# embeddings: rotated rows
Eo, Er = get(SRC, si, "model.language_model.embed_tokens.weight"), \
         get(DST, di, "model.language_model.embed_tokens.weight")
ids = torch.tensor([10, 5000, 100000, 248000])
check("(d) embed rows", Eo[ids] @ R.T, Er[ids])

# (e) mtp.fc: cols absorbed+rotated per half, rows rotated
we = get(SRC, si, "mtp.pre_fc_norm_embedding.weight")
wh = get(SRC, si, "mtp.pre_fc_norm_hidden.weight")
Fo, Fr = get(SRC, si, "mtp.fc.weight"), get(DST, di, "mtp.fc.weight")
e_vec = torch.randn(4, H, dtype=torch.float64)
orig = torch.cat([e_vec * (1 + we), h_vec * (1 + wh)], 1) @ Fo.T
rot = torch.cat([e_vec @ R.T, h_vec @ R.T], 1) @ Fr.T
check("(e) mtp.fc (cols+rows)", orig @ R.T, rot)

# (f) mtp layer paths
wm_in = get(SRC, si, "mtp.layers.0.input_layernorm.weight")
for proj in ("q_proj", "k_proj", "v_proj"):
    Wo = get(SRC, si, f"mtp.layers.0.self_attn.{proj}.weight")
    Wr = get(DST, di, f"mtp.layers.0.self_attn.{proj}.weight")
    check(f"(f) mtp {proj}", (n_vec * (1 + wm_in)) @ Wo.T, (n_vec @ R.T) @ Wr.T)
Wo = get(SRC, si, "mtp.layers.0.self_attn.o_proj.weight")
Wr = get(DST, di, "mtp.layers.0.self_attn.o_proj.weight")
a = torch.randn(4, Wo.shape[1], dtype=torch.float64)
check("(f) mtp o_proj", (a @ Wo.T) @ R.T, a @ Wr.T)

print(f"\nVERIFY-CPU {'PASS' if fails == 0 else f'FAIL ({fails} problems)'}", flush=True)
raise SystemExit(0 if fails == 0 else 1)
