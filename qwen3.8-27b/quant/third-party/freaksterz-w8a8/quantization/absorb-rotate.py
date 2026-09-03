#!/usr/bin/env python3
"""absorb-rotate.py — T4 step 1: offline QuaRot-style rotation of Qwen3.8-27B.

GOAL: multiply the residual stream by ONE orthogonal R (5120^2, random via QR)
so per-token activation vectors become near-isotropic -> dynamic INT8
quantization error drops hard. Fully OFFLINE: vLLM serves the result as a
plain checkpoint, zero runtime support.

MATH (x_rot = R x everywhere on the residual stream):
  weightless RMSNorm is rotation-EQUIVARIANT (||Rx|| = ||x||) -> all
  zero-centered norms ((1+w), modeling_qwen3_5.py:753) are absorbed into their
  consumers' input columns (diag(1+w) pre-multiply) and set to w=0 FIRST.
  consumers  (read post-norm stream): W' = W @ diag(1+w) @ R.T   (cols x R.T)
  producers  (emit into residual):   W' = R @ W                  (rows x R)
  embed/lm_head (residual vectors):  E' = E @ R.T, L' = L R.T (after absorb)
  residual adds are linear -> pass through. Everything INSIDE a block
  (q_norm/k_norm, GDN conv/A_log/dt_bias, linear_attn.norm, rotary) is
  post-projection and untouched.

MTP (BF16, drafter-only): pre_fc_norm_{embedding,hidden} absorbed into the
matching halves of mtp.fc's input cols; mtp.fc rows x R so the drafter's
internal residual is ALSO rotated (consistent with the shared lm_head, which
now reads the rotated basis); mtp.layers.0 treated like a main layer.
⚠ KNOWN COMPROMISE: mtp.norm feeds the SHARED lm_head, which already carries
the final-norm absorption — two different (1+w) folds into one matrix are
impossible. mtp.norm is ZEROED uncompensated (drafter-only perturbation;
acceptance impact is what the AL gate measures; norm weights are ~1+-small).

VISION (fixed 2026-08-15): the tower is internal and untouched, but its merger
`linear_fc2` emits INTO the residual stream, so it takes the producer rule
(rows x R) including its bias. Omitting it produced a silently TEXT-ONLY
checkpoint: text gates all pass, images arrive in the OLD basis. The finished
artifacts were repaired in place by `patch-vision-rotation.py` (exact: the
tower carries no quant scales and is byte-identical to the source at every
pipeline stage, so post-hoc rotation == rotating here).

Output: fp16 checkpoint + R stored for reversal/verification.
Sanity: per-tensor absmax before/after (orthogonal R preserves row norms);
assert counts everywhere. VERIFIED NEXT by verify-rotation.py (logit diff).
"""
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/glm/qwen38-bf16/Qwen3.8-27B")
DST = Path("/glm/qwen38-sq-w8a8/Qwen3.8-27B-BF16-rotated")
RPATH = "/glm/qwen38-sq-w8a8/rotation-R.pt"
H = 5120

t0 = time.time()
DST.mkdir(parents=True, exist_ok=True)

torch.manual_seed(20260815)
A = torch.randn(H, H, dtype=torch.float64)
R, _ = torch.linalg.qr(A)          # random orthogonal, fp64
R32 = R.float()
torch.save(R32, RPATH)

idx = json.load(open(SRC / "model.safetensors.index.json"))
wm = idx["weight_map"]

L = "model.language_model.layers"
layers = sorted({int(k.split(".")[3]) for k in wm if k.startswith(L + ".")})
assert len(layers) == 64, f"expected 64 layers, got {len(layers)}"

# ---- build op plan ----------------------------------------------------------
# absorb[norm_name] = list of (tensor_name, col_slice) to pre-multiply by
# diag(1+w); zero_norms = norms set to 0 with NO absorb (mtp.norm);
# cols_rt / rows_r = tensor lists for the rotation multiplies.
absorb: dict[str, list] = {}
zero_norms: list[str] = []
cols_rt: list[str] = []
rows_r: list[str] = []


def norm_consumers(layer_key, is_attn, is_mtp=False):
    p = f"{layer_key}."
    if is_attn:
        ins = [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
               p + "self_attn.v_proj.weight"]
    else:
        ins = [p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight",
               p + "linear_attn.in_proj_a.weight", p + "linear_attn.in_proj_b.weight"]
    posts = [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]
    return ins, posts


for i in layers:
    lk = f"{L}.{i}"
    is_attn = f"{lk}.self_attn.q_proj.weight" in wm
    ins, posts = norm_consumers(lk, is_attn)
    absorb[f"{lk}.input_layernorm.weight"] = [(n, None) for n in ins]
    absorb[f"{lk}.post_attention_layernorm.weight"] = [(n, None) for n in posts]
    cols_rt += ins + posts
    rows_r += [f"{lk}.self_attn.o_proj.weight", f"{lk}.linear_attn.out_proj.weight",
               f"{lk}.mlp.down_proj.weight"]

# final norm -> lm_head
absorb["model.language_model.norm.weight"] = [("lm_head.weight", None)]
cols_rt.append("lm_head.weight")

# embeddings: residual vectors
cols_rt.append("model.language_model.embed_tokens.weight")  # handled as E @ R.T

# MTP
FC = "mtp.fc.weight"
fc_in = None  # resolved at rewrite time from the tensor itself
absorb["mtp.pre_fc_norm_embedding.weight"] = [(FC, "embed_half")]
absorb["mtp.pre_fc_norm_hidden.weight"] = [(FC, "hidden_half")]
mlk = "mtp.layers.0"
ins, posts = norm_consumers(mlk, is_attn=True)
absorb[f"{mlk}.input_layernorm.weight"] = [(n, None) for n in ins]
absorb[f"{mlk}.post_attention_layernorm.weight"] = [(n, None) for n in posts]
cols_rt += ins + posts + [FC]
rows_r += [f"{mlk}.self_attn.o_proj.weight", f"{mlk}.mlp.down_proj.weight", FC]
zero_norms.append("mtp.norm.weight")

# VISION: the tower is internal and untouched, but its merger EMITS into the
# language residual stream, so linear_fc2 takes the producer rule (rows x R)
# INCLUDING its bias -- otherwise image features arrive in the OLD basis and the
# checkpoint is silently text-only (every text gate still passes). Single
# injection point: vision_config.deepstack_visual_indexes is empty.
VIS_OUT = ["model.visual.merger.linear_fc2.weight",
           "model.visual.merger.linear_fc2.bias"]
assert all(v in wm for v in VIS_OUT), "vision merger output tensors not found"
assert json.load(open(SRC / "config.json")).get(
    "vision_config", {}).get("deepstack_visual_indexes") in ([], None), \
    "deepstack injection present -- more vision tensors emit into the residual"
rows_r += VIS_OUT

# ---- apply ------------------------------------------------------------------
absorb_done: set = set()

def load_norm(n):
    shard = wm[n]
    f = load_file(str(SRC / shard))
    return f[n].float()

# pre-load all absorb vectors (norm tensors are tiny)
gains = {}
for n in absorb:
    w = load_norm(n)
    gains[n] = (1.0 + w)   # effective gain of the zero-centered norm
for n in zero_norms:
    gains[n] = None

# group tensor ops: which diag gains hit which tensor (+ optional half)
tensor_gains: dict[str, list] = {}
for norm, cons in absorb.items():
    for tn, half in cons:
        tensor_gains.setdefault(tn, []).append((gains[norm], half))

by_shard: dict[str, list] = {}
for tname, shard in wm.items():
    by_shard.setdefault(shard, []).append(tname)

report = []
for shard, names in sorted(by_shard.items()):
    tensors = load_file(str(SRC / shard))
    for n in list(tensors):
        if "visual" in n and n not in rows_r:
            continue  # tower internals untouched, byte-copied below
        orig_dtype = tensors[n].dtype
        x = tensors[n].float()
        pre_absmax = x.abs().max().item()
        # 1) norm absorption (pre-multiply input cols by diag gain)
        if n in tensor_gains:
            for g, half in tensor_gains[n]:
                C = x.shape[1]
                if half == "embed_half":
                    sl = slice(0, C // 2)
                elif half == "hidden_half":
                    sl = slice(C // 2, C)
                else:
                    sl = slice(None)
                assert C in (H, 2 * H), f"{n}: in_features {C} unexpected"
                x = x.clone()
                x[:, sl] = x[:, sl] * g.unsqueeze(0)
        # 2) rotation
        if n in cols_rt:
            if n == "model.language_model.embed_tokens.weight":
                x = x @ R32.T                       # embedding rows -> rotated
            elif n == FC:
                C = x.shape[1]
                assert C == 2 * H, f"mtp.fc in_features {C} != {2*H}"
                x = torch.cat([x[:, :H] @ R32.T, x[:, H:] @ R32.T], dim=1)
            else:
                x = x @ R32.T
        if n in rows_r:
            x = R32 @ x
        # 3) zero the norms themselves
        if n in absorb or n in zero_norms:
            x = torch.zeros_like(x)
        post_absmax = x.abs().max().item()
        if n in cols_rt or n in rows_r:
            report.append((n, pre_absmax, post_absmax))
        # language stack is emitted fp16; the rotated vision merger keeps the
        # tower's own dtype so the visual namespace stays homogeneous
        out_dtype = orig_dtype if "visual" in n else torch.float16
        tensors[n] = x.to(out_dtype)
    # untouched visual tensors stay as-is (bf16); copy rest verbatim
    save_file(tensors, str(DST / shard), metadata={"format": "pt"})
    print(f"  {shard} done ({time.time()-t0:.0f}s)", flush=True)
    del tensors

for f in SRC.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors"):
        shutil.copy2(f, DST / f.name)
cfg = json.load(open(DST / "config.json"))
cfg["torch_dtype"] = "float16"
if isinstance(cfg.get("text_config"), dict):
    cfg["text_config"]["torch_dtype"] = "float16"
json.dump(cfg, open(DST / "config.json", "w"), indent=2)

# sanity report: orthogonal transforms preserve row/col norms in expectation;
# absmax drift >>10x on any tensor = a bug
drift = [max(q, p) / max(min(q, p), 1e-12) for _, q, p in report]
import math
worst = sorted(zip(drift, [r[0] for r in report]), reverse=True)[:3]
print(f"rotated {len(report)} tensors; worst absmax drift: "
      + ", ".join(f"{n} x{d:.2f}" for d, n in worst), flush=True)
assert all(d < 20 for d in drift), "absmax drift >20x — rotation bug"
print(f"TOTAL {time.time()-t0:.0f}s -> {DST} (+ R at {RPATH})", flush=True)
