#!/usr/bin/env python3
"""apply-smoothing-rot.py — T4 step 6: fold the SURVIVING smoothing classes
onto the rotated checkpoint.

After rotation, classes A (ffn-in) and B (attn/gdn-in) are GONE: their fold
target — the zero-centered norm weight — has been absorbed to weightless, and
per-channel scaling of the residual stream is impossible (it is shared by all
64 layers). Rotation replaces them.

Surviving folds (none touch the residual stream):
  C down-in : up_proj ROWS / s  -> down_proj cols * s   (SiLU product)
  D gdn-out : linear_attn.norm (PLAIN, [128] per value-head dim) / s
              -> out_proj cols * s(repeat)
  E attn-out: v_proj ROWS / t   -> o_proj cols * t      (GQA-constrained)

Scales come from alpha-proxy-rot.py (rot/scales-rot.pt), chosen per hook on
the ROTATED activations.
"""
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/glm/qwen38-sq-w8a8/Qwen3.8-27B-BF16-rotated")
DST = Path("/glm/qwen38-sq-w8a8/Qwen3.8-27B-ROT-smoothed")
SCALES = "/glm/qwen38-sq-w8a8/rot/scales-rot.pt"
OUT_DTYPE = torch.float16
N_Q, N_KV, HD = 24, 4, 256
G = N_Q // N_KV
VHD = 128

t0 = time.time()
DST.mkdir(parents=True, exist_ok=True)
CH = {k: v.float() for k, v in torch.load(SCALES).items()}
assert len(CH) == 128, f"expected 128 chosen scales, got {len(CH)}"

idx = json.load(open(SRC / "model.safetensors.index.json"))
wm = idx["weight_map"]
L = "model.language_model.layers"
layers = sorted({int(k.split(".")[3]) for k in wm if k.startswith(L + ".")})
assert len(layers) == 64

ops: dict[str, list] = {}


def add(n, kind, vec):
    ops.setdefault(n, []).append((kind, vec))


n_attn = n_gdn = 0
for i in layers:
    lk = f"{L}.{i}"
    is_attn = f"{lk}.self_attn.q_proj.weight" in wm
    # --- C ---
    s_c = CH[f"{lk}.mlp.down_proj"]
    add(f"{lk}.mlp.up_proj.weight", "rowdiv", s_c)
    add(f"{lk}.mlp.down_proj.weight", "colmul", s_c)
    if is_attn:
        n_attn += 1
        t = CH[f"{lk}.self_attn.o_proj"]
        assert t.shape == (N_KV, HD), f"o_proj scale shape {tuple(t.shape)}"
        add(f"{lk}.self_attn.v_proj.weight", "rowdiv", t.reshape(-1))
        add(f"{lk}.self_attn.o_proj.weight", "colmul",
            t.unsqueeze(1).expand(N_KV, G, HD).reshape(-1))
    else:
        n_gdn += 1
        s_d = CH[f"{lk}.linear_attn.out_proj"]
        assert s_d.numel() == VHD
        add(f"{lk}.linear_attn.norm.weight", "norm_plain", s_d)
        add(f"{lk}.linear_attn.out_proj.weight", "colmul", s_d.repeat(6144 // VHD))
assert n_attn == 16 and n_gdn == 48
print(f"plan: {len(ops)} tensors ({n_attn} attn, {n_gdn} gdn), {time.time()-t0:.0f}s",
      flush=True)

by_shard: dict[str, list] = {}
for tname, shard in wm.items():
    by_shard.setdefault(shard, []).append(tname)

applied = 0
for shard, names in sorted(by_shard.items()):
    tensors = load_file(str(SRC / shard))
    changed = 0
    for n in names:
        if n in ops:
            x = tensors[n].float()
            for kind, vec in ops[n]:
                if kind == "norm_plain":
                    x = x / vec
                elif kind == "colmul":
                    x = x * vec.unsqueeze(0)
                elif kind == "rowdiv":
                    x = x / vec.unsqueeze(1)
            tensors[n] = x.to(OUT_DTYPE)
            changed += 1
    applied += changed
    save_file(tensors, str(DST / shard), metadata={"format": "pt"})
    print(f"  {shard}: {changed} transformed", flush=True)
    del tensors
assert applied == len(ops), f"applied {applied} != planned {len(ops)}"

for f in SRC.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors"):
        shutil.copy2(f, DST / f.name)
print(f"TOTAL {time.time()-t0:.0f}s -> {DST}", flush=True)
