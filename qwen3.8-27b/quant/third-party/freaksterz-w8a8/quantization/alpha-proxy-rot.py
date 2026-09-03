#!/usr/bin/env python3
"""alpha-proxy.py — T2b: pick per-hook (scale-base, alpha) OFFLINE by
simulating the fold + INT8 quantization on captured activation rows, without
any build/KLD cycle per candidate.

For each hook and each candidate s(base, alpha):
    X' = X / s        (per-token dynamic INT8, symmetric, amax/127)
    W' = W * s        (per-out-channel INT8, symmetric)
    err = ||q(X') @ q(W').T - X @ W.T||_F^2 / ||X @ W.T||_F^2
Candidates: base in {absmax, median-of-window-maxima} x alpha in
{0.5, 0.65, 0.8, 0.9}. The (absmax, 0.8) cell reproduces the shipped v1
recipe = the reference. Constrained classes handled exactly like the fold:
D reduced to [128] over value heads, E group-reduced to [KV, HD].

Output: scales-v2.pt {hook: chosen s tensor (constrained form for D/E)}
        + per-class report of chosen cells and predicted err vs reference.
"""
import glob
import json
import time
from collections import defaultdict

import torch
from safetensors import safe_open

SRC = "/glm/qwen38-sq-w8a8/Qwen3.8-27B-BF16-rotated"
D8 = "/glm/qwen38-sq-w8a8/rot"
DEV = "cuda:0"
ALPHAS = [0.5, 0.65, 0.8, 0.9]
S_MIN, S_MAX = 1e-5, 16.0
N_Q, N_KV, HD = 24, 4, 256
G = N_Q // N_KV
VHD = 128

t0 = time.time()
winmax = torch.load(f"{D8}/actmax-v2-winmax.pt")
samples: dict[str, list] = defaultdict(list)
for f in sorted(glob.glob(f"{D8}/actsample-w*.pt")):
    for k, v in torch.load(f).items():
        samples[k].append(v)
X_all = {k: torch.cat(v).float() for k, v in samples.items()}
print(f"loaded {len(winmax)} hooks, sample rows "
      f"{next(iter(X_all.values())).shape[0]} ({time.time()-t0:.0f}s)", flush=True)

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]


def get_w(names):
    out = []
    for n in names:
        with safe_open(f"{SRC}/{idx[n]}", framework="pt") as f:
            out.append(f.get_tensor(n).float())
    return torch.cat(out, 0)


def q_rows(t):  # symmetric int8 along dim -1 per row
    s = t.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
    return (t / s).round().clamp(-127, 127) * s


def err_for(X, W, s):
    Xf, Wf = X / s, W * s
    ref = X @ W.T
    got = q_rows(Xf) @ q_rows(Wf).T
    return ((got - ref).pow(2).sum() / ref.pow(2).sum()).item()


def calc_s(a, wmax, alpha):
    s = a.pow(alpha) / wmax.clamp(min=1e-8).pow(1 - alpha)
    return torch.where(a > 0, s.clamp(min=S_MIN, max=S_MAX), torch.ones_like(s))


layers = sorted({k.rsplit(".mlp.down_proj", 1)[0] for k in winmax if k.endswith("mlp.down_proj")})
assert len(layers) == 64

chosen: dict[str, torch.Tensor] = {}
stats = defaultdict(lambda: defaultdict(int))
gains = defaultdict(list)

for li, lk in enumerate(layers):
    is_attn = f"{lk}.self_attn.q_proj.weight" in idx
    # ROTATED MODEL: only C/D/E remain (A/B fold targets no longer exist).
    jobs = [(f"{lk}.mlp.down_proj", [f"{lk}.mlp.down_proj.weight"], None)]
    if is_attn:
        jobs.append((f"{lk}.self_attn.o_proj", [f"{lk}.self_attn.o_proj.weight"], "gqa"))
    else:
        jobs.append((f"{lk}.linear_attn.out_proj", [f"{lk}.linear_attn.out_proj.weight"], "vhd"))

    for hook, wnames, constraint in jobs:
        W = get_w(wnames).to(DEV)
        X = X_all[hook].to(DEV)
        wm = winmax[hook].to(DEV)                      # [96, C]
        wmax = W.abs().amax(0)
        cls = hook.split(".")[-1]
        best = (None, None, float("inf"))
        ref_err = None
        for base_name, a_full in (("absmax", wm.amax(0)), ("median", wm.median(0).values)):
            for alpha in ALPHAS:
                if constraint == "vhd":
                    a_c = a_full.view(-1, VHD).amax(0)
                    w_c = wmax.view(-1, VHD).amax(0)
                    s_c = calc_s(a_c, w_c, alpha)
                    s = s_c.repeat(a_full.numel() // VHD)
                elif constraint == "gqa":
                    a_c = a_full.view(N_KV, G, HD).amax(1)
                    w_c = wmax.view(N_KV, G, HD).amax(1)
                    s_c = calc_s(a_c, w_c, alpha)
                    s = s_c.unsqueeze(1).expand(N_KV, G, HD).reshape(-1)
                else:
                    s_c = s = calc_s(a_full, wmax, alpha)
                e = err_for(X, W, s)
                if base_name == "absmax" and alpha == 0.8:
                    ref_err = e
                if e < best[2]:
                    best = (base_name, alpha, e, s_c.cpu())
        chosen[hook] = best[3]
        stats[cls][f"{best[0]}/a{best[1]}"] += 1
        gains[cls].append((ref_err - best[2]) / max(ref_err, 1e-12))
        del W, X
    if (li + 1) % 16 == 0:
        print(f"  {li+1}/64 layers ({time.time()-t0:.0f}s)", flush=True)

torch.save(chosen, f"{D8}/scales-rot.pt")
print("\nchosen cells per class:", flush=True)
for cls, d in stats.items():
    g = torch.tensor(gains[cls])
    print(f"  {cls:<14} {dict(d)}  proxy-err reduction vs (absmax,0.8): "
          f"mean {100*g.mean():.1f}% max {100*g.max():.1f}%", flush=True)
print(f"saved scales-rot.pt ({len(chosen)} hooks). TOTAL {time.time()-t0:.0f}s", flush=True)
