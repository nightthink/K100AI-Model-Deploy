#!/usr/bin/env python3
"""capture-actmax-v2.py — T2 capture: per-WINDOW channel maxima (for robust
percentile scales) + sampled activation ROWS (for the offline alpha proxy),
all 4 GEMM input classes, Qwen3.8-27B.

Outputs:
  actmax-v2-winmax.pt   {hook: [n_windows, C] fp32}  per-window abs maxima
  actsample-w<i>.pt     {hook: [rows_per_win, C] fp16}  raw input rows,
                        every SAMPLE_EVERYth window (written incrementally --
                        30 GB host RAM cannot hold the full sample set)
Corpus: mixed wiki + code + tool-JSON (T2c) built by make-corpus.py.
"""
import time

import torch
from transformers import AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM as LOADER
except ImportError:
    from transformers import AutoModelForCausalLM as LOADER

MODEL = "/glm/qwen38-sq-w8a8/Qwen3.8-27B-BF16-rotated"
CORPUS = "/glm/qwen38-sq-w8a8/calib-mixed.txt"
OUTDIR = "/glm/qwen38-sq-w8a8/rot"
N_WINDOWS = 96
CTX = 2048
SAMPLE_EVERY = 8          # windows 0,8,16,... -> 12 sample files
ROWS_PER_WIN = 176        # 12 x 176 = 2112 rows per hook for the proxy

# ROTATED MODEL: classes A/B are gone (their fold targets — the zero-centered
# norms — are absorbed to weightless, and per-channel residual scaling is
# impossible). Rotation replaces them. Only C/D/E survive.
SUFFIXES = (
    "mlp.down_proj",            # C
    "self_attn.o_proj",         # E
    "linear_attn.out_proj",     # D
)

tok = AutoTokenizer.from_pretrained(MODEL)
text = open(CORPUS, errors="replace").read()
ids = tok(text, add_special_tokens=False).input_ids
wins = [ids[s:s + CTX] for s in range(0, len(ids) - CTX, CTX)][:N_WINDOWS]
assert len(wins) == N_WINDOWS, f"corpus too small: {len(wins)} windows"
print(f"{len(wins)} windows of {CTX}", flush=True)

t0 = time.time()
mm = {i: "18GiB" for i in range(torch.cuda.device_count())}
mm["cpu"] = "8GiB"
model = LOADER.from_pretrained(
    MODEL, dtype="auto", device_map="auto", max_memory=mm,
    offload_folder=f"{OUTDIR}/offload-v2", offload_state_dict=True,
)
model.eval()
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

winmax: dict[str, list] = {}
cur_max: dict[str, torch.Tensor] = {}
cur_sample: dict[str, torch.Tensor] = {}
sampling = False
g = torch.Generator().manual_seed(0)

hooks = []
for name, mod in model.named_modules():
    if "mtp" in name:
        continue
    if any(name.endswith(sfx) for sfx in SUFFIXES):
        def make_hook(n):
            def hook(module, args, output):
                x = args[0].detach()
                flat = x.reshape(-1, x.shape[-1])
                cur_max[n] = flat.abs().amax(0).float().cpu()
                if sampling:
                    idx = torch.randperm(flat.shape[0], generator=g)[:ROWS_PER_WIN]
                    cur_sample[n] = flat[idx.to(flat.device)].to(torch.float16).cpu()
            return hook
        hooks.append(mod.register_forward_hook(make_hook(name)))
assert len(hooks) == 128, f"expected 128 hooks (64+16+48), got {len(hooks)}"
print(f"{len(hooks)} hooks registered", flush=True)

with torch.no_grad():
    for wi, w in enumerate(wins):
        sampling = (wi % SAMPLE_EVERY == 0)
        cur_sample = {}
        model(input_ids=torch.tensor([w]).to(model.device), use_cache=False)
        for n, m in cur_max.items():
            winmax.setdefault(n, []).append(m)
        if sampling:
            torch.save(cur_sample, f"{OUTDIR}/actsample-w{wi}.pt")
        if (wi + 1) % 8 == 0:
            print(f"  {wi+1}/{len(wins)}  ({time.time()-t0:.0f}s)", flush=True)

for h in hooks:
    h.remove()

stacked = {k: torch.stack(v) for k, v in winmax.items()}
torch.save(stacked, f"{OUTDIR}/actmax-v2-winmax.pt")
tot = sum(v.shape[1] for v in stacked.values())
print(f"saved winmax: {len(stacked)} hooks x {N_WINDOWS} windows, {tot} channels",
      flush=True)
print(f"TOTAL {time.time()-t0:.0f}s", flush=True)
