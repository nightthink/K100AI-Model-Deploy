#!/usr/bin/env python3
"""build-gptq-38.py — T1: GPTQ (learned rounding via Hessian-based error
compensation) instead of RTN, on the T2-smoothed checkpoint. Same W8A8
scheme, same ignore list. Calibration = the T2c mixed corpus.

⚠ MUST run with pipeline="sequential": GPTQ Hessians are C_in^2 fp32 per
linear (down_proj alone = 1.2 GB); the basic pipeline keeps all of them
alive. Sequential processes and frees layer-by-layer.
⚠ transformers must stay 5.13.1 (save path, TODO.md issue 3).
"""
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM  # transformers >= 5
    LOADER = AutoModelForMultimodalLM
except ImportError:
    LOADER = AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
MODEL = "/glm/qwen38-sq-w8a8/Qwen3.8-27B-ROT-smoothed"
OUT = "/glm/qwen38-sq-w8a8/Qwen3.8-27B-ROT-GPTQ-W8A8"
CORPUS = "/glm/qwen38-sq-w8a8/calib-mixed.txt"
N_SAMPLES = 128
SEQ_LEN = 2048

IGNORE = [
    "lm_head",
    "re:.*embed_tokens",
    "re:.*linear_attn[.]in_proj_a$",
    "re:.*linear_attn[.]in_proj_b$",
    "re:.*visual.*",
    "re:.*mtp.*",
]

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
text = open(CORPUS, errors="replace").read()
ids = tok(text, add_special_tokens=False).input_ids
wins = [ids[s:s + SEQ_LEN] for s in range(0, len(ids) - SEQ_LEN, SEQ_LEN)][:N_SAMPLES]
assert len(wins) == N_SAMPLES, f"corpus too small: {len(wins)}"
ds = Dataset.from_dict({"input_ids": wins, "attention_mask": [[1] * SEQ_LEN] * len(wins)})
print(f"calibration: {len(wins)} x {SEQ_LEN} from mixed corpus", flush=True)

mm = {i: "16GiB" for i in range(torch.cuda.device_count())}
mm["cpu"] = "8GiB"
model = LOADER.from_pretrained(
    MODEL, dtype="auto", device_map="auto", max_memory=mm,
    offload_folder="/glm/qwen38-sq-w8a8/offload-gptq-rot", offload_state_dict=True,
)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

oneshot(
    model=model,
    dataset=ds,
    recipe=[GPTQModifier(targets=["Linear"], scheme="W8A8", ignore=IGNORE)],
    max_seq_length=SEQ_LEN,
    num_calibration_samples=N_SAMPLES,
    # ⚠ do NOT set sequential_targets=["Linear"] — llm-compressor 0.13.0's
    # per-linear AST autowrap crashes (KeyError 'forward', ast_helpers.py:86).
    # Whole-layer default works; OOM at 18GiB was survived at 16GiB budget +
    # expandable_segments (this container does no custom-AR IPC, so ES is safe
    # HERE — never in the serving container).
    pipeline="sequential",
)
print(f"quantization done at {time.time()-t0:.0f}s; saving...", flush=True)

for kw in (dict(max_shard_size="2GB"), dict(max_shard_size="1GB")):
    try:
        print(f"save_pretrained({kw}) ...", flush=True)
        model.save_pretrained(OUT, **kw)
        print(f"SAVED with {kw}", flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"save failed with {kw}: {type(e).__name__}: {e}", flush=True)
else:
    raise SystemExit("ALL save attempts failed")

tok.save_pretrained(OUT)
print(f"TOTAL {time.time()-t0:.0f}s", flush=True)
