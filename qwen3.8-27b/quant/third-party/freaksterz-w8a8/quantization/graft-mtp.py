#!/usr/bin/env python3
"""graft-mtp.py <quantized_ckpt_dir> [<bf16_source_dir>] — re-attach the 15
BF16 mtp.* tensors. Source defaults to the 3.6 BF16 checkpoint.

transformers' modeling class does not register the MTP head, so
save_pretrained silently DROPS mtp.* . vLLM's Qwen3_5MTP drafter needs them,
and they must stay BF16 (a quantized MTP head loads clean and then rejects
every draft -- acceptance length 1.00; this is why config.json must also carry
`re:.*mtp.*` in quantization_config.ignore).

!! THE SOURCE MUST MATCH THE TRANSFORM LINEAGE !!
Pass the BF16 checkpoint that went through the SAME transforms as the target.
For this repo that is the ROTATED BF16 intermediate, NOT the original: grafting
the original drafter onto a rotated model feeds it rotated activations with
unrotated weights, and acceptance collapses to ~1.00 while every target-side
gate (warmup, smoke, tool calls, KLD) still passes. Acceptance length is the
only gate that detects it.
Writes model-mtp.safetensors + patches the index. Idempotent.
"""
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path(sys.argv[2] if len(sys.argv) > 2 else "/glm/qwen36-bf16/Qwen3.6-27B")
DST = Path(sys.argv[1])

src_idx = json.load(open(SRC / "model.safetensors.index.json"))
mtp_names = [n for n in src_idx["weight_map"] if "mtp" in n]
assert len(mtp_names) == 15, f"expected 15 mtp tensors in source, got {len(mtp_names)}"

dst_idx_path = DST / "model.safetensors.index.json"
dst_idx = json.load(open(dst_idx_path))
already = [n for n in dst_idx["weight_map"] if "mtp" in n]
if len(already) == 15 and (DST / "model-mtp.safetensors").exists():
    print("already grafted, nothing to do")
    sys.exit(0)

tensors = {}
for n in mtp_names:
    with safe_open(SRC / src_idx["weight_map"][n], framework="pt") as f:
        tensors[n] = f.get_tensor(n)
save_file(tensors, str(DST / "model-mtp.safetensors"), metadata={"format": "pt"})

for n in mtp_names:
    dst_idx["weight_map"][n] = "model-mtp.safetensors"
dst_idx["metadata"]["total_size"] = dst_idx["metadata"].get("total_size", 0) + sum(
    t.numel() * t.element_size() for t in tensors.values()
)
json.dump(dst_idx, open(dst_idx_path, "w"), indent=2)
print(f"grafted {len(tensors)} mtp tensors into {DST}")
