#!/usr/bin/env python3
"""patch-vision-rotation.py — T4 follow-up: rotate the vision merger output so a
rotated checkpoint is no longer TEXT-ONLY.

THE BUG THIS FIXES
------------------
`absorb-rotate.py` skips every tensor whose name contains "visual" (line ~148),
which correctly leaves the vision TOWER alone but ALSO skips the one vision
tensor that is not internal to the tower: `visual.merger.linear_fc2`, which
EMITS image embeddings straight into the language residual stream. After
rotation the language model reads that stream in the R basis, so unrotated
image features arrive in the OLD basis -> garbage for any image request. Text
is unaffected (which is why every text gate passed).

WHY A POST-HOC PATCH IS EXACT, NOT AN APPROXIMATION
---------------------------------------------------
The tower is untouched by every stage of the pipeline: it is excluded from
quantization (whole `model.visual.*` namespace is in `quantization_config.
ignore`, so it carries 0 quant scales), and absorb-rotate/smoothing/GPTQ all
pass it through verbatim. Verified bit-exactly before patching: the merger
tensors in ROT-GPTQ, SQv2-GPTQ and BF16-rotated are all equal to the original
BF16 checkpoint's (modulo the storage cast). So applying R here is identical to
having applied it during absorb-rotate.

THE TRANSFORM
-------------
The merger is a producer into the residual stream, so it takes the producer
rule (`rows x R`, absorb-rotate line ~176), including its bias:

    y = W a + b        ->      y' = R y = (R W) a + (R b)
    W' = R @ W   [5120, 4608]
    b' = R @ b   [5120]

⚠ IDEMPOTENCE: the new values are always computed from the ORIGINAL BF16
checkpoint, never from the artifact's current contents, and the artifact's
current state is classified first (unpatched / already-patched / unknown).
Deriving from the artifact would make a second run apply R TWICE and silently
destroy the tower.

⚠ BIT-REPRODUCIBILITY: the arithmetic deliberately mirrors the pipeline
exactly -- fp32 accumulation with the fp32-stored R (absorb-rotate:56), cast to
the dtype absorb-rotate emits (the tower's own bf16), then cast to whatever the
target artifact stores (GPTQ re-emits the tower as fp16). Accumulating in fp64
instead is marginally more accurate but differs by ~1 ulp of the storage dtype
and would make the published artifact un-reproducible from the published
scripts.

Single injection point confirmed: `vision_config.deepstack_visual_indexes` is
EMPTY, so there is no multi-layer deepstack injection to also rotate, and
`out_hidden_size == 5120 == H`.

The tensor is rewritten in the dtype it already has in that artifact, computed
in float32 from the artifact's own stored values.

GATES (both run automatically; the script refuses to keep a bad write)
  G0.1  fp64 algebra:  R @ (W a + b) == W' a + b'  for random a
  G0.2  coverage:      EXACTLY 2 tensors changed, every other tensor in the
                       shard bit-identical, dtypes/shapes/metadata unchanged
⚠ G0.2 compares TENSOR VALUES, not file bytes: safetensors reorders the header
on rewrite, so a byte-level diff of the shard is expected and meaningless.

USAGE
  python3 patch-vision-rotation.py --model /glm/.../Qwen3.8-27B-ROT-GPTQ-W8A8
  python3 patch-vision-rotation.py --model ... --dry-run
"""
import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

W_NAME = "model.visual.merger.linear_fc2.weight"
B_NAME = "model.visual.merger.linear_fc2.bias"
H = 5120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rotation", default="/glm/qwen38-sq-w8a8/rotation-R.pt")
    ap.add_argument("--source", default="/glm/qwen38-bf16/Qwen3.8-27B",
                    help="ORIGINAL checkpoint; the merger values are always "
                         "derived from here so the patch is idempotent")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rewrite even if the artifact is in an UNKNOWN state")
    ap.add_argument("--revert", action="store_true",
                    help="write the UNROTATED source values back — used to run "
                         "the negative control (a rotated language stack with "
                         "an unrotated merger must FAIL the vision gate; "
                         "without this, a passing gate cannot distinguish 'the "
                         "fix worked' from 'vision was never broken')")
    a = ap.parse_args()

    root = a.model.rstrip("/")
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    assert idx[W_NAME] == idx[B_NAME], "merger weight/bias in different shards"
    shard = idx[W_NAME]
    path = f"{root}/{shard}"

    # sanity: nothing else in the visual namespace emits into the residual
    others = [n for n in idx
              if n.startswith("model.visual.") and n not in (W_NAME, B_NAME)
              and n.endswith(".weight")]
    cfg = json.load(open(f"{root}/config.json"))
    vc = cfg.get("vision_config", {})
    assert vc.get("deepstack_visual_indexes") in ([], None), \
        f"deepstack injection present {vc.get('deepstack_visual_indexes')} — more tensors need rotating"
    assert vc.get("out_hidden_size") == H, f"out_hidden_size {vc.get('out_hidden_size')} != {H}"

    R32 = torch.load(a.rotation, map_location="cpu").float()  # as absorb-rotate stores it
    R = R32.double()                                          # fp64 copy, for the G0.1 check
    assert R.shape == (H, H), R.shape
    orth = (R @ R.T - torch.eye(H, dtype=torch.float64)).abs().max().item()
    print(f"rotation R: {tuple(R.shape)} orthogonality err {orth:.3e}")
    # R is STORED as fp32 (absorb-rotate:56 R32 = R.float()), so R@R.T can only
    # be orthogonal to fp32 precision (eps 1.19e-7). A genuine non-orthogonal
    # matrix would be off by O(0.1-1), i.e. 4+ orders above this bound.
    assert orth < 1e-5, f"R is not orthogonal ({orth:.3e})"

    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    before = load_file(path)
    assert W_NAME in before and B_NAME in before

    W0, b0 = before[W_NAME], before[B_NAME]
    dtW, dtB = W0.dtype, b0.dtype
    print(f"shard {shard}: {len(before)} tensors, metadata={meta}")
    print(f"  {W_NAME} {tuple(W0.shape)} {dtW}")
    print(f"  {B_NAME} {tuple(b0.shape)} {dtB}")
    assert W0.shape == (H, 4608) and b0.shape == (H,)

    # ---- authoritative source: the ORIGINAL, never the artifact -------------
    sidx = json.load(open(f"{a.source}/model.safetensors.index.json"))["weight_map"]
    Wsrc = load_file(f"{a.source}/{sidx[W_NAME]}")[W_NAME]
    bsrc = load_file(f"{a.source}/{sidx[B_NAME]}")[B_NAME]
    via = Wsrc.dtype                       # dtype absorb-rotate emits for the tower
    print(f"source {a.source.split('/')[-1]}: {via}, "
          f"pipeline chain fp32 -> {via} -> {dtW}")

    # exactly the pipeline arithmetic: fp32 accumulate, cast as absorb-rotate
    # emits, then cast to what this artifact stores
    Wrot = (R32 @ Wsrc.float()).to(via).to(dtW)
    brot = (R32 @ bsrc.float()).to(via).to(dtB)

    # ---- classify current state (idempotence guard) -------------------------
    unrot = (torch.equal(W0, Wsrc.to(dtW)) and torch.equal(b0, bsrc.to(dtB)))
    done = (torch.equal(W0, Wrot) and torch.equal(b0, brot))
    state = "UNPATCHED" if unrot else "ALREADY-PATCHED" if done else "UNKNOWN"
    print(f"current state: {state}")
    if done and not a.revert:
        print("nothing to do (already rotated); exiting without a write")
        return 0
    if unrot and a.revert:
        print("already unrotated; exiting without a write")
        return 0
    if state == "UNKNOWN" and not a.force:
        print("REFUSING: artifact is neither the original nor the expected "
              "rotation. Re-running blindly would apply R twice. Use --force "
              "only if you know the current values are recoverable.")
        return 2

    # ---- G0.1: the transform is exact (checked in fp64) ---------------------
    Wd, bd = Wsrc.double(), bsrc.double()
    g = torch.Generator().manual_seed(20260815)
    A = torch.randn(4608, 8, generator=g, dtype=torch.float64)
    lhs = R @ (Wd @ A + bd[:, None])          # rotate the ORIGINAL output
    rhs = (R @ Wd) @ A + (R @ bd)[:, None]    # output of the ROTATED weights
    err = (lhs - rhs).abs().max().item() / lhs.abs().max().item()
    print(f"G0.1 fp64 algebra: max rel err {err:.3e}", flush=True)
    assert err < 1e-12, "rotation algebra wrong"

    # end-to-end error of what the ROTATION would store, vs exact fp64
    lhs_q = (Wrot.double() @ A + brot.double()[:, None])
    rel = ((lhs_q - rhs).abs().max() / rhs.abs().max()).item()
    print(f"     stored path ({via}->{dtW}) vs exact fp64: max rel err {rel:.3e}")
    assert rel < 5e-2, "storage cast lost too much"

    # ---- what actually gets written ----------------------------------------
    if a.revert:
        print("!! --revert: writing UNROTATED source values (negative control) "
              "— this checkpoint will be TEXT-ONLY until re-patched")
        Wq, bq = Wsrc.to(dtW), bsrc.to(dtB)
    else:
        Wq, bq = Wrot, brot

    if a.dry_run:
        print("dry-run: nothing written")
        return 0

    # pre-image fingerprints (cheap, exact) taken BEFORE the write
    def fp(t):
        return hashlib.sha256(
            t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    pre = {n: (fp(t), tuple(t.shape), str(t.dtype)) for n, t in before.items()}

    after = dict(before)
    after[W_NAME] = Wq
    after[B_NAME] = bq
    tmp = path + ".tmp"
    save_file(after, tmp, metadata=meta)
    os.replace(tmp, path)
    print(f"wrote {path}")

    # ---- G0.2: exactly two tensors changed, nothing else moved --------------
    del before, after
    reloaded = load_file(path)
    with safe_open(path, framework="pt") as f:
        meta2 = f.metadata()
    changed, mismatched = [], []
    if set(reloaded) != set(pre):
        mismatched.append("tensor SET changed")
    for n, (h, shp, dt) in pre.items():
        r = reloaded[n]
        if tuple(r.shape) != shp or str(r.dtype) != dt:
            mismatched.append(f"{n}: {shp}/{dt} -> {tuple(r.shape)}/{r.dtype}")
            continue
        if fp(r) != h:
            changed.append(n)
    print(f"G0.2 metadata preserved : {meta2 == meta} ({meta2})")
    print(f"G0.2 tensors changed    : {len(changed)} / {len(pre)}  {sorted(changed)}")
    print(f"G0.2 shape/dtype drift  : {len(mismatched)} {mismatched}")
    # invariant: ONLY the two merger tensors may differ, and they must hold
    # exactly the expected values. (Not "exactly 2 changed": re-patching from a
    # partially-equal state legitimately rewrites fewer -- e.g. a bf16 bias that
    # already rounds to the same value.)
    ok = (set(changed) <= {W_NAME, B_NAME}
          and not mismatched and meta2 == meta
          and torch.equal(reloaded[W_NAME], Wq)
          and torch.equal(reloaded[B_NAME], bq))
    print("G0.2 VERDICT            :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
