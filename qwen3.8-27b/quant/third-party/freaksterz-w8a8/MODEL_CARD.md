---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- vllm
- qwen3_5
- qwen3.8
- compressed-tensors
- w8a8
- int8
- smoothquant
- gptq
- rotation
- quarot
- quantized
- mtp
- speculative-decoding
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — Rotation + SmoothQuant + GPTQ W8A8-INT8 (+ BF16 MTP)

**The lowest-divergence INT8 W8A8 of this model that we can measure: mean
full-vocab KLD vs BF16 = 0.0110 at top-K 512 (~0.0081 floor-free — see
[Measurement caveats](#measurement-caveats-read-before-comparing)).** Against
the only other INT8 W8A8 of Qwen3.8-27B, re-measured by us on an identical
harness, that is **1.92× lower divergence while quantizing 51% more modules**
— and it keeps the **native INT8 tensor-core path**
(`CutlassInt8ScaledMMLinearKernel`), ~3.5K tok/s prefill against ~2000 for
weight-only (Marlin) checkpoints.

Numerics only, no fine-tuning. All credit for the model to Qwen.

## Why this exists

On SM86 (RTX 3090 class) there are exactly two viable 8-bit families:

| family | kernel path | prefill (2×3090 TP2) | measured KLD vs BF16 |
|---|---|---|---|
| W8A16 / FP8 weight-only | Marlin dequant | ~2000 tok/s | 0.002–0.006 |
| W8A8 dynamic INT8 | **CUTLASS native INT8** | **~3500 tok/s** | 0.021 (plain recipe) → **0.011 (this)** |

The gap between those two rows is almost entirely the **dynamic per-token INT8
activation cost**, and it is the price of the faster kernel. Weight-side
choices barely move it: observer and layer-exclusion variants measured worth
~zero (two independent public W8A8 quants of the 3.6 sibling land within 0.0005
of each other). So this checkpoint attacks the activation term directly, with
three stacked mechanisms:

1. **Offline rotation** (QuaRot-style): one 5120² orthogonal `R` applied to the
   residual stream so per-token activation vectors become near-isotropic and
   outliers stop owning the per-token scale. Fully offline — the served
   checkpoint is a plain `compressed-tensors` model with no runtime support.
2. **SmoothQuant** on the GEMM classes rotation does not reach.
3. **GPTQ** learned rounding on the weight side.

## Recipe

**Stage 1 — norm absorption + rotation.** Weightless RMSNorm is
rotation-equivariant, so every zero-centered norm is first absorbed into its
consumers' input columns and set to zero; then consumers take `W @ Rᵀ` and
producers take `R @ W`. Embeddings and `lm_head` are rotated as residual
vectors. The vision merger's output projection is rotated too (it emits into
the residual stream). Verified with an fp64 whole-graph check.

**Stage 2 — SmoothQuant on the remaining classes.** After rotation the
attention/FFN *input* folds are structurally unnecessary (their target norms
are absorbed to weightless), so only the classes that do not touch the residual
stream remain:

| fold | coverage | mechanism (exact pre-quantization) |
|---|---|---|
| C down-in | 64 layers | `up_proj` rows ÷ s → `down_proj` columns (SiLU product is elementwise) |
| D gdn-out | 48 layers | `linear_attn.norm` (gated, per-head-dim) → `out_proj` columns |
| E attn-out | 16 layers | `v_proj` rows ÷ t → `o_proj` columns, GQA-constrained (shared scale per KV group) |

Scales come from per-**window** channel maxima (not a global abs-max), and the
per-hook α is chosen **offline** by simulating fold+INT8 on captured activation
rows. Median-of-window-maxima beat abs-max on ~75% of hooks: a single spike
token was otherwise owning each channel's scale. On the rotated model the
chosen α collapses to 0.5 (neutral) — independent confirmation that rotation
had already flattened the activations.

**Stage 3 — quantization.** [llm-compressor](https://github.com/vllm-project/llm-compressor)
`GPTQModifier`, scheme **W8A8**: INT8 weights per-channel symmetric with
learned rounding, **INT8 dynamic per-token activations** → `compressed-tensors`
`int-quantized`. Calibration is a mixed wiki/code/tool-JSON corpus.

**Preserved BF16:** `mtp.*` (the drafter — quantizing it silently destroys
speculative decoding), `lm_head`, `embed_tokens`, the vision tower, GDN
`in_proj_a`/`in_proj_b` recurrent gates, all norms.

The MTP head is grafted back post-export (`model-mtp.safetensors`, 15 tensors)
because transformers' save path drops modules the modeling class never
registers, and its ignore-list serialization drops the `mtp` pattern. The
shipped `config.json` carries the corrected ignore list.

### Reproduction notes (each of these cost us a broken build)

1. **`Qwen3_5RMSNorm` is zero-centered** (`out = x̂ · (1 + w)`). The textbook
   SmoothQuant fold `w/s` silently produces garbage; the correct fold is
   `w' = (1+w)/s − 1`. (`Qwen3_5RMSNormGated` in the GDN blocks is plain
   `x̂ · w`, where `w/s` IS correct. Check the norm class per fold.)
2. Storing that fold in bf16 destroys precision near gain ≈ 0 (up to 14%
   per-channel error at high s) — store **float16**, cap **s ≤ 16**.
3. **The MTP graft source must match the transform lineage.** Grafting the
   original (unrotated) drafter onto a rotated model gives the drafter rotated
   inputs with unrotated weights: acceptance collapses to ~1.0 while every
   target-side gate still passes. Acceptance length is the only gate that sees
   this.

## Measured quality

Full-vocab KLD vs a teacher-forced BF16 reference, 122,640 positions
(240×512-token WikiText-2 windows), top-K 512.

**Every row below was measured by us, on the same reference, the same corpus,
the same harness and the same server settings** — including the competing
checkpoints, which were downloaded and re-measured rather than quoted. Figures
those projects publish themselves are listed separately further down, because
they are not on this scale.

| Qwen3.8-27B checkpoint | activations | quantized Linears | mean KLD | median | top-1 | P99.9 | max | PPL ln |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `lokeshe09/Qwen3.8-27B-INT8` | **INT8 dynamic** | 264 (no GDN) | 0.02110 | 0.0095 | 94.00% | 0.600 | 14.6 | +0.0079 |
| our v1 (SmoothQuant + RTN) | **INT8 dynamic** | 400 | 0.01414 | 0.0058 | 95.32% | 0.44 | 17.2 | +0.0069 |
| our v2 (robust scales + GPTQ) | **INT8 dynamic** | 400 | 0.01231 | 0.0050 | 95.67% | 0.40 | 3.32 | +0.0051 |
| **this checkpoint (+ rotation)** | **INT8 dynamic** | **400** | **0.01098** | **0.0044** | **95.90%** | **0.33** | 7.87 | **+0.0040** |
| `Qwen/Qwen3.8-27B-FP8` (official) | none on SM86 † | 407 | 0.00584 | 0.0027 | 96.85% | 0.145 | 10.2 | +0.0021 |
| `lued/…-INT8-W8A16-MTP` | none (weight-only) | 400 | 0.00212 | 0.0009 | 97.91% | 0.039 | 7.88 | −0.0001 |

† The official FP8 checkpoint is block-scaled DeepSeek-format FP8. On SM86 vLLM
selects `MarlinFP8ScaledMMLinearKernel` — a **weight-only dequant path with
BF16 activations**, so on this hardware it belongs to the W8A16 column, not to
the W8A8 one, and it runs at W8A16 speed. On SM89+ it executes natively and
this row would not transfer.

**Read the table by column, not by row.** The two groups are different
products:

* Among checkpoints that actually quantize activations (the native-INT8
  tensor-core path, ~3.5K tok/s prefill), this checkpoint is the best measured
  — **1.92× lower divergence than the only other INT8 W8A8 of this model**,
  while quantizing **51% more modules** (400 vs 264: `lokeshe09` leaves all 144
  GDN projections in BF16, which is also why its checkpoint is 33.9 GiB against
  our 30 GB). A floor-free paired test puts the gap at **+102% (t = +62.4)**,
  with this checkpoint better at 79.6% of all scored positions.
* Weight-only checkpoints (official FP8 on Ampere, `lued` W8A16) are 2–5×
  closer to BF16 and we do not claim otherwise — they buy that with the Marlin
  dequant path at roughly **2000 tok/s prefill against our ~3500**. That is the
  entire trade this quantization exists to make.

**A clean measurement of what activation quantization actually costs.**
`lued/…-INT8-W8A16-MTP` quantizes *exactly the same 400 Linear modules* this
checkpoint does — same 144 GDN projections, same 192 MLP, same 64
full-attention, same BF16 exclusions — and differs only in that its
activations stay at 16-bit. Measured on one harness against one reference:

| | weights | activations | quantized Linears | mean KLD |
|---|---|---|---:|---:|
| `lued` W8A16 | INT8 g128 | 16-bit | 400 | 0.00212 |
| **this checkpoint** | INT8 per-channel + GPTQ | **INT8 dynamic per-token** | 400 | 0.01098 |

So at identical weight coverage, dynamic per-token INT8 activations cost about
**0.0089 nats/token** on this architecture — and that is *after* rotation,
robust scales and learned rounding have already removed roughly half of it (a
plain recipe over fewer modules measures 0.0211). This is the number to beat
for anyone pursuing W8A8 on Qwen3.5-family models; it is not reachable by
weight-side work alone.

⚠ The `max` column is a single-position statistic and is not stable: this
checkpoint's 7.87 is worse than v2's 3.32 at K=512, but on the same windows at
K=4096 the ordering reverses (1.32 here vs 2.34 for v2). Do not read one
position as a quality signal in either direction. Every distributional
statistic — mean, median, top-1, P99.9, PPL — favours this checkpoint within
its class.

### Measurement caveats (read before comparing)

**These numbers are upper bounds, and they are not comparable to KLD figures
from other projects.** Both statements are measured, not hedging:

* **Top-K truncation inflates the number by ~21%.** Reference ids missing from
  the candidate's top-K get assigned a uniform floor, which under-estimates `q`
  and therefore over-estimates KL. Re-measuring the same windows at **K=4096**
  gives **0.00859** for this checkpoint (and 0.00944 for the unrotated v2), so
  the ranking is unchanged but the level drops. A floor-free statistic —
  restricting the sum to ids present in both arms' supports — gives **0.0081**
  and is *stable* across K (0.008090 at K=512 vs 0.008063 at K=4096), which
  identifies the floor as the entire source of K-sensitivity. **~0.0081 is the
  best estimate of this checkpoint's divergence.**
* **The ranking is not an artifact of that floor.** The floor's hit-rate tracks
  the ranking (5.27% here vs 5.59% / 6.08% for the weaker arms), so it was
  tested directly: on a paired comparison over ids present in *both* supports —
  identical positions, identical `p_i` — this checkpoint leads v2 by **+14.0%
  (paired t = +11.6)** and v1 by **+31.2% (t = +20.0)**. Removing the floor
  *widened* the lead.
* **KV-cache dtype is included.** Served with `fp8_e4m3` KV, as measured. With
  bf16 KV the same run gives 0.01064, so the KV term is only **0.00034** (3.1%
  of the total). Published KLD from other projects usually contains no KV term
  at all.
* **Cross-engine.** The reference is offline (transformers, BF16); the
  candidate is served through vLLM, because W8A8's activation quantization only
  exists inside the INT8 kernels — scoring it offline would measure the weight
  term alone and understate the divergence.
* Consequently these values are **not** on the same scale as llama.cpp
  `--kl-divergence` output (full-vocab, same-engine FP16-vs-quant). llama.cpp's
  own documentation warns against cross-project KLD comparison.

### Figures published by other Qwen3.8-27B quantizations

Listed for completeness, **not** merged into the table above, because they are
measured on different corpora, position counts and execution paths:

| source | their checkpoint | official Qwen FP8, same harness | protocol |
|---|---:|---:|---|
| `lued/…-INT8-W8A16-MTP` | 0.000894 | **0.004396** | 4,563 positions; 8 short prompts + a 4K probe; HF dequant path; no KV cache |
| `huginnfork/…-FP8` | 0.0362 wt-only / 0.0756 deployed | **0.0523 / 0.1001** | 8 samples of `neuralmagic/calibration` @ seq 1024 |
| this card | 0.01098 | **0.00584** | 122,640 positions; WikiText-2; served vLLM kernels; fp8 KV; top-K 512 |

**Look at the middle column.** Three independent measurements of the *same*
official Qwen FP8 checkpoint, all reported as KLD against the same BF16 parent,
span **0.0044 → 0.0058 → 0.0523 — a factor of 12**. Two of the three agree to
within 33% (both are weight-dominated paths); the third differs by an order of
magnitude on corpus and harness alone.

That is the concrete reason the tables are kept apart, and the reason we
re-measured the competing checkpoints ourselves instead of quoting them. **A
KLD number without its protocol is not a comparable quantity**, and a ranking
assembled from numbers published by different projects is not a ranking.

### Task-level evaluation

`lm-evaluation-harness` 0.4.12 against a served vLLM endpoint, 0-shot, chat
template applied, model's own sampling contract (temp 1.0 / top_p 0.95 /
top_k 20), single seed.

| benchmark | metric | score |
|---|---|---:|
| GSM8K-Platinum (n=1209) | exact_match, flexible-extract | **94.13% ± 0.68** |
| IFEval (n=541) | prompt-level strict | **91.13% ± 1.22** |
| IFEval | prompt-level loose | 93.72% ± 1.04 |
| IFEval | instruction-level strict / loose | 94.12% / 95.80% |

⚠ Two harness adaptations were required, and both are published alongside this
model rather than hidden: upstream `gsm8k_*_cot_zeroshot` sets
`until: ["Q:", …]`, which a reasoning model trips while restating the problem
inside its own reasoning — generation halts mid-thought and the answer is lost
(5/40 items, at *both* 4k and 16k token budgets). Upstream `ifeval` caps
generation at 1280 tokens, below a reasoning model's budget. Only the stop
condition, the budget and sampling were changed; datasets, prompts, metrics and
filters are untouched.

⚠ GSM8K `strict-match` scores **0.00%** for this model and is not reported: it
requires the literal phrasing "The answer is *n*.", which a chat model does not
produce. Of the 5.87% `flexible-extract` miss rate, **2.81 points are string
formatting** (`$26.00` vs `26`) rather than arithmetic — numeric-equivalence
scoring gives 96.94%. We publish the upstream metric and disclose the audit
rather than reword the prompt to farm the filter.

⚠ **No in-house "recovery %" is published.** Computing it honestly requires
serving the BF16 base generatively, which does not fit the hardware this was
built on (52 GB weights vs 48 GB VRAM). The absolute scores above are what we
can stand behind.

### Tool-calling fidelity

Tool-argument corruption is the failure mode that matters most for agentic use,
and it is stochastic and temperature-dependent, so it was tested at scale:
**320 single calls + 40 multi-turn chains, at both N=1 and N=6 concurrency, at
the model's own sampling contract** — 8 argument families including
commas-inside-strings, nested arrays of objects, 3-level nested objects,
escaped quotes/newlines, float precision, CJK/emoji, and 5-tool
disambiguation.

| concurrency | clean | structural failures | argument corruptions |
|---|---:|---:|---:|
| N=1 | 360/360 (100.00%) | 0 | **0** |
| N=6 | 359/360 (99.72%) | 0 | **0** |

**Zero argument corruptions in 720 calls.** The single N=6 miss was the model
answering in prose instead of calling a tool (`finish_reason: stop`), not a
malformed call.

### Vision

This checkpoint is vision-capable and verified: **8/8** on a synthetic image
gate (text + three coloured shapes), including correct spatial relations.

⚠ If you rotate a VLM yourself, note that the vision merger's output
projection emits into the *language* residual stream and must be rotated with
it. Skipping it produces a checkpoint that passes every text benchmark and
confidently hallucinates on every image — our unrotated-merger control scored
**0/8**, describing the test image as "the word 'Babylon' in dark, serif
lettering".

## Verified serving (vLLM, TP2 on 2×RTX 3090)

| check | result |
|---|---|
| kernel | `CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8` |
| KV pool | 373,445 tokens @ 7.13 GiB/GPU pin, fp8_e4m3 KV, 262,144 max len |
| MTP spec decode (K=3) | acceptance length 2.70 (N=1) / 2.27–3.05 (N=6) |
| tool calls (`qwen3_coder` parser) | see battery above |
| `reasoning_effort` | monotone (xhigh > medium > low); `enable_thinking=false` safe |

### Throughput

Measured on **this checkpoint** (not a sibling), TP2 on 2×RTX 3090, vLLM with
MTP K=3, fp8_e4m3 KV, `--max-num-batched-tokens 4096`, 4 reps + 2 warmup.

| workload | tok/s | CV |
|---|---:|---:|
| prefill, 2K prompt | **3500** | 0.07% |
| prefill, 8K prompt | **3585** | 0.18% |
| prefill, 16K prompt | **3438** | 0.06% |
| decode, N=1 narrative | 66.7 | 2.4% |
| decode, N=1 code | 105.5 | 3.8% |
| **aggregate, 6 concurrent, narrative** | **330.3** | — |
| **aggregate, 6 concurrent, code** | **440.9** | — |

The ~3.5K tok/s prefill is the point of this quantization: it is the native
INT8 tensor-core path, roughly **1.75×** what a weight-only INT8/FP8 checkpoint
reaches on the same hardware (~2000 tok/s), because those fall back to Marlin
dequant-into-FP16 GEMMs.

⚠ Cards were at a 420 W limit and drew ~348 W with **0.0%** of samples
power-capped, i.e. these numbers are not power-limited; at a 330 W cap prefill
is capped and lands ~2% lower. Quote a power cap with any throughput figure for
this class of GPU.

⚠ `num_speculative_tokens` **> 3 is not recommended** on hybrid-GDN models:
K=5 reproduced the known illegal-memory-access crash class under concurrent
load on 2×3090 (vLLM #37035 family) in our testing.

## Serve

```bash
vllm serve <this-repo> \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization compressed-tensors \
  --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Add `--language-model-only` for text-only serving (saves the vision tower's
memory). Sampling per the upstream card: temperature 1.0, top_p 0.95,
top_k 20. Reasoning effort via `reasoning_effort: xhigh|medium|low` — other
values are rejected by the chat template.

## Revisions

| revision | recipe | mean KLD (K=512) | top-1 |
|---|---|---:|---:|
| `main` | rotation + SmoothQuant + GPTQ | **0.01098** | 95.90% |
| branch `v1-smoothquant-rtn` (`417ede1e`) | SmoothQuant + RTN | 0.01414 | 95.32% |

Load the original release with
`revision="v1-smoothquant-rtn"` if you need byte-identical weights to the first
publication.

⚠ **`main` also fixes a packaging bug present in v1**: `preprocessor_config.json`,
`video_preprocessor_config.json`, `vocab.json` and `merges.txt` were missing,
because `save_pretrained` does not copy processor auxiliaries and all of our own
serving used `--language-model-only`, which never constructs an image processor.
vLLM therefore refused to start with vision enabled
(`OSError: Can't load image processor`). The vision *weights* were never
affected. Reported in discussion #1 — thanks to @Neiko2002 for catching it.

## Known limitations

* **`mtp.norm` is an approximation.** The drafter's final norm and the target's
  both feed one shared `lm_head`, so under rotation both `(1+w)` folds cannot
  be exact — this is provable, not an oversight. `mtp.norm` is zeroed
  uncompensated. The cost is drafter-only and measured: acceptance length
  2.715 → 2.674 on a fixed harness. Target-model quality is unaffected.
* The KLD values are upper bounds (see caveats above).
* No long-duration soak test has been run on this exact revision.

## Files

- `model-0000N-of-00015.safetensors` — W8A8-INT8 weights + BF16 exclusions
- `model-mtp.safetensors` — BF16 MTP drafter (15 tensors)
- `config.json` — includes the corrected quantization ignore list (do not
  regenerate it from the recipe; see notes above)
- `recipe.yaml` — llm-compressor stage (quantization step only)
- `quantization/` — the full pipeline: rotation + verifier, activation capture,
  α-proxy, fold transform, build and MTP-graft scripts
- `chat_template.jinja`, `preprocessor_config.json`, tokenizer files — upstream

## License

Apache-2.0 for this packaging; the base model is Apache-2.0 by Qwen.
