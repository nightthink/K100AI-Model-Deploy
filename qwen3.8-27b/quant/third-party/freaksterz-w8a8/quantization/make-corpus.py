#!/usr/bin/env python3
"""make-corpus.py — T2c mixed calibration corpus.

~60% WikiText-2 train + ~25% real code + ~15% tool-call JSON transcripts.
Tool-JSON was the worst KLD domain on the 3.6 measurements (truncated-KL
0.149 vs 0.076 corpus mean); abs-max calibration never saw that distribution.
Interleaved in blocks so every 2048-token window is mostly one domain.
"""
import glob
import json
import random

OUT = "/glm/qwen38-sq-w8a8/calib-mixed.txt"
WIKI = "/glm/muse/kld-corpus/wiki.train.raw"

rng = random.Random(0)

wiki = open(WIKI, errors="replace").read()

# code: this project's own scripts (real, varied python/bash)
code_files = sorted(
    glob.glob("/glm/qwen38-sq-w8a8/*.py")
    + glob.glob("/glm/qwen36-sq-w8a8/*.py")
    + glob.glob("/eval/*.py")
)
code = "\n\n".join(open(f, errors="replace").read() for f in code_files)

# tool-JSON: synthesized OpenAI-style tool-call transcripts (the production shape)
TOOLS = ["get_quote", "place_order", "fetch_ledger", "search_docs", "run_backtest"]
SYMS = ["AAPL", "MSFT", "SPY", "NVDA", "TSM", "QQQ", "GLD", "BTC-USD"]
frags = []
for i in range(400):
    t = rng.choice(TOOLS)
    args = {
        "symbol": rng.choice(SYMS),
        "qty": rng.randint(1, 500),
        "side": rng.choice(["buy", "sell"]),
        "note": rng.choice([
            "took profit, trimming position, see thesis doc",
            "rebalance leg 2 of 3, keep delta neutral",
            "stop moved to breakeven; watch CPI print at 8:30",
        ]),
        "legs": [{"strike": rng.randint(50, 600), "kind": rng.choice(["call", "put"])}
                 for _ in range(rng.randint(1, 3))],
    }
    call = {"id": f"call_{i:04d}", "type": "function",
            "function": {"name": t, "arguments": json.dumps(args)}}
    result = {"status": "ok", "filled": args["qty"], "avg_px": round(rng.uniform(10, 600), 2),
              "ts": f"2026-08-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z"}
    frags.append(
        f'<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>\n'
        f'<tool_response>\n{json.dumps(result)}\n</tool_response>\n'
    )
tooljson = "\n".join(frags)

# interleave in ~8 KB blocks, 60/25/15
def blocks(s, size=8192):
    return [s[i:i+size] for i in range(0, len(s), size)]

wb, cb, jb = blocks(wiki), blocks(code), blocks(tooljson)
out, wi, ci, ji = [], 0, 0, 0
# pattern of 20 blocks: 12 wiki, 5 code, 3 json. Code/json REPEAT via modulo
# (they are smaller pools); wiki is large enough to never wrap at this size.
pattern = ["w"]*12 + ["c"]*5 + ["j"]*3
while len(out) * 8192 < 4_000_000:
    for p in pattern:
        if p == "w": out.append(wb[wi % len(wb)]); wi += 1
        elif p == "c": out.append(cb[ci % len(cb)]); ci += 1
        else: out.append(jb[ji % len(jb)]); ji += 1

open(OUT, "w").write("".join(out))
print(f"wrote {OUT}: {len(''.join(out)):,} chars "
      f"(wiki {wi} / code {ci} / json {ji} blocks)")
