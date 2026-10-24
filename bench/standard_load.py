# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""★★ 唯一标准性能测试 —— 全部线（01/02/09/10/11/12/13…）的性能数字都出自本脚本。

    用法: standard_load.py --port 8101 [--cycles 2] [--out 目录]
          换线只改 --port / --model，口径不变，因此各线数字可直接横向比较。

本仓**只有这一个**性能基准脚本。任何其它脚本都不是标准测试，不得用于出性能数字。

## 负载口径（用户 2026-09-17 确认，唯一权威）

    输入      45K – 350K token，均值 160K
    输出      500 – 4K token，均值 800
    命中率    输入 cache 命中：均值 95%，最小 85%
    周期      涨到 350K 后，切换到 ~70K 的全新前缀（全新 = 必然未命中）

历史 README 里那套「23K冷启→+1-2K/轮→350K→压缩回40K、出300」是旧口径，已作废。

## 产出：四个指标的矩阵（输入深度 × 输出规模）

    TTFT、prefill 速率、decode 速率、E2E 时间

## 两个设计要点

**一、命中率是构造出来的。** 某轮命中率 = 上轮 prompt / 本轮 prompt。设上轮深度 P、
本轮新增 d（上轮模型输出 + 本轮用户输入），命中率 = P/(P+d)。要 95% 则
d = P × 0.05/0.95 ≈ 0.0526×P：70K 时加约 3.7K、160K 时加约 8.4K、350K 时加约 18.4K。
每 7 轮按最小命中 85% 取一次大增量，覆盖真实负载里的抖动。

**二、矩阵靠定深扫输出填满。** 若输出规模跟着轮次走，它会与深度强相关，
格子打不满。故每到一个深度档，就在**同一深度**上连发几种输出规模的探针，
探针**不追加进对话历史**——深度因此不变，前缀仍然命中，扫完再继续增长。
"""
import argparse, json, os, random, sys, time, urllib.request

FILLER = ("在分布式系统的日常运维中，配置管理、服务发现与健康检查构成三个彼此独立又相互支撑的支柱。"
          "运维人员需持续关注节点状态、网络分区与数据一致性带来的影响，并在容量规划时兼顾冗余与成本。"
          "灰度发布、回滚预案与可观测性建设同样不可偏废，任何一环缺失都会在故障时被放大。")

LOAD = dict(in_min=45_000, in_max=350_000, in_mean=160_000,
            out_min=500, out_max=4_000, out_mean=800,
            hit_mean=0.95, hit_min=0.85, fresh_prefix=70_000)

IN_BUCKETS = [(45_000, 80_000, "45–80K"), (80_000, 120_000, "80–120K"),
              (120_000, 180_000, "120–180K"), (180_000, 250_000, "180–250K"),
              (250_000, 320_000, "250–320K"), (320_000, 10**9, "320–350K")]

OUT_BUCKETS = [(500, 900, "500–900"), (900, 1_500, "0.9–1.5K"),
               (1_500, 2_500, "1.5–2.5K"), (2_500, 4_000, "2.5–4K")]

# 定深扫描用的输出规模，四个档各取一个代表值
OUT_PROBES = [700, 1_200, 2_000, 3_500]


# 字符 → token 的换算比。默认值来自 Qwen3.8 实测（中文填充约 1.73 字/token），
# 启动时会用一次真实请求标定，避免「按 1 字 1 token 估算」导致永远涨不到 350K。
# ★ 2026-09-17 实测教训：按 1:1 估算时 pad(350000) 只产出约 20 万 token。
CHARS_PER_TOK = 1.73


def pad(n_chars):
    """造 n_chars 个字符的中文文本。"""
    return (FILLER * (n_chars // len(FILLER) + 1))[:n_chars] if n_chars > 0 else ""


def pad_tok(n_tok):
    """造**约 n_tok 个 token**的中文文本（按标定后的字/token 比换算）。"""
    return pad(int(n_tok * CHARS_PER_TOK)) if n_tok > 0 else ""


def calibrate(url, model):
    """用一次真实请求标定字/token 比——不同分词器差异很大，估算不得。"""
    global CHARS_PER_TOK
    probe_chars = 20_000
    _, _, ptok, _, _, _ = chat(url, model, [{"role": "user",
                                             "content": pad(probe_chars) + "\n\n回复「收到」。"}], 16)
    if ptok and ptok > 100:
        CHARS_PER_TOK = probe_chars / ptok
        print(f"  标定：{probe_chars:,} 字 → {ptok:,} token，"
              f"字/token = {CHARS_PER_TOK:.3f}", flush=True)
    else:
        print(f"  ⚠ 标定失败（prompt_tok={ptok}），沿用默认 {CHARS_PER_TOK}", flush=True)


def chat(url, model, msgs, max_tokens, timeout=2400, force_len=False):
    """流式请求 → (ttft, e2el, prompt_tok, completion_tok, cached_tok, text)。

    TTFT 只能流式测；cached_tok 服务端不回时为 None，由调用方按上轮深度推算。

    ★ force_len：`max_tokens` 只是**上限**不是目标——模型答完就停，实测常只出
      2–4 个 token，输出维度会整个塌掉（2026-09-17 实测：输出均值 146 而非 800）。
      故测性能的轮次一律 `ignore_eos` 强制满额，才拿得到指定输出规模下的 decode/E2E。
    """
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.8, "stream": True,
            "stream_options": {"include_usage": True}}
    if force_len:
        body["ignore_eos"] = True
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, out, ptok, ctok, cached = time.time(), None, [], 0, 0, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            s = raw.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if s == "[DONE]":
                break
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                u = d["usage"]
                ptok = u.get("prompt_tokens", ptok)
                ctok = u.get("completion_tokens", ctok)
                det = u.get("prompt_tokens_details") or {}
                if det.get("cached_tokens") is not None:
                    cached = det["cached_tokens"]
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                # ★ 本模型可能只吐 reasoning 而 content 为空（实测）。只看 content
                #   会让 TTFT 丢成 None——首 token 时刻必须两者都算。
                p = delta.get("content") or ""
                r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if p or r:
                    if ttft is None:
                        ttft = time.time() - t0
                    if p:
                        out.append(p)
    return ttft, time.time() - t0, ptok, ctok, cached, "".join(out)


def metrics(ttft, e2el, ptok, ctok, cached, prev_depth):
    """四个标准指标。

    prefill 速率取**总输入 / TTFT**（用户实际感受：整个输入等了多久出首字）。
    95% 命中下这个值天然偏高，故同时算出未命中部分的速率备查。
    """
    hit = (cached / ptok) if cached else ((prev_depth / ptok) if ptok and prev_depth else 0.0)
    uncached = ptok - (cached if cached else int(ptok * hit))
    return dict(ttft=ttft, e2el=e2el, depth=ptok, out_tok=ctok, hit=hit,
                prefill_tok_s=(ptok / ttft) if ttft else 0.0,
                prefill_uncached_tok_s=(uncached / ttft) if ttft else 0.0,
                decode_tok_s=(ctok / (e2el - ttft)) if ttft and e2el > ttft else 0.0)


def run_cycle(url, model, cyc, rng, log):
    rows = []

    def rec(kind, m, **extra):
        r = dict(kind=kind, cycle=cyc, **m, **extra)
        rows.append(r)
        log(r)

    # ── 全新前缀（必然未命中）；首周期从负载下沿 45K 起 ──────────
    start = LOAD["in_min"] if cyc == 1 else LOAD["fresh_prefix"]
    msgs = [{"role": "user",
             "content": f"[全新会话 {cyc} · {time.time():.6f}] 以下为本次任务的全部背景材料。\n"
                        + pad_tok(start) + "\n\n请通读以上材料，用一句话确认。"}]
    ttft, e2el, ptok, ctok, cached, txt = chat(url, model, msgs, 700, force_len=True)
    if not ptok:
        raise SystemExit("✗ 服务端未回 usage.prompt_tokens，深度与命中率无从判定，中止。")
    rec("全新前缀", metrics(ttft, e2el, ptok, ctok, cached, 0))
    msgs.append({"role": "assistant", "content": txt or "已确认。"})

    done_buckets, rnd = set(), 0
    while ptok < LOAD["in_max"]:
        # ── 到达一个新深度档 → 就地扫输出规模，填矩阵 ──────────
        lab = next((l for lo, hi, l in IN_BUCKETS if lo <= ptok < hi), None)
        if lab and lab not in done_buckets:
            done_buckets.add(lab)
            # ★ R20.1：探针必须与常态轮**同等命中率**，否则矩阵的「输出规模」这一维
            #   会被缓存热度污染。此前探针只重发同一前缀 ⇒ 命中近 100%，同深度下
            #   探针列 TTFT 比常态轮小一个数量级、prefill 虚高 24 倍，看着像
            #   「输出越大越快」，实为探针没付未命中 prefill 的代价。
            #   故给每个探针补一段与常态轮等量（当前深度的 5.26%）的**新内容**，
            #   且带唯一标记开头，使各探针之间也互不命中。
            for mt in OUT_PROBES:
                d_probe = int(ptok * (1 - LOAD["hit_mean"]) / LOAD["hit_mean"])
                fresh = f"[探针{mt}·{time.time():.6f}]\n" + pad_tok(max(200, d_probe))
                probe = msgs + [{"role": "user",
                                 "content": fresh + f"\n\n（定深探针·出{mt}）请就以上内容展开论述。"}]
                p_ttft, p_e2el, p_ptok, p_ctok, p_cached, _ = chat(
                    url, model, probe, mt, force_len=True)
                rec("探针", metrics(p_ttft, p_e2el, p_ptok, p_ctok, p_cached, ptok),
                    want_out=mt, in_bucket=lab)

        # ── 常态轮：按目标命中率定增量，推进深度 ────────────────
        rnd += 1
        prev = ptok
        target_hit = LOAD["hit_min"] if rnd % 7 == 0 else LOAD["hit_mean"]
        d_target = prev * (1 - target_hit) / target_hit
        user_tok = max(200, int(d_target - ctok))
        msgs.append({"role": "user", "content": pad_tok(user_tok) +
                     f"\n\n（第 {rnd} 轮）请基于以上全部内容给出结论。"})
        # ★ R20.2：常态轮的输出分布必须真的以 800 为均值（口径写的是
        #   「输出 500–4K、均值 800」）。此前只在 500–900 里取，均值约 700，
        #   长输出全靠探针撑，于是「输出均值」这项标签与所测对象对不上。
        #   现按混合分布取：88% 取 500–900、10% 取 1000–1800、2% 取 2500–4000，
        #   期望值 ≈ 0.88×700 + 0.10×1400 + 0.02×3250 ≈ 821。
        _r = rng.random()
        if _r < 0.88:   want = rng.randint(LOAD["out_min"], 900)
        elif _r < 0.98: want = rng.randint(1_000, 1_800)
        else:           want = rng.randint(2_500, LOAD["out_max"])
        ttft, e2el, ptok, ctok, cached, txt = chat(url, model, msgs, want, force_len=True)
        msgs.append({"role": "assistant", "content": txt or "。"})
        rec("常态轮", metrics(ttft, e2el, ptok, ctok, cached, prev), round=rnd)
        if rnd > 200:
            raise SystemExit(f"✗ 已 {rnd} 轮仍未到 350K（当前 {ptok}），增量逻辑异常。")
    return rows


def _p50(vals):
    v = sorted(x for x in vals if x)
    return v[len(v) // 2] if v else None


def report(rows):
    perf = [r for r in rows if r["kind"] in ("探针", "常态轮")]
    norm = [r for r in rows if r["kind"] == "常态轮"]
    fresh = [r for r in rows if r["kind"] == "全新前缀"]
    o = ["# 标准负载性能报告", "",
         "负载：输入 45K–350K（均值 160K）／输出 500–4K（均值 800）／"
         "命中率均值 95% 最小 85%／350K 后切 ~70K 全新前缀", ""]

    # ── 一、自证 ──────────────────────────────────────────────
    if norm:
        ins = [r["depth"] for r in norm]
        outs = [r["out_tok"] for r in norm]
        hits = [r["hit"] for r in norm]
        mi, mo = sum(ins) / len(ins), sum(outs) / len(outs)
        mh, lh = sum(hits) / len(hits), min(hits)
        o += ["## 一、自证：实际跑成的负载", "",
              "| 项 | 目标 | 实际 | 判定 |", "|---|---|---|---|"]
        for name, tgt, act, ok in (
                ("输入范围", "45K–350K", f"{min(ins):,}–{max(ins):,}",
                 min(ins) >= 40_000 and max(ins) >= 320_000),
                ("输入均值", "160K", f"{mi:,.0f}", 130_000 <= mi <= 190_000),
                # ★ R20.2：标签必须与所测对象一致。这一项统计的是**常态轮**
                #   ——它代表生产流量；探针是为填矩阵而加的仪器，不属于负载本身。
                ("常态轮输出均值", "800", f"{mo:,.0f}", 650 <= mo <= 950),
                ("命中率均值", "95%", f"{mh*100:.1f}%", mh >= 0.92),
                ("命中率最小", "≥85%", f"{lh*100:.1f}%", lh >= 0.83)):
            o.append(f"| {name} | {tgt} | {act} | {'✓' if ok else '✗ 未达标'} |")
        # 参考行：含探针的全体均值。探针为仪器、刻意偏重长输出，故**不参与判定**，
        # 但必须列出来，免得读者把上面那行误当成「全部请求的输出均值」。
        allo = [r["out_tok"] for r in perf if r.get("out_tok")]
        if allo:
            o.append(f"| 全体输出均值（含探针） | — | {sum(allo)/len(allo):,.0f} | 参考，不判定 |")
        o += ["", "★ 任一项 ✗ 则下方矩阵不作数——说明没跑成该负载。",
              "（「常态轮」= 生产流量；「探针」= 为填矩阵在同一深度上加发的请求，"
              "已与常态轮同等命中率，见下方矩阵说明。）", ""]

    # ── 二、四个指标矩阵（输入 × 输出）────────────────────────
    def cell(lo_i, hi_i, lo_o, hi_o, key, fmt):
        g = [r for r in perf if lo_i <= r["depth"] < hi_i and lo_o <= r["out_tok"] < hi_o]
        v = _p50([r[key] for r in g])
        return (fmt.format(v) + f"<br><sub>n={len(g)}</sub>") if v else "—"

    for key, title, fmt in (
            ("ttft", "TTFT（首 token 时延）", "{:.2f}s"),
            ("prefill_tok_s", "prefill 速率（总输入/TTFT）", "{:,.0f}"),
            ("decode_tok_s", "decode 速率", "{:.1f}"),
            ("e2el", "E2E 时间", "{:.1f}s")):
        o += [f"## 二·{title}　（中位数，行=输入深度，列=输出规模）", "",
              "| 输入＼输出 | " + " | ".join(l for _, _, l in OUT_BUCKETS) + " |",
              "|---|" + "---|" * len(OUT_BUCKETS)]
        for lo_i, hi_i, li in IN_BUCKETS:
            cells = [cell(lo_i, hi_i, lo_o, hi_o, key, fmt) for lo_o, hi_o, _ in OUT_BUCKETS]
            if all(c == "—" for c in cells):
                continue
            o.append(f"| **{li}** | " + " | ".join(cells) + " |")
        o.append("")

    # ── 三、全新前缀切换成本 ──────────────────────────────────
    if fresh:
        o += ["## 三、全新前缀切换成本（未命中）", "",
              "| 周期 | 输入 tok | TTFT | prefill tok/s | decode tok/s |",
              "|---|---|---|---|---|"]
        for r in fresh:
            o.append(f"| {r['cycle']} | {r['depth']:,} | {r['ttft']:.1f}s | "
                     f"{r['prefill_tok_s']:,.0f} | {r['decode_tok_s']:.1f} |")
        o.append("")

    # ── 附注：95% 命中下 prefill 的两种口径 ───────────────────
    up = _p50([r["prefill_uncached_tok_s"] for r in norm])
    gp = _p50([r["prefill_tok_s"] for r in norm])
    if up and gp:
        o += ["> **prefill 速率的两种口径**：上表用「总输入 ÷ TTFT」"
              f"（常态轮中位 {gp:,.0f} tok/s），这是用户实际感受——整个输入等多久出首字；"
              f"若只算未命中部分则为 {up:,.0f} tok/s。95% 命中下前者必然远高于后者，"
              "跨线比较时须确认用的是同一口径。", ""]
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8101")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--cycles", type=int, default=2)
    # 给了 --hours 就按时长跑（压测口径通常是「压两小时」），--cycles 被忽略。
    # 到点不开新周期，但不腰斩正在跑的周期，以免最后一个周期的矩阵是残的。
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--seed", type=int, default=7)
    # 从已落盘的 rounds.jsonl 重算报告，不发任何请求。
    # 用途：跑被截断时（report() 只在周期循环正常退出后才执行）报告会丢，
    # 有了它就能从原始行重算，不必为了拿报告而白等一个超预算的周期。
    ap.add_argument("--report-only", action="store_true")
    # 只取前 N 个周期（配合 --report-only）：截断的末周期可据此剔除，
    # 使报告严格对应「预算内完整跑完的那几个周期」。0=全取。
    ap.add_argument("--max-cycle", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # ── 只重算报告：读 rounds.jsonl，不碰服务 ──────────────────────────
    if a.report_only:
        if not a.out:
            raise SystemExit("✗ --report-only 需要 --out 指向含 rounds.jsonl 的目录")
        src = os.path.join(a.out, "rounds.jsonl")
        if not os.path.exists(src):
            raise SystemExit(f"✗ 找不到 {src}")
        rows = []
        for line in open(src):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if a.max_cycle and r.get("cycle", 0) > a.max_cycle:
                continue
            rows.append(r)
        cycles = sorted({r.get("cycle", 0) for r in rows})
        print(f"  从 {src} 读入 {len(rows)} 行，周期 {cycles}"
              + (f"（已按 --max-cycle {a.max_cycle} 截取）" if a.max_cycle else ""))
        rep = report(rows)
        print("\n" + rep)
        with open(os.path.join(a.out, "report.md"), "w") as f:
            f.write(rep)
        print(f"→ {a.out}/report.md")
        return

    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    rng = random.Random(a.seed)
    # ★ 必须先标定字/token 比，否则按默认值造的上下文深度会系统性偏小，
    #   永远涨不到 350K（2026-09-17 实测：按 1:1 估算只到 20 万）。
    calibrate(url, a.model)

    fh = None
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fh = open(os.path.join(a.out, "rounds.jsonl"), "a")

    def log(r):
        b = [f"[{time.strftime('%H:%M:%S')}]", f"{r['kind']:<6}",
             f"入{r['depth']:>7,}", f"出{r['out_tok']:>5}", f"命中{r['hit']*100:5.1f}%"]
        if r.get("ttft"):
            b.append(f"TTFT{r['ttft']:>6.2f}s")
        if r.get("prefill_tok_s"):
            b.append(f"prefill{r['prefill_tok_s']:>8,.0f}")
        if r.get("decode_tok_s"):
            b.append(f"decode{r['decode_tok_s']:>5.1f}")
        if r.get("e2el"):
            b.append(f"E2E{r['e2el']:>6.1f}s")
        print("  ".join(b), flush=True)
        if fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    rows = []
    if a.hours > 0:
        # 按时长跑：压测时长是人给的口径（「压两小时」），按周期数估算不准——
        # 单周期耗时随线的 decode/prefill 差异很大，同一个 --cycles 在不同线上
        # 可能相差一倍。到点后不再开新周期，但**不腰斩正在跑的周期**，
        # 否则最后一个周期的矩阵是残的，自证表也会因此失真。
        #
        # ★ 2026-09-17 实测缺陷：只在周期开头判 `now < deadline` 是不够的。
        #   那次 A 在距截止仅 1.5 分钟时通过了判断，随即承诺一个整周期，
        #   实际跑到超预算 56 分钟（2h 预算跑成 2h56m）。故还要求**剩余预算
        #   足以跑完一个周期**：按已完成周期的实测均长，剩余不足 60% 就不再开。
        deadline = time.time() + a.hours * 3600
        durs, c = [], 0
        while True:
            left = deadline - time.time()
            if left <= 0:
                print(f"\n── 时长已到，共完成 {c} 个周期 ──", flush=True)
                break
            if durs:
                need = 0.6 * (sum(durs) / len(durs))
                if left < need:
                    print(f"\n── 剩余 {left/60:.1f} 分钟，不足跑完一个周期"
                          f"（实测均长 {sum(durs)/len(durs)/60:.1f} 分钟的 60%），"
                          f"不再开新周期；共完成 {c} 个 ──", flush=True)
                    break
            c += 1
            print(f"\n═══ 周期 {c}（按时长跑，剩 {left/3600:.2f}h）═══", flush=True)
            t0 = time.time()
            rows += run_cycle(url, a.model, c, rng, log)
            durs.append(time.time() - t0)
    else:
        for c in range(1, a.cycles + 1):
            print(f"\n═══ 周期 {c}/{a.cycles} ═══", flush=True)
            rows += run_cycle(url, a.model, c, rng, log)

    rep = report(rows)
    print("\n" + rep)
    if a.out:
        with open(os.path.join(a.out, "report.md"), "w") as f:
            f.write(rep)
        print(f"→ {a.out}/report.md")


if __name__ == "__main__":
    sys.exit(main())
