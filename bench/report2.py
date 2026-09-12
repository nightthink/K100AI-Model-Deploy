#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""同口径报告生成器：prefill 与 decode 全程分开统计。

用法:
  python3 report2.py <requests.jsonl> [标题] > report.md
  python3 report2.py a.jsonl b.jsonl --compare "C线" "A线" > compare.md

口径定义（两条线强制一致）：
  prefill 速率 = prompt_tok / ttft          —— 消化输入的速度
  decode 速率  = 1 / tpot                    —— 吐出 token 的速度（单流体感）
  合计吞吐     = (prompt_tok + completion_tok) / e2el
冷/热：同一会话内 rnd==0 记为冷（无前缀可复用），rnd>0 记为热。
"""
import json, sys, statistics as st


def load(path, until=None):
    """until: "HH:MM:SS"，只保留该时刻之前的记录。
    服务崩溃后驱动仍会继续打请求，那些"打空气"的失败不是服务表现，
    必须截断，否则错误率会被虚报（实测曾虚报到 65%）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "probe" or "scen" not in d:
                continue
            if until and str(d.get("t", "")) >= until:
                continue
            rows.append(d)
    return rows


def ok_rows(rows):
    return [r for r in rows if r.get("ok") and r.get("ttft") and r.get("prompt_tok")]


# 行为结果（模型没在限定轮数内做某事），不是服务故障——必须与服务错误分开计
BEHAVIOR = {"no_write_call", "no_tool_call", "no_output"}


def split_bad(rows):
    """把非 ok 行拆成「服务错误」与「场景未达成」。"""
    svc, beh = [], []
    for r in rows:
        if r.get("ok"):
            continue
        e = str(r.get("err") or "")
        (beh if e in BEHAVIOR else svc).append(r)
    return svc, beh


def pct(v, q):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * q))]


def f(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"


def prefill_rate(r):
    return r["prompt_tok"] / r["ttft"] if r.get("ttft") else None


def decode_rate(r):
    return 1.0 / r["tpot"] if r.get("tpot") else None


# 按**字符**分桶——生产口径是字符（输入 40K–350K、均值约 150K）。
# 早先用 token 分桶且边界落在 8–20K/20–40K，几乎把真实分布全压进两个桶里，看不出差异。
BUCKETS = [(0, 40000, "<40K"), (40000, 80000, "40–80K"), (80000, 150000, "80–150K"),
           (150000, 250000, "150–250K"), (250000, 350000, "250–350K"),
           (350000, 10**9, "≥350K")]

CHARS_PER_TOK = 3.6   # 本语料实测值，仅在缺 prompt_chars 的历史数据上回退使用


def in_chars(r):
    """请求的输入字符数：优先用实测值，老数据回退到 token×3.6 估算。"""
    v = r.get("prompt_chars")
    if v:
        return v
    return int(r.get("prompt_tok", 0) * CHARS_PER_TOK)


def bucket(n):
    for lo, hi, lab in BUCKETS:
        if lo <= n < hi:
            return lab
    return "?"


OUT_BUCKETS = [(0, 64, "<64"), (64, 256, "64–256"), (256, 1024, "256–1K"),
               (1024, 4096, "1–4K"), (4096, 10**9, "≥4K")]


def out_bucket(n):
    for lo, hi, lab in OUT_BUCKETS:
        if lo <= n < hi:
            return lab
    return "?"


def batch_span(rows):
    """一组并发请求的墙钟跨度：按发起时刻（ts-e2el）聚成批，逐批取
    max(结束) - min(开始) 再累加。聚合速率必须用它做分母——
    用 Σe2el 会把并行时间重复计入，把吞吐算低。"""
    by = {}
    for r in rows:
        by.setdefault(round((r["ts"] - (r.get("e2el") or 0)) / 60), []).append(r)
    span = 0.0
    for b in by.values():
        span += max(x["ts"] for x in b) - min(x["ts"] - (x.get("e2el") or 0) for x in b)
    return span or 1.0


def section_single(rows, out):
    """单流：无并发标记的场景（S1/S2/S4/S5/S6），按输出长度分桶。"""
    sub = [r for r in rows if not r.get("conc")]
    if not sub:
        return
    out.append("\n## 单流性能（并发=1，按输出长度分桶）\n")
    out.append("| 输出规模 | 样本 | 中位输入 tok | 中位输出 tok | prefill tok/s P50（冷启） "
               "| **decode tok/s P50** | TTFT P50 |")
    out.append("|---|---|---|---|---|---|---|")
    for _, _, lab in OUT_BUCKETS:
        b = [r for r in sub if out_bucket(r.get("completion_tok", 0)) == lab]
        if not b:
            continue
        cold = [r for r in b if r.get("rnd") == 0]   # 只认显式轮次标记
        pr = [x for x in (prefill_rate(r) for r in cold) if x]
        dr = [decode_rate(r) for r in b if r.get("tpot")]
        out.append(f"| {lab} | {len(b)} | {int(st.median([r['prompt_tok'] for r in b])):,} "
                   f"| {int(st.median([r.get('completion_tok',0) for r in b]))} "
                   f"| {f(pct(pr,.5),0)} | **{f(pct(dr,.5))}** "
                   f"| {f(pct([r['ttft'] for r in b],.5),2)}s |")
    cold = [r for r in sub if r.get("rnd") == 0]
    pr = [x for x in (prefill_rate(r) for r in cold) if x]
    dr = [decode_rate(r) for r in sub if r.get("tpot")]
    out.append(f"\n**单流总计：prefill（冷启）P50 {f(pct(pr,.5),0)} tok/s"
               f"（冷启样本 {len(cold)}）；decode P50 {f(pct(dr,.5))} tok/s**，样本 {len(sub)}\n")
    out.append("> prefill 列只统计**有轮次标记且为首轮**的请求（目前只有 S1 会打 `rnd`）。"
               "其余场景无法判定是否命中前缀缓存——SGLang 的 OpenAI 接口不返回 "
               "`prompt_tokens_details.cached_tokens`——把它们并入冷启会让 prefill 虚高数倍，"
               "故此处留空而不是猜。\n")


def section_prefill(rows, out):
    out.append("## 一、prefill 性能（消化输入）\n")
    out.append("口径：`prefill 速率 = prompt_tok / TTFT`。冷=会话首轮（无前缀可复用），"
               "热=同会话后续轮（前缀命中）。\n")
    out.append("> ⚠ **这个口径在高命中率下会严重高估 prefill 的真实工作量**：命中的 token "
               "根本没被重新计算，却仍计入分子。生产实际约 95% 命中，因此该列只反映"
               "「用户感受到的输入消化速度」，**不代表 prefill 算力**。\n"
               "> 真实 prefill 吞吐请用 `cachestat.py <容器名>`，它从容器日志的 "
               "`#new-token` / `#cached-token` 实测——SGLang 不打 `Prefix cache hit rate`，"
               "只能这样拿。\n")
    out.append("> ⚠ 冷/热只能靠**轮次标记**判定，而目前只有 S1 会打 `rnd`。"
               "SGLang 的 OpenAI 接口 `prompt_tokens_details` 恒为 null，拿不到真实命中数，"
               "所以无标记的场景一律归入「冷热未知」单列，**不并入冷启**。\n")
    for tag, sel in (("冷启（首轮 · 仅 S1）", lambda r: r.get("rnd") == 0),
                     ("命中（后续轮 · 仅 S1）", lambda r: isinstance(r.get("rnd"), int) and r["rnd"] > 0),
                     ("冷热未知（无轮次标记的场景）", lambda r: r.get("rnd") is None)):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        out.append(f"\n### {tag}\n")
        out.append("| 输入规模(字符) | 样本 | TTFT P50 | P95 | prefill tok/s P50 | 疑似命中(剔除) | 中位输入字符 | 中位输入 tok |")
        out.append("|---|---|---|---|---|---|---|---|")
        for _, _, lab in BUCKETS:
            b = [r for r in sub if bucket(in_chars(r)) == lab]
            if not b:
                continue
            # 同一输入规模下，prefill 速率远高于中位者只可能是前缀缓存命中
            # （实测：同为 24,983 tok 的请求，冷启 3,720 tok/s vs 命中 41,124 tok/s）。
            # rnd 字段不可靠——run_bench 每循环复用同一 sess_id、语料又跨循环相同，
            # 所以 rnd=0 并不保证冷。这里改用物理证据剔除，并公开剔除条数。
            raw_pr = sorted(((prefill_rate(r), r) for r in b if prefill_rate(r)),
                            key=lambda z: z[0])
            # 阈值不能用桶内中位——中位本身会被命中样本污染（实测 80–150K 桶
            # 因此只剔除 10/90 条，仍报出 9,369 的假冷启）。改用**下半段的中位**
            # 作冷启参考：命中样本速率高、只在上半段，污染不到它。
            lower = [x for x, _ in raw_pr[:max(1, len(raw_pr) // 2)]]
            med = st.median(lower) if lower else 0
            keep = [(x, r) for x, r in raw_pr if x <= 3 * med]
            drop = len(raw_pr) - len(keep)
            tt = [r["ttft"] for _, r in keep] or [r["ttft"] for r in b]
            pr = [x for x, _ in keep]
            bb = [r for _, r in keep] or b
            out.append(f"| {lab} | {len(b)} | {f(pct(tt,.5),2)}s | {f(pct(tt,.95),2)}s "
                       f"| **{f(pct(pr,.5),0)}** | {drop} "
                       f"| {int(st.median([in_chars(r) for r in bb])):,} "
                       f"| {int(st.median([r['prompt_tok'] for r in bb])):,} |")
        _raw = sorted(x for x in (prefill_rate(r) for r in sub) if x)
        _low = _raw[:max(1, len(_raw) // 2)]
        _med = st.median(_low) if _low else 0
        allpr = [x for x in _raw if x <= 3 * _med]
        out.append(f"\n全部 {tag}：prefill 速率 P50 **{f(pct(allpr,.5),0)}** / "
                   f"P95 {f(pct(allpr,.95),0)} tok/s，样本 {len(sub)}\n")


def section_decode(rows, out):
    out.append("\n## 二、decode 性能（吐出 token）\n")
    out.append("口径：`decode 速率 = 1 / TPOT`，与 prefill 完全分开。"
               "TPOT 只覆盖首 token 之后的解码阶段，不含 prefill。\n")
    out.append("\n| 场景 | 并发 | 样本 | TPOT P50 | P95 | **decode tok/s P50** | P95 | 中位输出 tok |")
    out.append("|---|---|---|---|---|---|---|---|")
    for sc in sorted({r["scen"] for r in rows}):
        b = [r for r in rows if r["scen"] == sc and r.get("tpot")]
        if not b:
            continue
        tp = [r["tpot"] for r in b]
        dr = [decode_rate(r) for r in b]
        cset = sorted({r.get("conc") for r in b if r.get("conc")})
        ctag = "/".join(str(x) for x in cset) if cset else "1（单流）"
        out.append(f"| {sc} | {ctag} | {len(b)} | {f(pct(tp,.5)*1000,1)}ms | {f(pct(tp,.95)*1000,1)}ms "
                   f"| **{f(pct(dr,.5))}** | {f(pct(dr,.95))} "
                   f"| {int(st.median([r['completion_tok'] for r in b]))} |")
    alld = [decode_rate(r) for r in rows if r.get("tpot")]
    out.append(f"\n全场景 decode 速率：P50 **{f(pct(alld,.5))}** / P95 {f(pct(alld,.95))} tok/s\n")

    # decode 是否随上下文长度衰减
    out.append("\n### decode 速率 vs 输入长度\n")
    out.append("★ **必须按单流/并发拆开**：混在一起会得出错误结论。实测同一个 40–80K 桶里，"
               "148 条单流是 63.6 tok/s、330 条并发是 24.3 tok/s，"
               "平均成 27.8 就看不出任何东西了。\n")
    out.append("\n| 输入规模(字符) | 类型 | 样本 | decode tok/s P50 |")
    out.append("|---|---|---|---|")
    for _, _, lab in BUCKETS:
        for kind, sel in (("单流", lambda r: not r.get("conc")),
                          ("并发", lambda r: bool(r.get("conc")))):
            b = [r for r in rows
                 if bucket(in_chars(r)) == lab and r.get("tpot") and sel(r)]
            if b:
                out.append(f"| {lab} | {kind} | {len(b)} "
                           f"| {f(pct([decode_rate(r) for r in b],.5))} |")


def section_concurrency(rows, out):
    sub = [r for r in rows if r["scen"] == "S3" and r.get("tpot")]
    if not sub:
        return
    out.append("\n## 三、并发下的 prefill / decode 分解\n")
    out.append("真实 opencode 负载是 **prefill 主导**的，只看一个数会误判。\n")
    out.append("★ 两种口径必须分开看，混用会得出相反结论：\n"
               "  · **单请求 P50**＝每个请求自己感受到的速率（体感）\n"
               "  · **聚合**＝整机同时做的功＝Σtok / 批次墙钟（产能）\n")
    # ★ 2026-09-11 补：此前 §一 有这条警告而本节没有，险些把口径假象当成 4 倍提速交出去。
    # ★ 2026-09-11 补②：「单请求 prefill P50」在三档并发之间也不可比，实证见下。
    out.append("\n> ⚠ **「单请求 prefill P50」不能跨并发档比较**：每轮 S3 按 4→8→16 路依次跑，\n"
               "> **4 路是第一档，此时共享前缀尚未建立、几乎全冷算**；到 8 路前缀已热。\n"
               "> 13 线实测：三档中位输入 tok 完全相同（均 43,414），但 4 路 TTFT 30.13s /\n"
               "> prefill 1,441，8 路 TTFT 6.74s / prefill 6,445——且 4 路的 10 个批次逐批一致\n"
               "> （TTFT 22~34s、prefill 1,27x~1,97x），不是抖动。**这一列只能同档跨线比。**\n")
    out.append("\n> ⚠ **「聚合 prefill tok/s」不是 prefill 算力**：分子是全额 `prompt_tok`，\n"
               "> 其中命中前缀缓存的部分根本没有被重新计算。S3 改为并发各路共用同一仓库前缀后，\n"
               "> 命中率升到 92~94%，批次墙钟大幅缩短而分子不变，这个数会成倍虚高。\n"
               "> 它只能读作「整机每秒能**消化**多少输入 token（含命中）」，即用户视角的吞吐；\n"
               "> 要真实 prefill 算力，用 `cachestat.py` 的 `#new-token` 除以同一窗口墙钟。\n")
    out.append("\n| 并发 | 样本 | 单请求 prefill P50 | 单请求 decode P50 "
               "| **聚合 prefill tok/s** | **聚合 decode tok/s** | **聚合合计 tok/s** "
               "| TTFT P50 | prompt:output |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted({r.get("conc") for r in sub if r.get("conc")}):
        b = [r for r in sub if r.get("conc") == k]
        pr = [x for x in (prefill_rate(r) for r in b) if x]
        dr = [decode_rate(r) for r in b]
        tt = [r["ttft"] for r in b]
        ptot = sum(r["prompt_tok"] for r in b)
        ctot = sum(r.get("completion_tok", 0) for r in b)
        span = batch_span(b)
        ratio = ptot / max(1, ctot)
        out.append(f"| {k} 路 | {len(b)} | {f(pct(pr,.5),0)} | {f(pct(dr,.5))} "
                   f"| **{ptot/span:,.0f}** | **{ctot/span:,.0f}** | **{(ptot+ctot)/span:,.0f}** "
                   f"| {f(pct(tt,.5),2)}s | {ratio:.0f}:1 |")
    if not any(r.get("conc") for r in sub):
        pr = [x for x in (prefill_rate(r) for r in sub) if x]
        dr = [decode_rate(r) for r in sub]
        out.append(f"| 混合 | {len(sub)} | **{f(pct(pr,.5),0)}** | **{f(pct(dr,.5))}** | "
                   f"{f(pct([r['ttft'] for r in sub],.5),2)}s | "
                   f"{sum(r['prompt_tok'] for r in sub)/max(1,sum(r['completion_tok'] for r in sub)):.0f}:1 |")


def section_drift(rows, out):
    if not rows:
        return
    t0 = min(r["ts"] for r in rows)
    out.append("\n## 四、随时间的退化检测（每小时分桶）\n")
    out.append("| 时段 | 请求 | prefill tok/s P50 | **decode tok/s P50** | TTFT P50 |")
    out.append("|---|---|---|---|---|")
    span = max(r["ts"] for r in rows) - t0
    for h in range(int(span // 3600) + 1):
        b = [r for r in rows if h * 3600 <= r["ts"] - t0 < (h + 1) * 3600]
        if not b:
            continue
        pr = [x for x in (prefill_rate(r) for r in b) if x]
        dr = [decode_rate(r) for r in b if r.get("tpot")]
        out.append(f"| 第 {h+1} 小时 | {len(b)} | {f(pct(pr,.5),0)} | **{f(pct(dr,.5))}** "
                   f"| {f(pct([r['ttft'] for r in b],.5),2)}s |")


def report(path, title, until=None):
    raw = load(path, until)
    rows = ok_rows(raw)
    svc_err, beh = split_bad(raw)
    span = (max(r["ts"] for r in raw) - min(r["ts"] for r in raw)) / 3600 if raw else 0
    out = [f"# {title}\n"]
    out.append(f"总请求 **{len(raw)}**，成功 {len(rows)}；"
               f"**服务错误 {len(svc_err)} 次（错误率 {len(svc_err)/max(1,len(raw))*100:.2f}%）**；"
               f"场景未达成 {len(beh)} 次（模型行为结果，非故障）；"
               f"跨度 **{span:.2f} 小时**\n")
    out.append(f"输入合计 **{sum(r['prompt_tok'] for r in rows):,}** tok，"
               f"输出合计 **{sum(r['completion_tok'] for r in rows):,}** tok\n")
    section_single(rows, out)
    section_prefill(rows, out)
    section_decode(rows, out)
    section_concurrency(rows, out)
    section_drift(rows, out)
    from collections import Counter
    out.append("\n## 五、错误\n")
    if svc_err:
        out.append("**服务错误**（真故障）：\n")
        for e, n in Counter(str(r.get("err"))[:70] for r in svc_err).most_common(8):
            out.append(f"- `{e}` × {n}")
    else:
        out.append("**服务错误：0** ✅\n")
    if beh:
        out.append("\n场景未达成（模型未在限定轮数内完成动作，属行为结果非故障）：\n")
        for e, n in Counter(str(r.get("err")) for r in beh).most_common(5):
            out.append(f"- `{e}` × {n}")
    return "\n".join(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    until = None
    if "--until" in a:
        i = a.index("--until"); until = a[i + 1]; a = a[:i] + a[i + 2:]
    if "--compare" in a:
        i = a.index("--compare")
        files, names = a[:i], a[i + 1:]
        for fpath, nm in zip(files, names):
            print(report(fpath, nm, until))
            print("\n---\n")
    else:
        print(report(a[0], a[1] if len(a) > 1 else "基准报告", until))
