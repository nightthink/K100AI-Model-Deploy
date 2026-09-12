#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""opencode 编程后端 · 长时综合基准（替代原 1 小时稳定性）

用法:
  python3 run_bench.py --profile qwen3.8-27b --container <容器名> \
                       --port 8100 --hours 6 --out ./bench_out

  多条线串行请用编排器：bash run_lines.sh <profile> <线清单文件> [每线小时数]
  模型相关的差异（model 字段 / thinking 键 / 就绪探针）全在 profiles.py

设计依据（2026-08 网络调研）:
  * opencode 是 agentic loop：~10 内置工具、流式、多轮往返、可并行调工具、
    文件按 cat -n 注入、上下文 90–96% 触发压缩          （cefboud.com 深度拆解）
  * 指标对齐 vLLM bench serve：TTFT / ITL / TPOT / E2EL 的分位数
  * PPT 用 Marp Markdown 最适合 LLM 生成，python-pptx / reveal.js 为另两条路

与旧 stability.sh 的区别：
  旧的发的是无意义中文长文本、只量单流均值。本套用**真实代码语料**、
  复刻 opencode 的**系统提示 + 10 个工具定义 + 多轮工具循环**，
  并按分位数记录，同时覆盖 PPT 生成与工具调用正确性。
"""
import argparse, json, os, random, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus, profiles, scenarios
from engine import pct


def now():
    return time.strftime("%H:%M:%S")


class Recorder:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.n = 0; self.err = 0

    def __call__(self, row):
        row["ts"] = time.time()
        row["t"] = now()
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # probe 行是服务端状态采样，没有 ok 字段；行为结果（模型没调 write 之类）
        # 也不是服务故障。两者都不能计入错误数，否则实时日志会虚高。
        if row.get("kind") == "probe":
            return
        self.n += 1
        if not row.get("ok") and str(row.get("err")) not in ("no_write_call", "no_tool_call", "no_output"):
            self.err += 1

    def close(self):
        self.f.close()


def sample_server(port, out, container):
    """周期采样服务端状态：前缀缓存命中率、GPU 显存、宿主内存。"""
    row = {"ts": time.time(), "t": now(), "kind": "probe"}
    try:
        lg = subprocess.run(["docker", "logs", "--tail", "40", container],
                            capture_output=True, timeout=30)
        txt = (lg.stdout + lg.stderr).decode("utf-8", "replace").replace("\r", "\n")
        for ln in reversed(txt.split("\n")):
            if "Prefix cache hit rate" in ln:
                try:
                    row["prefix_hit_rate"] = float(ln.split("Prefix cache hit rate:")[1]
                                                   .strip().rstrip("%"))
                except (IndexError, ValueError):
                    pass
                if "GPU KV cache usage" in ln:
                    try:
                        row["kv_usage"] = float(ln.split("GPU KV cache usage:")[1]
                                                .split("%")[0].strip())
                    except (IndexError, ValueError):
                        pass
                break
    except Exception as e:
        row["probe_err"] = str(e)[:120]
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable"):
                    row["host_mem_avail_gb"] = round(int(ln.split()[1]) / 1048576, 1)
                    break
    except OSError:
        pass
    out(row)


def summarize(jsonl, hours):
    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    # note 行是辅助记录（如丢弃坏 JSON 的计数），不是请求，必须排除，
    # 否则它们带着 scen 却没有 prompt_tok / ttft 等字段，会让后面的统计炸掉。
    reqs = [r for r in rows if r.get("kind") not in ("probe", "note")]
    probes = [r for r in rows if r.get("kind") == "probe"]
    notes = [r for r in rows if r.get("kind") == "note"]
    ok = [r for r in reqs if r.get("ok")]
    # ★ 区分两类"失败"：服务错误 vs 场景未达成。
    #   no_write_call 表示模型在限定轮数内没调 write —— 那是被测出来的行为结果，
    #   不是服务故障。混在一起会把健康的服务报成"失败率 5%"。
    SCEN_OUTCOMES = {"no_write_call"}
    svc_fail = [r for r in reqs if not r.get("ok") and str(r.get("err")) not in SCEN_OUTCOMES]
    scen_miss = [r for r in reqs if not r.get("ok") and str(r.get("err")) in SCEN_OUTCOMES]
    L = []
    A = L.append
    A(f"# opencode 编程后端 · {hours} 小时综合基准报告\n")
    A(f"总请求 **{len(reqs)}**；**服务错误 {len(svc_fail)} 次"
      f"（错误率 {100*len(svc_fail)/max(len(reqs),1):.2f}%）**；"
      f"场景未达成 {len(scen_miss)} 次（模型未在限定轮数内调 write，属行为结果非故障）\n")
    if reqs:
        span = (max(r["ts"] for r in reqs) - min(r["ts"] for r in reqs)) / 3600
        A(f"实际跨度 **{span:.2f} 小时**；"
          f"输出 token 合计 **{sum(r.get('completion_tok',0) for r in ok):,}**\n")

    A("\n## 一、按场景的延迟分位数\n")
    A("| 场景 | 请求数 | 成功率 | TTFT P50 | P95 | P99 | TPOT P50 | P95 | E2EL P50 |")
    A("|---|---|---|---|---|---|---|---|---|")
    names = {"S1": "S1 多轮编程会话", "S2": "S2 长上下文冷启", "S3": "S3 并发混合",
             "S4": "S4 工具调用密集", "S5": "S5 PPT 生成", "S6": "S6 长输出"}
    for s in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        g = [r for r in reqs if r.get("scen") == s]
        go = [r for r in g if r.get("ok")]
        if not g:
            continue
        tt = [r.get("ttft") for r in go if r.get("ttft") is not None]
        tp = [r.get("tpot") for r in go if r.get("tpot") is not None]
        ee = [r.get("e2el") for r in go if r.get("e2el") is not None]
        f2 = lambda v: f"{v:.2f}s" if v is not None else "—"
        f4 = lambda v: f"{v*1000:.1f}ms" if v is not None else "—"
        A(f"| {names[s]} | {len(g)} | {100*len(go)/len(g):.1f}% | "
          f"{f2(pct(tt,50))} | {f2(pct(tt,95))} | {f2(pct(tt,99))} | "
          f"{f4(pct(tp,50))} | {f4(pct(tp,95))} | {f2(pct(ee,50))} |")

    A("\n## 二、S1 多轮会话：缓存随轮次的效果\n")
    A("| 轮次 | 样本 | TTFT P50 | TTFT P95 | prompt tok 中位 |")
    A("|---|---|---|---|---|")
    for rnd in range(8):
        g = [r for r in reqs if r.get("scen") == "S1" and r.get("rnd") == rnd and r.get("ok")]
        if not g:
            continue
        tt = [r.get("ttft") for r in g if r.get("ttft") is not None]
        pt = [r.get("prompt_tok", 0) for r in g]
        if not tt:
            continue
        A(f"| 第 {rnd+1} 轮 | {len(g)} | {pct(tt,50):.2f}s | {pct(tt,95):.2f}s | "
          f"{pct(pt,50):,} |")

    A("\n## 三、S3 并发：真实 opencode 负载下的吞吐\n")
    A("★ 关键：真实负载是 **prefill 主导**的——每请求带约 11K token 仓库上下文，"
      "而输出只有几百 token。只看\"输出 tok/s\"会严重低估系统实际做的功，"
      "所以这里同时给出 prefill 吞吐与合计吞吐。\n")
    A("| 并发 | 批次 | 成功率 | TTFT P50 | P99 | 单路 tok/s | prefill tok/s | 输出 tok/s | **合计 tok/s** | prompt:output |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted({r.get("conc") for r in reqs if r.get("scen") == "S3" and r.get("conc")}):
        g = [r for r in reqs if r.get("scen") == "S3" and r.get("conc") == c]
        go = [r for r in g if r.get("ok")]
        if not go:
            A(f"| {c} 路 | {len(g)//c} | 0% | — | — | — | — | — | — | — |")
            continue
        tt = [r.get("ttft") for r in go if r.get("ttft") is not None]
        tp = [r.get("tpot") for r in go if r.get("tpot") is not None]
        # 按批次（同一 conc 的一组）分别算墙钟，再汇总
        pt = sum(r.get("prompt_tok", 0) for r in go)
        ct = sum(r.get("completion_tok", 0) for r in go)
        batches = max(1, len(g) // c)
        span = 0.0
        by_batch = {}
        for r in go:
            by_batch.setdefault(round((r["ts"] - (r.get("e2el") or 0)) / 60), []).append(r)
        for b in by_batch.values():
            span += max(x["ts"] for x in b) - min(x["ts"] - (x.get("e2el") or 0) for x in b)
        span = span or 1.0
        A(f"| {c} 路 | {batches} | {100*len(go)/len(g):.0f}% | "
          f"{pct(tt,50):.2f}s | {pct(tt,99):.2f}s | "
          f"{1/pct(tp,50):.1f} | {pt/span:,.0f} | {ct/span:.0f} | "
          f"**{(pt+ct)/span:,.0f}** | {pt/max(ct,1):.0f}:1 |")

    A("\n## 四、S4 工具调用正确性\n")
    A("| 探针 | 样本 | 成功率 | 平均工具数 | JSON 合法率 | 并行调用占比 |")
    A("|---|---|---|---|---|---|")
    for pr in sorted({r.get("probe") for r in reqs if r.get("scen") == "S4" and r.get("probe")}):
        g = [r for r in reqs if r.get("scen") == "S4" and r.get("probe") == pr]
        go = [r for r in g if r.get("ok")]
        ntc = sum(r.get("n_tool_calls", 0) for r in go)
        nvj = sum(r.get("n_valid_json", 0) for r in go)
        par = sum(1 for r in go if r.get("parallel"))
        A(f"| {pr} | {len(g)} | {100*len(go)/max(len(g),1):.0f}% | "
          f"{ntc/max(len(go),1):.1f} | {100*nvj/max(ntc,1):.0f}% | "
          f"{100*par/max(len(go),1):.0f}% |")

    A("\n## 五、S5 PPT 生成质量\n")
    A("agentic = 真实 opencode 路径（探索后用 write 落文件）；direct = 中性提示直出（纯生成能力基线）。\n")
    A("| 形式 | 路径 | 样本 | 产出率 | 结构合格率 | 产物字符中位 | 平均轮数 | E2EL P50 |")
    A("|---|---|---|---|---|---|---|---|")
    for k in ["marp", "pptx", "revealjs"]:
        for mode in ["agentic", "direct"]:
            g = [r for r in reqs if r.get("scen") == "S5" and r.get("kind") == k
                 and r.get("mode") == mode]
            if not g:
                continue
            go = [r for r in g if r.get("ok")]
            good = sum(1 for r in g if r.get("chk_结构合格"))
            sz = [r.get("chk_产物字符数", 0) for r in g]
            rd = [r.get("rounds", 1) for r in g]
            ee = [r.get("e2el") for r in g if r.get("e2el")]
            A(f"| {k} | {mode} | {len(g)} | {100*len(go)/len(g):.0f}% | "
              f"{100*good/len(g):.0f}% | {pct(sz,50):,} | "
              f"{sum(rd)/len(rd):.1f} | {pct(ee,50):.1f}s |")

    A("\n## 六、S2 长上下文 needle 准确率\n")
    A("| 上下文 | 样本 | needle 命中率 | TTFT P50 | P95 |")
    A("|---|---|---|---|---|")
    for cc in sorted({r.get("ctx_chars") for r in reqs if r.get("scen") == "S2"}):
        g = [r for r in reqs if r.get("scen") == "S2" and r.get("ctx_chars") == cc and r.get("ok")]
        if not g:
            continue
        hits = sum(1 for r in g if r.get("needle_hit"))
        tt = [r.get("ttft") for r in g if r.get("ttft")]
        A(f"| ~{cc//1000}K 字符 | {len(g)} | {100*hits/len(g):.0f}% | "
          f"{pct(tt,50):.1f}s | {pct(tt,95):.1f}s |")

    A("\n## 七、随时间的退化检测（每小时分桶）\n")
    A("| 时段 | 请求 | 失败 | S1 TTFT P50 | S1 TPOT P50 | S1 输出 tok/s | 前缀命中率 |")
    A("|---|---|---|---|---|---|---|")
    if reqs:
        t0 = min(r["ts"] for r in reqs)
        for h in range(int(hours) + 1):
            lo, hi = t0 + h * 3600, t0 + (h + 1) * 3600
            g = [r for r in reqs if lo <= r["ts"] < hi]
            if not g:
                continue
            s1 = [r for r in g if r.get("scen") == "S1" and r.get("ok")]
            tt = [r.get("ttft") for r in s1 if r.get("ttft")]
            tp = [r.get("tpot") for r in s1 if r.get("tpot")]
            pb = [r.get("prefix_hit_rate") for r in probes
                  if lo <= r["ts"] < hi and "prefix_hit_rate" in r]
            # 前缀命中率只有 vLLM 会打进日志，sglang 没有 —— 缺它不该让整行退化成空
            f_tt = f"{pct(tt,50):.2f}s" if tt else "—"
            f_tp = f"{pct(tp,50)*1000:.1f}ms" if tp else "—"
            f_ts = f"{1/pct(tp,50):.1f}" if tp else "—"
            f_pb = f"{sum(pb)/len(pb):.1f}%" if pb else "—"
            A(f"| 第 {h+1} 小时 | {len(g)} | {sum(1 for r in g if not r.get('ok'))} | "
              f"{f_tt} | {f_tp} | {f_ts} | {f_pb} |")

    A("\n## 八、错误分布\n")
    if svc_fail:
        errs = {}
        for r in svc_fail:
            errs[str(r.get("err"))[:70]] = errs.get(str(r.get("err"))[:70], 0) + 1
        A("**服务错误**（真故障）：\n")
        A("| 错误 | 次数 | 场景 |")
        A("|---|---|---|")
        for k, v in sorted(errs.items(), key=lambda x: -x[1]):
            sc = sorted({str(r.get("scen")) for r in svc_fail
                         if str(r.get("err"))[:70] == k})
            A(f"| `{k}` | {v} | {','.join(sc)} |")
    else:
        A("**服务错误：0 次。全程无 HTTP 错误、无超时、无空响应。**\n")
    if notes:
        n_drop = sum(x.get("n", 0) for x in notes if x.get("note") == "dropped_bad_tool_json")
        if n_drop:
            A(f"\n另有 **{n_drop} 个工具调用的参数不是合法 JSON**（被 max_tokens 截断），"
              "回灌前已丢弃——真实客户端同样需要这层防护，否则服务端会返回 HTTP 400。\n")
    if scen_miss:
        A(f"\n场景未达成 {len(scen_miss)} 次（`no_write_call`：模型未在 4 轮内调用 write）——"
          "这是被测出的行为结果，已计入 §五 的产出率，不计作服务故障。\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8100")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--out", default="./bench_out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", required=True,
                    help=f"线档案，决定 model 字段与 thinking 键；可选：{', '.join(profiles.names())}")
    ap.add_argument("--container", required=True,
                    help="服务端容器名，用于采样前缀缓存命中率等（取自容器日志的 Prefill batch 行）")
    a = ap.parse_args()

    # ★ 必须在任何场景函数之前注入：MODEL 与 thinking 键都从这里来。
    #   --profile 与 --container 都不给默认值——给了默认值就会出现
    #   「打错模型名却照跑两小时」这种最贵的失败。
    prof = profiles.get(a.profile)
    scenarios.configure(prof)

    os.makedirs(a.out, exist_ok=True)
    jsonl = os.path.join(a.out, "requests.jsonl")
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    rec = Recorder(jsonl)
    files = corpus.build(seed=a.seed, n_files=60)
    rng = random.Random(a.seed)

    t_end = time.time() + a.hours * 3600
    t_probe = 0
    cyc = 0
    print(f"[{now()}] 基准开始：{a.hours} 小时，输出 {a.out}", flush=True)
    print(f"[{now()}] 语料 {len(files)} 文件 / "
          f"{sum(len(c) for _,c in files):,} 字符", flush=True)

    while time.time() < t_end:
        cyc += 1
        left = (t_end - time.time()) / 3600
        print(f"[{now()}] --- 循环 {cyc}（剩 {left:.2f}h，"
              f"累计 {rec.n} 请求 / {rec.err} 失败）---", flush=True)

        if time.time() - t_probe > 300:
            sample_server(a.port, rec, a.container); t_probe = time.time()

        t_cyc = time.time()
        # ---- 每循环必跑的核心负载（最贴近 opencode 日常）----
        # S1 冷前缀（新会话）→ S1 热前缀（同会话再来）→ S4 工具压力
        # ★ 2026-09-11 按生产实际重配：输入 40K–350K 字符、均值约 150K、
        #   缓存命中率目标 95%（原配置均值仅 68,889 字符，不到生产的一半，
        #   且 S3/S4 占 53% 却都卡在 ~40K，把均值整体拖低）。
        #   命中率靠「每次冷启后跟更多热轮」构造，并由 cachestat.py 从
        #   容器日志的 #new-token/#cached-token 实测校验，不靠假设。
        scenarios.s1_agentic_session(url, files, f"sess{cyc}", rounds=10,
                                     ctx_chars=rng.choice([40000, 90000, 150000,
                                                           250000, 350000]),
                                     log=rec)
        scenarios.s1_agentic_session(url, files, f"sess{cyc}", rounds=6,
                                     ctx_chars=150000, log=rec)
        scenarios.s4_tool_pressure(url, files, log=rec)

        # ---- 重场景四选一轮转 ----
        # ★ 试跑教训：S5 曾独占 85% 的时间（单个 revealjs agentic 用了 22 分钟），
        #   导致 6 小时只能跑 6 个循环、S3/S6 各只覆盖一两次。改为每循环只跑一个重场景，
        #   把循环压到约 15 分钟，让六个场景都拿到足够样本。
        # ★ 2026-09-11：S3（并发）改为**每两个循环跑一次**。
        #   原来 cyc%4 让它只占 1/4 的循环，而新配置把上下文抬到 350K 字符后
        #   每循环更慢——聚合数据反而会最稀。聚合是选型的关键口径，不能最少。
        if cyc % 2 == 0:
            for n in (4, 8, 16):
                scenarios.s3_concurrent(url, files, n, log=rec, ctx_chars=150000)
            heavy = 2
        else:
            heavy = (cyc // 2) % 3
            if heavy == 0:
                for cc in (60000, 180000, 340000):
                    scenarios.s2_long_context(url, files, f"c{cyc}", cc, log=rec)
            elif heavy == 1:
                scenarios.s5_ppt(url, files, log=rec, max_rounds=3)
            else:
                scenarios.s6_long_output(url, files, log=rec)
        # ★ 标签 bug 修正（2026-09-11）：原式 ['S2','S5','S6','S3'][heavy if heavy!=2 else 3]
        #   把**奇数**分支里 heavy==2 的循环也标成 S3，而该分支 heavy==2 跑的是
        #   s6_long_output。实测循环 5 被误标为 S3（真实是 S6），害我把 S6 的命中率
        #   当成 S3 的来分析。只影响控制台标签，不影响 jsonl（scen 由各场景自己打）。
        print(f"[{now()}] 循环 {cyc} 用时 {time.time()-t_cyc:.0f}s"
              f"（重场景 {'S3' if cyc % 2 == 0 else ['S2','S5','S6'][heavy]}）", flush=True)

    sample_server(a.port, rec, a.container)
    rec.close()
    print(f"[{now()}] 结束：{cyc} 循环，{rec.n} 请求，{rec.err} 失败", flush=True)
    md = summarize(jsonl, a.hours)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
