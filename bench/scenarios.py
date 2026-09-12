# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""六个测试场景，逐一对应 opencode 的真实用法。"""
import json, random, re

import corpus, ocproto
from engine import stream_chat

# 由 run_bench.py 在启动时按线档案注入（见 profiles.py）。
# 保留 MODEL 这个模块级名字，是为了让下面所有 "model": MODEL 的写法不必改动。
MODEL = None
_PROFILE = None


def configure(profile):
    """注入线档案。必须在任何场景函数之前调用。"""
    global MODEL, _PROFILE
    _PROFILE = dict(profile)
    MODEL = _PROFILE["model"]


def _ctk_off():
    """「关闭思考」的请求片段——键名与语义都随模型而变，故由档案给出。

    Qwen3 族看 `enable_thinking`；DeepSeek-V4 看 `thinking` 且是 explicit 语义
    （不显式传就不思考），所以它的 thinking_off 是空字典，此时**什么都不传**，
    而不是传一个空的 chat_template_kwargs。

    S5/S6 需要关思考的原因与模型无关：带 opencode 系统提示测长文会 content=0
    （预算全进 reasoning，已实测复现），所以场景只管「要关」，怎么关看档案。
    """
    off = (_PROFILE or {}).get("thinking_off") or {}
    return {"chat_template_kwargs": off} if off else {}


def _samp():
    """该线的采样侧硬边界（见 profiles.py 的 sampling_override）。

    必须**放在 temperature 之后**合并，因为它的作用就是覆盖场景自带的采样参数。
    对没有这类边界的线是空字典，不影响任何行为。
    """
    return (_PROFILE or {}).get("sampling_override") or {}


def _base(msgs, tools=True, max_tokens=1024, temperature=0.7):
    b = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
         "temperature": temperature}
    b.update(_samp())
    if tools:
        b["tools"] = ocproto.TOOLS
        b["tool_choice"] = "auto"
    return b



def _valid_tool_calls(tool_calls, prefix):
    """回灌前过滤掉参数不是合法 JSON 的工具调用。

    ★ 试跑教训：模型的 tool_call arguments 可能被 max_tokens 截断成半截 JSON，
    原样回灌会让服务端直接 HTTP 400（Expecting ',' delimiter）。
    真实客户端也会丢弃或修复，所以这里丢弃并计数，而不是把 400 记成服务故障。
    """
    good, dropped = [], 0
    for k, tc in enumerate(tool_calls):
        try:
            json.loads(tc["args"] or "{}")
        except json.JSONDecodeError:
            dropped += 1
            continue
        good.append((tc["id"] or f"{prefix}_{k}", tc))
    return good, dropped


def _repo_context(files, budget_chars):
    """按 opencode read 的 cat -n 格式拼装仓库上下文，直到达到预算。"""
    parts, n = [], 0
    for p, c in files:
        blk = corpus.as_cat_n(p, c)
        if n + len(blk) > budget_chars:
            break
        parts.append(blk); n += len(blk)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- S1 多轮会话
def s1_agentic_session(url, files, sess_id, rounds=6, ctx_chars=90000, log=None):
    """单人编程会话：长仓库上下文 + 多轮工具循环。opencode 最核心的模式。

    关键点：每轮把上一轮的 assistant 与 tool 消息追加进去，前缀单调增长——
    这正是前缀缓存该发挥作用的形态，也是命中率随轮次上升的原因。
    """
    ctx = _repo_context(files, ctx_chars)
    msgs = [{"role": "system", "content": ocproto.SYSTEM_PROMPT},
            {"role": "user", "content":
                f"<project_context session={sess_id}>\n{ctx}\n</project_context>\n\n"
                "请先通读上面的仓库上下文，然后帮我做一件事：给 scheduler 模块加上"
                "指数退避重试，并确保和 storage 模块现有的重试风格一致。先说计划。"}]
    tasks = [
        "很好，请用 grep 找出所有已经实现了重试逻辑的位置，我要看现状。",
        "读一下你找到的第一个文件，确认它的退避实现细节。",
        "现在把 scheduler 的对应函数改成同样的风格，给出 edit 调用。",
        "改完了。请检查是否有别的调用方依赖旧的重试次数常量。",
        "最后总结这次改动的影响面，一句话。",
    ]
    out = []
    for i in range(rounds):
        r = stream_chat(url, _base(msgs, max_tokens=900))
        if log:
            log(r.as_row(scen="S1", sess=sess_id, rnd=i, ctx_chars=ctx_chars))
        out.append(r)
        if not r.ok:
            break
        # 把 assistant 回复与（若有）工具结果追加，模拟 agentic 循环
        good, dropped = _valid_tool_calls(r.tool_calls, f"call_{i}")
        am = {"role": "assistant", "content": r.text or None}
        if good:
            am["tool_calls"] = [{"id": cid, "type": "function",
                                 "function": {"name": tc["name"], "arguments": tc["args"]}}
                                for cid, tc in good]
        msgs.append(am)
        for cid, tc in good:
            msgs.append(ocproto.tool_result_msg(
                cid, tc["name"], _fake_tool_result(tc["name"], tc["args"], files)))
        if dropped and log:
            log({"scen": "S1", "sess": sess_id, "rnd": i, "ok": True,
                 "kind": "note", "note": "dropped_bad_tool_json", "n": dropped})
        if i < len(tasks):
            msgs.append({"role": "user", "content": tasks[i]})
    return out


def _fake_tool_result(name, args, files):
    """构造以假乱真的工具返回值（体量与真实相当，这会显著影响下一轮的上下文长度）。"""
    try:
        a = json.loads(args) if args else {}
    except json.JSONDecodeError:
        a = {}
    if name == "read":
        p = a.get("filePath", "") or files[0][0]
        for fp, c in files:
            if fp.split("/")[-1] in p or p.endswith(fp):
                return corpus.as_cat_n(fp, c, limit=400)
        return corpus.as_cat_n(files[1][0], files[1][1], limit=400)
    if name in ("grep", "glob"):
        hits = [f"{p}:{random.randint(10,300)}" for p, _ in files[:24]]
        return "Found %d matches\n" % len(hits) + "\n".join(hits)
    if name == "list":
        return "\n".join(p for p, _ in files[:40])
    if name == "bash":
        return "exit 0\n" + "\n".join(f"ok {i}" for i in range(20))
    if name == "edit":
        return "The file has been updated successfully."
    if name == "todowrite":
        return "Todos updated."
    return "OK"


# ------------------------------------------------------------- S2 长上下文冷启
def s2_long_context(url, files, tag, target_chars, log=None):
    """大仓库首次进入：冷启 TTFT + needle 准确率。"""
    ctx = _repo_context(files, target_chars)
    needle = "DEPLOY_KEY_SENTINEL = \"XK-7742-QR\"  # 部署密钥，勿删"
    pos = int(len(ctx) * 0.68)
    ctx = ctx[:pos] + "\n" + needle + "\n" + ctx[pos:]
    msgs = [{"role": "system", "content": ocproto.SYSTEM_PROMPT},
            {"role": "user", "content":
                f"<project_context tag={tag}>\n{ctx}\n</project_context>\n\n"
                "上文里 DEPLOY_KEY_SENTINEL 的值是什么？只回答值本身。"}]
    r = stream_chat(url, _base(msgs, max_tokens=120))
    hit = "XK-7742-QR" in (r.text + r.reasoning)
    if log:
        log(r.as_row(scen="S2", tag=tag, ctx_chars=target_chars, needle_hit=hit))
    return r, hit


# ------------------------------------------------------------- S3 并发混合负载
def s3_concurrent(url, files, n, log=None, ctx_chars=40000):
    """多人同时用：**共享同一仓库前缀**，量聚合吞吐与分位数。

    ★ 2026-09-11 改：原实现让每一路的前缀都不同——① 语料按 i 轮转
    （`files[i%len:]+files[:i%len]`）；② `<project_context user={i}>` 把 i 写进
    前缀最前面。结果 16 路 = 16 个互不共享的前缀，每轮循环还重来，必然大量冷启。
    实测三条线的缓存命中率只有 52~70%，而生产实态是同一仓库上下文被反复复用、
    命中率约 95%——口径不符，prefill 数字失真（命中的 token 根本不需重算）。
    现改为：**语料顺序固定、前缀不含 i**，区分只放在尾部提问里，
    这样除首批外全部命中，与真实 opencode 一致。
    """
    import concurrent.futures as cf
    ctx = _repo_context(files, ctx_chars)          # 所有并发路共用，前缀可命中
    def one(i):
        msgs = [{"role": "system", "content": ocproto.SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"<project_context>\n{ctx}\n</project_context>\n\n"
                    f"用户 {i} 的问题：这个仓库里 {corpus.MODULES[i%len(corpus.MODULES)]} "
                    "模块的职责是什么？它和哪些模块有耦合？简要回答。"}]
        r = stream_chat(url, _base(msgs, max_tokens=400))
        if log:
            log(r.as_row(scen="S3", conc=n, worker=i, ctx_chars=ctx_chars))
        return r
    with cf.ThreadPoolExecutor(n) as ex:
        return list(ex.map(one, range(n)))


# ------------------------------------------------------------ S4 工具调用密集
def s4_tool_pressure(url, files, log=None):
    """agentic 压力：验证 tool_calls 能否被正确解析、参数是否合法 JSON、能否并行调用。"""
    ctx = _repo_context(files, 30000)
    probes = [
        # 注意：必须问上下文里没有的文件，否则好的 agent 会直接回答而不调工具
        ("单工具", "读取 /etc/app/production.yaml 这个配置文件，我要看它的超时设置。"),
        ("并行多工具", "同时做三件事：用 glob 找出所有 .ts 文件、用 grep 搜索 MAX_RETRIES、"
                       "列出 src 目录。请在一轮里并行调用。"),
        ("带复杂参数", "把 src/billing/billing_1.py 里所有的 DEFAULT_TIMEOUT_S = 30 "
                       "替换成 DEFAULT_TIMEOUT_S = 60。必须用 edit 工具，replaceAll 打开。"),
        ("需选择工具", "我想知道哪个文件定义了 SchedulerLease 类，你会怎么查？直接调工具。"),
        ("任务规划", "把'给 storage 加缓存层'拆成 4 个待办项并写入 todo。"),
    ]
    out = []
    for name, q in probes:
        msgs = [{"role": "system", "content": ocproto.SYSTEM_PROMPT},
                {"role": "user", "content": f"<project_context>\n{ctx}\n</project_context>\n\n{q}"}]
        r = stream_chat(url, _base(msgs, max_tokens=700))
        valid = 0
        for tc in r.tool_calls:
            try:
                json.loads(tc["args"] or "{}"); valid += 1
            except json.JSONDecodeError:
                pass
        if log:
            log(r.as_row(scen="S4", probe=name, n_valid_json=valid,
                         parallel=len(r.tool_calls) > 1))
        out.append((name, r, valid))
    return out


# ------------------------------------------------------------------- S5 PPT
MARP_RE = re.compile(r"^---\s*$", re.M)

PPT_SPECS = {
    "marp": ("slides.md",
             "写一份 Marp 格式的技术分享 PPT（Markdown）：封面 + 至少 8 页内容 + 总结页；"
             "每页有标题和要点；含一页 mermaid 架构图；含一页代码示例。用 --- 分页。"),
    "pptx": ("gen_slides.py",
             "写一段完整可运行的 python-pptx 脚本，生成 10 页技术分享 PPT："
             "标题页 + 目录页 + 8 页内容页。需使用 Presentation()、slide_layouts、"
             "placeholders，并 save() 到文件。"),
    "revealjs": ("slides.html",
                 "生成一个单文件 reveal.js 风格 HTML 演示文稿，至少 8 张幻灯片，"
                 "内联 CSS 与 JS，不依赖外部 CDN。"),
}


def _check_ppt(kind, body):
    """对产物做结构检查。body 为 write 工具的 content 参数或直接正文。"""
    body = _unfence(body or "")
    c = {"产物字符数": len(body)}
    if kind == "marp":
        c["页数"] = sum(1 for l in body.split("\n") if l.strip() == "---")
        c["有mermaid"] = "mermaid" in body
        c["有代码块"] = body.count("```") >= 2
        c["结构合格"] = c["页数"] >= 8 and len(body) > 800
    elif kind == "pptx":
        c["有Presentation"] = "Presentation(" in body
        c["有slide_layouts"] = "slide_layouts" in body
        c["有save"] = ".save(" in body
        c["语法合格"] = _py_parses(body)
        c["结构合格"] = c["有Presentation"] and c["有save"] and c["语法合格"]
    else:
        low = body.lower()
        c["有html标签"] = "<html" in low
        c["section数"] = low.count("<section")
        c["无外部CDN"] = ("http://" not in body and "https://" not in body)
        c["结构合格"] = c["section数"] >= 8 and c["有html标签"]
    return c


def s5_ppt(url, files, log=None, max_rounds=4):
    """PPT 生成（agentic 形态）。

    ★ 为什么必须走 agentic：opencode 的系统提示要求"少于 4 行文本输出"，
    模型不会在对话里打印 PPT，而是**先用 list/glob/read 探索、再用 write 落文件**。
    实测对照：同一请求下「opencode 提示 + 带工具」只输出 61 token 并调 list/glob；
    换中性提示才会直接吐 5,977 字符 HTML。所以按对话长文来测 PPT 是测错了对象。

    本场景量三件事：
      1. 能否在 max_rounds 轮内产出 write 调用（agentic 收敛性）
      2. write 出来的产物结构是否合格
      3. 端到端耗时与 token
    另跑一条「中性提示直出」作为纯生成能力基线，与 agentic 路径对照。
    """
    ctx = _repo_context(files, 25000)
    out = []
    for kind, (fname, spec) in PPT_SPECS.items():
        msgs = [{"role": "system", "content": ocproto.SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"<project_context>\n{ctx}\n</project_context>\n\n"
                    f"{spec}\n把结果写到 {fname}。"}]
        artifact = None
        rounds_used = 0
        t_all = 0.0
        tok_all = 0
        for i in range(max_rounds):
            r = stream_chat(url, _base(msgs, max_tokens=6000))
            rounds_used = i + 1
            t_all += (r.e2el or 0)
            tok_all += r.completion_tok
            if not r.ok:
                break
            for tc in r.tool_calls:
                if tc["name"] == "write":
                    try:
                        artifact = json.loads(tc["args"]).get("content", "")
                    except json.JSONDecodeError:
                        artifact = tc["args"]      # JSON 坏也留证据
            if artifact:
                break
            good, _drop = _valid_tool_calls(r.tool_calls, f"w{i}")
            am = {"role": "assistant", "content": r.text or None}
            if good:
                am["tool_calls"] = [{"id": cid, "type": "function",
                                     "function": {"name": tc["name"], "arguments": tc["args"]}}
                                    for cid, tc in good]
            msgs.append(am)
            for cid, tc in good:
                msgs.append(ocproto.tool_result_msg(
                    cid, tc["name"], _fake_tool_result(tc["name"], tc["args"], files)))
            if not r.tool_calls:
                msgs.append({"role": "user", "content": f"请直接用 write 工具写到 {fname}。"})
        chk = _check_ppt(kind, artifact) if artifact else {"结构合格": False, "产物字符数": 0}
        if log:
            log({"scen": "S5", "kind": kind, "mode": "agentic", "ok": artifact is not None,
                 "err": None if artifact else "no_write_call",
                 "rounds": rounds_used, "e2el": round(t_all, 3),
                 "completion_tok": tok_all, "prompt_tok": 0, "ttft": None, "tpot": None,
                 **{f"chk_{k}": v for k, v in chk.items()}})
        out.append((kind, artifact, chk))

        # 纯生成能力基线：中性提示、无工具、关思考（避免思考吃光预算）
        r2 = stream_chat(url, {"model": MODEL, "temperature": 0.7, "max_tokens": 6000,
                               **_ctk_off(), **_samp(),
                               "messages": [
                                   {"role": "system", "content": "你是一个资深技术专家，善于制作演示文稿。"},
                                   {"role": "user", "content":
                                       f"<project_context>\n{ctx}\n</project_context>\n\n{spec}\n直接输出完整内容，不要解释。"}]})
        chk2 = _check_ppt(kind, r2.text)
        if log:
            log(r2.as_row(scen="S5", kind=kind, mode="direct",
                          **{f"chk_{k}": v for k, v in chk2.items()}))
    return out


def _unfence(text):
    """模型常把整份产物包在 ```lang ... ``` 里，去掉最外层围栏再做结构检查。"""
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", t, re.S)
    return m.group(1) if m else (text or "")


def _py_parses(text):
    import ast
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    code = m.group(1) if m else (text or "")
    try:
        ast.parse(code); return True
    except (SyntaxError, ValueError):
        return False


# --------------------------------------------------------------- S6 长输出
def s6_long_output(url, files, log=None):
    """8K token 长输出（要求⑥的输出上限），验证长生成不中断、不退化。

    ★ 必须用中性系统提示 + 关思考：opencode 的系统提示明确要求"少于 4 行文本输出"，
    带着它测长文会得到 content=0（全部预算进了 reasoning）——实测已复现。
    在真实 opencode 里，大段内容是通过 write 工具落文件的，不走对话正文。
    """
    msgs = [{"role": "system", "content": "你是一位资深分布式系统架构师，擅长撰写详尽的设计文档。"},
            {"role": "user", "content":
                "请写一份非常详尽的《分布式任务调度系统设计文档》，"
                "包含：背景与目标、总体架构、核心数据结构、状态机、"
                "一致性与容错、限流与配额、可观测性、部署拓扑、演进路线、附录 API 表。"
                "每章都要展开，不要省略。目标 8000 字以上。"}]
    r = stream_chat(url, {"model": MODEL, "messages": msgs, "temperature": 0.7,
                          "max_tokens": 8192,
                          **_ctk_off(), **_samp()},
                    timeout=600)
    if log:
        log(r.as_row(scen="S6", reached_8k=(r.completion_tok >= 8000),
                     out_is_empty=(len(r.text) == 0)))
    return r
