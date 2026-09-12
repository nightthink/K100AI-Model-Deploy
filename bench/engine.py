# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""流式请求引擎 + 指标采集（对齐 vLLM bench serve 的指标定义）。

指标定义（与 vLLM 官方一致，便于横向对比）：
  TTFT  首 token 延迟 = 发出请求 → 收到第一个流式输出
  ITL   token 间延迟 = 相邻两次流式输出之间的间隔（逐个记录）
  TPOT  每输出 token 时间 = (E2EL - TTFT) / (输出 token 数 - 1)
  E2EL  端到端延迟 = 发出请求 → 流结束
"""
import json, time, urllib.error, urllib.request


class Result:
    __slots__ = ("ok", "err", "ttft", "e2el", "itls", "tpot", "prompt_tok",
                 "completion_tok", "cached_tok", "prompt_chars", "text", "reasoning",
                 "tool_calls", "finish")

    def __init__(self):
        self.ok = False; self.err = None; self.ttft = None; self.e2el = None
        self.itls = []; self.tpot = None; self.prompt_tok = 0
        self.completion_tok = 0; self.cached_tok = None; self.prompt_chars = 0
        self.text = ""; self.reasoning = ""
        self.tool_calls = []; self.finish = None

    def as_row(self, **extra):
        d = {"ok": self.ok, "err": self.err,
             "ttft": round(self.ttft, 4) if self.ttft is not None else None,
             "e2el": round(self.e2el, 4) if self.e2el is not None else None,
             "tpot": round(self.tpot, 5) if self.tpot is not None else None,
             "itl_p50": None, "itl_p99": None,
             "prompt_tok": self.prompt_tok, "completion_tok": self.completion_tok,
             "cached_tok": self.cached_tok, "prompt_chars": self.prompt_chars,
             "n_tool_calls": len(self.tool_calls), "finish": self.finish,
             "out_chars": len(self.text), "reasoning_chars": len(self.reasoning)}
        if self.itls:
            s = sorted(self.itls)
            d["itl_p50"] = round(s[len(s) // 2], 5)
            d["itl_p99"] = round(s[min(len(s) - 1, int(len(s) * 0.99))], 5)
        d.update(extra)
        return d


def stream_chat(url, body, timeout=900):
    """timeout 是**单次 socket 读**的上限，不是总时长。

    ★ 试跑教训：默认 3600s 时曾有一个 S6 请求卡满一小时（服务端无任何错误），
    直接吃掉整轮预算。长任务也不该等一小时——超时就记为失败继续跑，
    比让一个卡死的请求毁掉整个 6 小时测试划算。"""
    """发一次流式 chat 请求，返回 Result。工具调用增量会被拼装还原。"""
    r = Result()
    body = dict(body)
    # 请求的真实字符数（含 system/tool 定义）——报告按字符分桶时不再靠 token×3.6 估算
    try:
        r.prompt_chars = sum(len(m.get("content") or "") for m in body.get("messages", []))
        r.prompt_chars += len(json.dumps(body.get("tools") or [], ensure_ascii=False))
    except Exception:
        r.prompt_chars = 0
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); last = None
    tc_acc = {}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                u = d["usage"]
                r.prompt_tok = u.get("prompt_tokens", r.prompt_tok)
                r.completion_tok = u.get("completion_tokens", r.completion_tok)
                # 前缀缓存命中数：有了它才能把 prefill 速率算在**未命中**的 token 上。
                # SGLang 的 OpenAI 接口目前恒返回 prompt_tokens_details=null，
                # 所以这里只是防御性采集——服务端哪天开始返回就自动生效，
                # 拿不到时保持 None，由报告侧显式标注「冷热未知」而不是猜。
                det = u.get("prompt_tokens_details") or {}
                if isinstance(det, dict) and det.get("cached_tokens") is not None:
                    r.cached_tok = det["cached_tokens"]
            ch = (d.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                r.finish = ch["finish_reason"]
            de = ch.get("delta") or {}
            piece = de.get("content") or ""
            think = de.get("reasoning") or de.get("reasoning_content") or ""
            for tc in (de.get("tool_calls") or []):
                i = tc.get("index", 0)
                slot = tc_acc.setdefault(i, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if piece or think or de.get("tool_calls"):
                now = time.time()
                if r.ttft is None:
                    r.ttft = now - t0
                else:
                    r.itls.append(now - last)
                last = now
                r.text += piece
                r.reasoning += think
        r.e2el = time.time() - t0
        r.tool_calls = [tc_acc[k] for k in sorted(tc_acc)]
        if r.ttft is not None and r.completion_tok > 1:
            r.tpot = (r.e2el - r.ttft) / (r.completion_tok - 1)
        r.ok = r.ttft is not None
        if not r.ok:
            r.err = "no_output"
    except urllib.error.HTTPError as e:
        r.err = f"HTTP{e.code}:{e.read()[:200].decode('utf-8','replace')}"
        r.e2el = time.time() - t0
    except Exception as e:
        r.err = f"{type(e).__name__}:{str(e)[:200]}"
        r.e2el = time.time() - t0
        # ★ 服务不可达时必须退避：否则连接被拒会以毫秒级速度空转。
        #   实测教训：sglang 容器崩溃重启的 8 分钟里，无退避的驱动产生了
        #   10 万条垃圾记录，把真实数据淹没（2026-08-19）。
        if "Connection refused" in r.err or "ConnectionReset" in r.err:
            time.sleep(5)
    return r


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]
