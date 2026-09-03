"""
DP2×TP4 均衡粘性路由：单一入口 8100 → 两个 TP4 副本（8101/8102）。

两个目标同时满足：
  * 粘性 —— 两个副本各有独立前缀缓存，同一会话必须永远落到同一副本，
            否则长前缀在两边各建一份，命中率腰斩。
  * 均衡 —— 纯哈希分流的偏斜是真实代价：实测 32 并发下哈希可能分成
            11:21，而每副本 --max-num-seqs 16，多出来的请求要排下一波，
            聚合从 612 掉到 393。所以**新会话按当前在飞请求数分配给最闲的副本**，
            一旦分配即固定。

粘性键优先级（取第一个存在的）：
  1. X-Session-Id 请求头（客户端显式指定，最可靠）
  2. body.user（OpenAI 标准字段）
  3. Authorization 头
  4. 第一条 system + 第一条 user 的前 512 字符
     —— 编程助手同一项目/会话共享这段长前缀，天然同键

会话表 LRU 上限 4096；上游连接失败时自动改派另一副本并重绑会话。
必须单 worker 运行（会话表与在飞计数是进程内状态）。
"""
import hashlib, itertools, json, os
from collections import OrderedDict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAMS = os.environ.get("UPSTREAMS", "http://127.0.0.1:8101,http://127.0.0.1:8102").split(",")
MAX_SESSIONS = int(os.environ.get("ROUTER_MAX_SESSIONS", "4096"))
TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=600.0, pool=10.0)

app = FastAPI()
client: httpx.AsyncClient = None
SESSIONS: "OrderedDict[str, str]" = OrderedDict()
INFLIGHT = {u: 0 for u in UPSTREAMS}
STATS = {u: 0 for u in UPSTREAMS}
_rr = itertools.cycle(range(len(UPSTREAMS)))


@app.on_event("startup")
async def _startup():
    global client
    client = httpx.AsyncClient(timeout=TIMEOUT, limits=httpx.Limits(max_connections=512))


def sticky_key(headers, body):
    if headers.get("x-session-id"):
        return headers["x-session-id"]
    if isinstance(body, dict) and body.get("user"):
        return str(body["user"])
    if headers.get("authorization"):
        return headers["authorization"]
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        parts = []
        for role in ("system", "user"):
            for m in body["messages"]:
                if m.get("role") == role:
                    c = m.get("content")
                    parts.append(c if isinstance(c, str) else json.dumps(c, sort_keys=True))
                    break
        if parts:
            return ("\n".join(parts))[:512]
    if isinstance(body, dict) and isinstance(body.get("prompt"), str):
        return body["prompt"][:512]
    return None


def least_loaded():
    """在飞最少者优先；打平时按轮转顺序选，保证真正轮流。

    不要写成 min(UPSTREAMS, key=lambda u: (INFLIGHT[u], next(_rr)))：
    min 会对每个候选各调一次 next()，而 cycle 周期恰好等于副本数，
    相位被锁死后永远选中同一个副本（实测 8 个会话全落单边）。
    正确做法是每次只取一个偏移量，用它轮转候选顺序。
    """
    off = next(_rr)
    order = UPSTREAMS[off:] + UPSTREAMS[:off]
    return min(order, key=lambda u: INFLIGHT[u])


def assign(key):
    """返回该会话应落的副本；新会话分给最闲的，老会话保持不变。"""
    if key is None:
        return least_loaded()
    up = SESSIONS.get(key)
    if up is None:
        up = least_loaded()
        SESSIONS[key] = up
        if len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
    else:
        SESSIONS.move_to_end(key)
    return up


@app.get("/router/stats")
async def stats():
    return {"upstreams": UPSTREAMS, "dispatched": STATS,
            "inflight": INFLIGHT, "sessions": len(SESSIONS)}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    raw = await request.body()
    body = None
    if raw:
        try:
            body = json.loads(raw)
        except Exception:
            body = None

    inference = path in ("v1/chat/completions", "v1/completions")
    key = sticky_key({k.lower(): v for k, v in request.headers.items()}, body) if inference else None
    up = assign(key) if inference else UPSTREAMS[next(_rr)]

    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    stream = isinstance(body, dict) and body.get("stream") is True

    held = up                 # 计数只认这一个键，改派时不改它，避免加减不配对
    INFLIGHT[held] += 1
    STATS[up] = STATS.get(up, 0) + 1
    try:
        if stream:
            req = client.build_request(request.method, f"{up}/{path}", content=raw,
                                       headers=headers, params=request.query_params)
            resp = await client.send(req, stream=True)

            async def gen():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
                    INFLIGHT[held] -= 1

            return StreamingResponse(gen(), status_code=resp.status_code,
                                     media_type=resp.headers.get("content-type", "text/event-stream"),
                                     headers={"X-Upstream": up})
        try:
            resp = await client.request(request.method, f"{up}/{path}", content=raw,
                                        headers=headers, params=request.query_params)
        except httpx.HTTPError:
            # 上游不可达：改派另一副本并重绑会话（缓存局部性让位于可用性）
            alt = [u for u in UPSTREAMS if u != up]
            if not alt:
                raise
            up2 = alt[0]
            if key is not None:
                SESSIONS[key] = up2
            INFLIGHT[up2] += 1
            try:
                resp = await client.request(request.method, f"{up2}/{path}", content=raw,
                                            headers=headers, params=request.query_params)
            finally:
                INFLIGHT[up2] -= 1
            up = up2
        out = {k: v for k, v in resp.headers.items()
               if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}
        out["X-Upstream"] = up
        return Response(content=resp.content, status_code=resp.status_code, headers=out,
                        media_type=resp.headers.get("content-type"))
    except httpx.HTTPError as e:
        return JSONResponse({"error": {"message": f"upstream unreachable: {e}"}}, status_code=502)
    finally:
        if not stream:
            INFLIGHT[held] -= 1
