# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""生成贴近真实仓库的代码语料（供 opencode 式基准使用）。

为什么必须用代码而不是中文散文：中文约 1 字/token，代码约 3–4 字符/token，
分词形态、注意力模式、模型负担都不同。用散文测出来的数不能代表编程助手负载。
"""
import hashlib, os, random

MODULES = ["auth", "billing", "scheduler", "storage", "gateway", "telemetry",
           "registry", "planner", "indexer", "replica", "quota", "session"]
VERBS = ["fetch", "resolve", "commit", "reconcile", "evict", "flush", "hydrate",
         "validate", "dispatch", "compact", "snapshot", "rebalance"]
NOUNS = ["Record", "Handle", "Cursor", "Segment", "Lease", "Shard", "Envelope",
         "Manifest", "Checkpoint", "Partition", "Token", "Frame"]


def _py_file(rng, mod, n_cls):
    L = [f'"""{mod} —— 内部服务模块，由基准语料生成器产出。"""',
         "from __future__ import annotations", "",
         "import asyncio", "import logging", "import time",
         "from dataclasses import dataclass, field",
         "from typing import Any, Iterable, Optional", "",
         f"logger = logging.getLogger(__name__)",
         f"DEFAULT_TIMEOUT_S = {rng.randint(5, 120)}",
         f"MAX_RETRIES = {rng.randint(2, 8)}", ""]
    for c in range(n_cls):
        noun = rng.choice(NOUNS)
        cls = f"{mod.capitalize()}{noun}{c}"
        L += ["@dataclass", f"class {cls}:",
              f'    """{mod} 的 {noun} 实体，负责状态流转与一致性校验。"""',
              "    key: str", "    revision: int = 0",
              "    payload: dict[str, Any] = field(default_factory=dict)",
              "    updated_at: float = field(default_factory=time.time)", ""]
        for v in rng.sample(VERBS, k=rng.randint(3, 6)):
            L += [f"    async def {v}(self, ctx: dict[str, Any],",
                  f"                  deadline: Optional[float] = None) -> bool:",
                  f'        """对 {noun} 执行 {v}；失败重试至多 MAX_RETRIES 次。"""',
                  "        deadline = deadline or (time.time() + DEFAULT_TIMEOUT_S)",
                  "        for attempt in range(MAX_RETRIES):",
                  "            if time.time() > deadline:",
                  f'                logger.warning("%s.{v} 超时 key=%s attempt=%d",',
                  f'                               "{cls}", self.key, attempt)',
                  "                return False",
                  "            try:",
                  f"                self.revision += 1",
                  f"                self.payload.setdefault(\"{v}\", []).append(attempt)",
                  "                await asyncio.sleep(0)",
                  "                return True",
                  "            except (KeyError, ValueError) as exc:",
                  f'                logger.error("%s.{v} 失败: %s", "{cls}", exc)',
                  "                await asyncio.sleep(0.05 * (attempt + 1))",
                  "        return False", ""]
    return "\n".join(L)


def _ts_file(rng, mod, n_fn):
    L = [f"// {mod}.ts —— 前端侧 {mod} 客户端，基准语料生成。",
         'import { EventEmitter } from "events";', "",
         f"export interface {mod.capitalize()}Options {{",
         "  endpoint: string;", "  timeoutMs?: number;",
         "  retries?: number;", "  onError?: (e: Error) => void;", "}", ""]
    for i in range(n_fn):
        v = rng.choice(VERBS); n = rng.choice(NOUNS)
        L += [f"export async function {v}{n}{i}(",
              f"  opts: {mod.capitalize()}Options,",
              "  key: string,", "  payload?: Record<string, unknown>,",
              "): Promise<{ ok: boolean; revision: number }> {",
              "  const retries = opts.retries ?? 3;",
              "  for (let attempt = 0; attempt < retries; attempt++) {",
              "    try {",
              "      const res = await fetch(`${opts.endpoint}/" + f"{v}/" + "${key}`, {",
              '        method: "POST",',
              '        headers: { "content-type": "application/json" },',
              "        body: JSON.stringify(payload ?? {}),",
              "      });",
              "      if (!res.ok) throw new Error(`HTTP ${res.status}`);",
              "      const j = await res.json();",
              "      return { ok: true, revision: j.revision ?? 0 };",
              "    } catch (e) {",
              "      opts.onError?.(e as Error);",
              "      await new Promise((r) => setTimeout(r, 50 * (attempt + 1)));",
              "    }",
              "  }",
              "  return { ok: false, revision: -1 };",
              "}", ""]
    return "\n".join(L)


def build(seed=7, n_files=60):
    """返回 [(路径, 内容), ...]，内容为可读的类真实代码。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_files):
        mod = MODULES[i % len(MODULES)]
        if i % 3 == 2:
            p = f"src/web/{mod}{i}.ts"
            c = _ts_file(rng, mod, rng.randint(4, 9))
        else:
            p = f"src/{mod}/{mod}_{i}.py"
            c = _py_file(rng, mod, rng.randint(2, 4))
        out.append((p, c))
    return out


def as_cat_n(path, content, start=1, limit=2000):
    """按 opencode read 工具的 cat -n 格式渲染（它就是这么注入文件的）。"""
    lines = content.split("\n")[start - 1: start - 1 + limit]
    w = len(str(start + len(lines)))
    body = "\n".join(f"{str(start+i).rjust(w)}\t{l}" for i, l in enumerate(lines))
    return f"<file path=\"{path}\">\n{body}\n</file>"


if __name__ == "__main__":
    fs = build()
    tot = sum(len(c) for _, c in fs)
    print(f"{len(fs)} 个文件，合计 {tot:,} 字符（约 {tot//3.5:,.0f} tokens 估）")
    for p, c in fs[:3]:
        print(f"  {p}: {len(c):,} 字符")
