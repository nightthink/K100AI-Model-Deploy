#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""从 SGLang 容器日志汇总**实测**前缀缓存命中率与真实 prefill 吞吐。

为什么需要它：
  * 在 95% 命中的生产口径下，`prompt_tok / TTFT` 量的不是 prefill 做的功——
    绝大部分 token 根本没被重新计算。真实 prefill 速率必须用 **#new-token**。
  * SGLang **不打** "Prefix cache hit rate" 这个字符串（实测计数 0），
    所以旧报告的命中率列一直是空的；但每条 `Prefill batch` 行都带
    `#new-token` 与 `#cached-token`，据此可如实算出。

用法:
  python3 cachestat.py <容器名> [起始HH:MM:SS] [结束HH:MM:SS]
"""
import re, subprocess, sys

PAT = re.compile(r"\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d).*?Prefill batch.*?"
                 r"#new-token: (\d+), #cached-token: (\d+)")


def main():
    name = sys.argv[1]
    lo = sys.argv[2] if len(sys.argv) > 2 else None
    hi = sys.argv[3] if len(sys.argv) > 3 else None
    p = subprocess.run(["sudo", "docker", "logs", name], capture_output=True)
    txt = (p.stdout + p.stderr).decode("utf-8", "replace")
    new = cached = 0
    n = 0
    for m in PAT.finditer(txt):
        t = m.group(2)
        if lo and t < lo:
            continue
        if hi and t >= hi:
            continue
        new += int(m.group(3)); cached += int(m.group(4)); n += 1
    tot = new + cached
    print(f"Prefill batch 行数: {n:,}")
    if not tot:
        print("没有匹配到任何 prefill 记录——确认容器名与时间窗")
        return
    print(f"  #new-token   合计 {new:,}   ← 真正重新计算的输入")
    print(f"  #cached-token合计 {cached:,}   ← 前缀缓存命中")
    print(f"  **实测命中率 {100*cached/tot:.1f}%**   （生产目标 95%）")
    print(f"  命中省下的计算量占比即上式；真实 prefill 只需消化 {100*new/tot:.1f}% 的 token")


if __name__ == "__main__":
    main()
