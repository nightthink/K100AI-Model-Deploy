# SPDX-License-Identifier: LicenseRef-Proprietary
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""qwen3_5(+mtp) 的 all 模式启用补丁：加标志 + 摘两道护栏（幂等）。"""
import re
import sys

F1 = "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_5.py"
s = open(F1).read()
changed = False
if "supports_mamba_prefix_caching" not in s:
    s = s.replace(
        "from vllm.model_executor.models.interfaces import (",
        "from vllm.model_executor.models.interfaces import (\n"
        "    SupportsMambaPrefixCaching,", 1)
    for cls in ("Qwen3_5ForCausalLMBase", "Qwen3_5ForConditionalGeneration"):
        pat = re.compile(r"(class %s\([^)]*\):\n)" % cls)
        if not pat.search(s):
            print(f"[patch] 未找到类 {cls}")
            sys.exit(1)
        s = pat.sub(r"\1    supports_mamba_prefix_caching = True\n", s, count=1)
    changed = True

GUARD_RE = re.compile(
    r"[ ]+if cache_config\.mamba_cache_mode == \"all\":\n"
    r"[ ]+raise NotImplementedError\(\n"
    r"(?:[^\n]*\n){1,4}?[ ]+\)\n")

m = GUARD_RE.search(s)
if m:
    s = s.replace(m.group(0), "        pass  # [gdn_prefix_cache] all-mode guard removed\n")
    changed = True
if changed:
    open(F1, "w").write(s)
print("[patch] qwen3_5 guard 残留:", len(GUARD_RE.findall(s)))

F2 = "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py"
s2 = open(F2).read()
m2 = GUARD_RE.search(s2)
if m2:
    s2 = s2.replace(m2.group(0), "        pass  # [gdn_prefix_cache] MTP all-mode guard removed\n")
    open(F2, "w").write(s2)
print("[patch] qwen3_5_mtp guard 残留:", len(GUARD_RE.findall(s2)))
