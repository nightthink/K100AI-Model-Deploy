# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
"""线档案：把「模型/部署相关的差异」从场景逻辑里剥出来。

这套 harness 对模型基本中立 —— engine / corpus / ocproto / report2 / cachestat
五个模块一行都不用改（指标是 vLLM bench serve 口径，语料是现生成的，
工具是标准 OpenAI function 格式，命中率取自 SGLang 的 `Prefill batch` 行）。

真正与模型绑定的只有三件事，全部收在这里：

  1. `model`        —— 请求体的 model 字段，即服务端的 served-model-name
  2. `thinking_off` —— 关闭思考的 chat_template_kwargs；**各模型不同名且语义不同**
  3. `ready_token`  —— 就绪探针在 /v1/models 响应里该匹配什么

## 关于 thinking 键（实测自镜像内 sglang/srt/parser/reasoning_parser.py）

    DetectorMap["qwen3"]       → Qwen3Detector          reasoning_default="enable_thinking"
    DetectorMap["deepseek-v4"] → _DeepSeekV3Detector    reasoning_default="explicit_thinking"

两者**不同名**：Qwen 看 `enable_thinking`，DSv4 看 `thinking`。
更要紧的是语义不同 —— `explicit_` 前缀表示**不显式传就不思考**。
所以 DSv4 的 `thinking_off` 是空字典（不传即关），而不是 `{"thinking": False}`；
给 DSv4 传 `enable_thinking` 是无效键，虽然碰巧也不思考，但那是巧合不是正确。

S5/S6 之所以要关思考：opencode 的系统提示要求「少于 4 行文本输出」，
带着它测长文会得到 content=0（预算全进 reasoning，已实测复现）。
这是与模型无关的场景需求，故由档案提供「怎么关」，场景只管「要关」。

## 关于 sampling_override

有些线有**采样侧的硬边界**，跑通用负载前必须先满足，否则不是测不准而是直接崩。
它合并进每个请求体，覆盖场景自带的 temperature 等参数。

DeepSeek-V4 的 101 线（DSpark 投机）就是这样：`temperature>0` 且并发 ≥8 时
触发 GPU 硬件异常 HSA 0x1016，watchdog 超时后服务挂死、需重启约 14 分钟。
根因在 DSpark 的**拒绝后重采样分支**（不是接受判定内核，所以
SGLANG_DSPARK_FORCE_TORCH_ACCEPT 无效），上游无修复，只能回避。

回避方式取 `top_k=1` 而不是 `temperature=0`：

    temp=0.7 + top_k=1  →  8 并发 8/8 通过，accept 0.60–0.77
    temp=0.7 + top_k=2  →  HSA 0x1016，accept 掉到 0.25

`top_k=1` 数学上等价贪心（候选集只有一个 token），但**仍走完整采样代码路径**，
比直接把温度压成 0 更贴近 harness 想测的东西。生产上的对应做法是网关对
`temperature>0` 的请求强制注入 `top_k=1`。

没有这类边界的线填空字典。
"""

PROFILES = {
    "qwen3.8-27b": {
        "model": "qwen38",
        "thinking_off": {"enable_thinking": False},
        "thinking_on": {"enable_thinking": True},
        "ready_token": "qwen38",
        "sampling_override": {},
    },
    "deepseek-v4-flash": {
        "model": "deepseek-v4-flash",
        # explicit_thinking：不传即不思考，故关思考是「什么都不传」
        "thinking_off": {},
        "thinking_on": {"thinking": True},
        "ready_token": "deepseek-v4-flash",
        # 见上文：不加这条，S3 的 8/16 路并发会把 101 线打挂
        "sampling_override": {"top_k": 1},
    },
}

_REQUIRED = ("model", "thinking_off", "thinking_on", "ready_token",
             "sampling_override")


def get(name):
    """按名取档案；未知名字直接报错并列出可选值，不要静默回退到某个默认模型。"""
    if name not in PROFILES:
        raise SystemExit(
            f"未知线档案 {name!r}。可选：{', '.join(sorted(PROFILES))}\n"
            f"新增模型请在 profiles.py 的 PROFILES 里加一项（需 {len(_REQUIRED)} 个字段）。")
    p = PROFILES[name]
    missing = [k for k in _REQUIRED if k not in p]
    if missing:
        raise SystemExit(f"档案 {name!r} 缺字段：{missing}")
    return dict(p)


def names():
    return sorted(PROFILES)
