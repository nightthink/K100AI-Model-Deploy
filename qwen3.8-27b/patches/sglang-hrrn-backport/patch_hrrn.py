"""HRRN 调度策略回输到 SGLang 0.5.12（上游 #32911，合入 2026-09-08，原生最早在 0.5.20）。

上游实测：mean TTFT −69%、p99 −8%（vs FCFS，GLM-5.2 生产 trace）。
纯调度器逻辑，不碰算子/显存/通信——这是它能跨平台、也能跨版本回输的原因。

算法：Highest Response Ratio Next，以「已处理 prefill token 数」为老化轴
    ratio_delta = (processed_tokens - r.arrival_processed_tokens) / uncached
    uncached    = len(origin_input_ids) - 已命中前缀长度
排序键 (-ratio_delta, rid)：比值大者先，同值按 rid 稳定。

★ 0.5.12 与上游的四处差异（全部实测确认，非推断）
 1. 上游 r.num_matched_prefix_tokens → 0.5.12 无此字段，用 len(r.prefix_indices)
    （LPM 在 0.5.12 自己就这么取，同一语义；见 schedule_policy.py:236 段）
 2. 上游在 arg_groups/fields/schedule.py 注册 CLI → 0.5.12 无该目录；
    但 server_args.py:4949 的 --schedule-policy **没有 choices= 限制**，可直接传 hrrn
 3. _validate_and_adjust_policy 用 CacheAwarePolicy(policy) 按 value 查 Enum
    ⇒ 必须**真正注册** Enum 成员，不能只塞 _value2member_map_（塞 None 会返回 None）
 4. calc_priority 在 0.5.12 **返回 bool**，wrapper 必须透传返回值

★ 调用方契约（缺了算法就退化成按 rid 排序，上游 docstring 明示）
 · Scheduler 维护单调递增 prefill token 计数器，每轮 run_batch 累加 extend_num_tokens
 · 请求入队（_add_request_to_queue）时快照 req.arrival_processed_tokens
两者均以 wrapper 补上，不改镜像内任何源码文件。

启用：--schedule-policy hrrn（或环境变量 SGLANG_SCHEDULE_POLICY=hrrn 仅作标记）
"""


def apply():
    from enum import Enum

    from sglang.srt.managers import schedule_policy as _sp
    from sglang.srt.managers import scheduler as _sched

    SchedulePolicy = _sp.SchedulePolicy
    CacheAware = _sp.CacheAwarePolicy

    notes = []

    # ── ① 真正把 HRRN 注册进 CacheAwarePolicy（差异 3）────────────────
    if "hrrn" not in CacheAware._value2member_map_:
        member = object.__new__(CacheAware)
        member._name_ = "HRRN"
        member._value_ = "hrrn"
        CacheAware._member_map_["HRRN"] = member
        CacheAware._value2member_map_["hrrn"] = member
        CacheAware._member_names_.append("HRRN")
        notes.append("Enum 已注册 HRRN")
    HRRN = CacheAware._value2member_map_["hrrn"]
    # 自证：按 value 构造必须拿到同一个成员，否则 _validate_and_adjust_policy 会炸
    assert CacheAware("hrrn") is HRRN, "HRRN Enum 注册失败"

    # ── ② 算法本体（差异 1：分母用 prefix_indices）────────────────────
    def _uncached_len(r):
        # ★★ 绝不能写成 `getattr(r, "prefix_indices", ()) or ()`
        #   实机崩溃（2026-09-20 13:23:46，四个 TP rank 同时）：
        #     RuntimeError: Boolean value of Tensor with no values is ambiguous
        #   r.prefix_indices 是 **torch.Tensor**，不是 list。`or` 会对它做真值判断，
        #   空 tensor 上 bool(t) 直接抛异常 —— 每个请求都触发，scheduler 必崩，
        #   继而 tokenizer_manager 收 SIGQUIT 杀掉整个进程树（表象是 HTTP 500 + 服务挂）。
        #   len() 对 tensor 与 list 都成立，所以只判 None、不做真值测试。
        pi = getattr(r, "prefix_indices", None)
        matched = 0 if pi is None else len(pi)
        return max(0, len(r.origin_input_ids) - matched)

    def _sort_by_hrrn(waiting_queue, temporary_deprioritized, processed_tokens):
        def _key(r):
            rid = r.rid
            if rid in temporary_deprioritized:
                return (float("inf"), rid)
            uncached = _uncached_len(r)
            if uncached <= 0:
                return (-float("inf"), rid)   # 无 prefill 工作，立即放行
            waited = max(0, processed_tokens - getattr(r, "arrival_processed_tokens", 0))
            return (-(waited / uncached), rid)

        waiting_queue.sort(key=_key)

    SchedulePolicy._uncached_len = staticmethod(_uncached_len)
    SchedulePolicy._sort_by_hrrn = staticmethod(_sort_by_hrrn)

    # ── ③ 接管 calc_priority（差异 4：必须透传 bool 返回值）───────────
    _orig_calc = SchedulePolicy.calc_priority

    def calc_priority(self, waiting_queue, running_batch=None):
        if self.policy is not HRRN:
            return _orig_calc(self, waiting_queue, running_batch)

        # ★★ 队列过长时**绝不能回退原生**（实测 HTTP 500 的根因）
        #   原生 calc_priority 第 26 行：不认识的 CacheAware 策略直接
        #     raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        #   而 self.policy 此时正是我们注册进 Enum 的 HRRN，必抛。
        #   更要命：原生 _determine_active_policy 是
        #     `if self.policy == CacheAwarePolicy.LPM and len(queue) > 128`
        #   ——**只对 LPM 降级**，HRRN 不在其中，原生不会帮我们降级。
        #   上游之所以安全，是因为它在原生里**加了 HRRN 分支**；我们是外包一层，
        #   必须自己承担降级，不能把请求送进那个必抛的 else。
        #   ⇒ 队列过长时就地按 FCFS 语义处理（等价于原生对 LPM 的降级行为）。
        if len(waiting_queue) > 128:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return False

        # _compute_prefix_matches 已查实不按 policy 分派（只做 in-batch 前缀匹配），
        # 传 HRRN 安全。
        deprio = self._compute_prefix_matches(waiting_queue, HRRN)
        _sort_by_hrrn(waiting_queue, deprio, getattr(self, "_hrrn_tokens", 0))
        return False

    SchedulePolicy.calc_priority = calc_priority

    # ── ④ 调用方契约：入队快照 + 每轮累加 ─────────────────────────────
    Scheduler = _sched.Scheduler
    _orig_add = Scheduler._add_request_to_queue

    def _add_request_to_queue(self, req, is_retracted=False):
        req.arrival_processed_tokens = getattr(self, "_hrrn_tokens", 0)
        return _orig_add(self, req, is_retracted)

    Scheduler._add_request_to_queue = _add_request_to_queue

    _orig_run = Scheduler.run_batch

    def run_batch(self, batch, *a, **kw):
        out = _orig_run(self, batch, *a, **kw)
        n = getattr(batch, "extend_num_tokens", None)
        if n:                                  # decode / prebuilt 批为 None 或 0
            self._hrrn_tokens = getattr(self, "_hrrn_tokens", 0) + n
            pol = getattr(self, "policy", None)
            if pol is not None:
                pol._hrrn_tokens = self._hrrn_tokens
        return out

    Scheduler.run_batch = run_batch

    # ── ⑤ CLI 白名单：argparse 的 choices 里没有 hrrn，传了会被拒 ───────
    #   实测（0.5.12 server_args.py:4952）：
    #     --schedule-policy 的 choices=["lpm","random","fcfs","dfs-weight",
    #                                   "lof","priority","routing-key"]
    #   是**内联列表**、无常量可改，且 argparse 在 add_cli_args 时就已构建。
    #   故包一层 add_cli_args，在它建完 parser 后把 hrrn 追加进该 action 的 choices。
    #   ★ 教训：我先前只看 4949-4951 三行就断言「没有 choices 限制」，
    #     C 臂因此起服失败（invalid choice: 'hrrn'）。choices 就在紧跟的 4952 行。
    from sglang.srt.server_args import ServerArgs

    _orig_add_cli = ServerArgs.add_cli_args

    def add_cli_args(parser):
        r = _orig_add_cli(parser)
        for act in parser._actions:
            if "--schedule-policy" in getattr(act, "option_strings", ()):
                if act.choices is not None and "hrrn" not in act.choices:
                    act.choices = list(act.choices) + ["hrrn"]
        return r

    ServerArgs.add_cli_args = staticmethod(add_cli_args)
    notes.append("CLI choices 已加 hrrn")

    notes.append("calc_priority/run_batch/_add_request_to_queue 已接管")
    return "HRRN 回输就绪：" + "；".join(notes)


if __name__ == "__main__":
    print(apply())
