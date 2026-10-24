"""让 ROCm/HIP 上的 FlashMLA 稀疏 prefill 可以被显式打开，并给出运行时凭证。

## 背景（2026-09-13 剖析 prefill 时查到）

DSv4 的 prefill 有两条路：稠密（`_forward_flash_mla_prefill`）与稀疏
（`_forward_prefill_sparse`，走 `flash_mla_sparse_fwd`，只算索引器选出的 top-512 键）。
派发点在 `layers/attention/deepseek_v4_backend.py: forward` 的 prefill 分支：

    if forward_batch.forward_mode.is_prefill(...):
        if _should_use_sparse_prefill(q, forward_batch):     # ← 1990 行，HCU 可达
            return self._forward_prefill_sparse(...)
        return self._forward_flash_mla_prefill(...)

    def _should_use_sparse_prefill(q, forward_batch):
        return not dsa_use_prefill_cp(forward_batch) and (
            q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD          # = 11673
            or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()     # 上游默认 True
        )

本机两条都不满足，所以**prefill 一直走稠密**：

  · chunk 4096 < 11673，尺寸触发不了；
  · 那个 env 上游默认 True，但 `arg_groups/deepseek_v4_hook.py` 在 HIP 上**无条件**
    `envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)`，理由是"在 MI355X 上输出不正确"。
    而 `EnvField.set()` 就是 `os.environ[name] = str(value)`，**会盖掉用户显式设的值**——
    日志里那句 "set it explicitly to override" 与代码不符，照它做没有任何效果。

`flash_mla_sparse_fwd` 在本镜像里是有的（海光自家 `flash_mla 1.2.0+das.opt1.dtk2604`，
`flash_mla_interface.py:403`，内部还有 gfx936 分支与稀疏索引排序），
`_forward_prefill_sparse` 也专为 `_is_hcu` 从该包导入。所以这条路在 HCU 上是**可试的**，
只是被上游一刀切关掉了。上游的"不正确"结论来自 MI355X（gfx950），与 K100-AI（gfx928）不是一回事，
必须由我们自己的 correctness 四项 + needle 判定，不能照抄，也不能照抄地信。

## 本补丁做两件事

① `deepseek_v4_hook.py`：HIP 上的强制关闭改为**仅在用户没有显式设置该 env 时**生效。
   不设 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` 时行为与原来完全一致（仍然关）。
② `deepseek_v4_backend.py: _forward_prefill_sparse`：首次真实调用打一行
   `[SPARSE_PREFILL] PATCH_SPARSE_PREFILL first sparse prefill call: q=…`。
   **判"稀疏 prefill 是否真在跑"只认这一行**——这一晚已经栽过一次：custom AR 的
   初始化警告被我当成了"在用"，白跑一个 2 小时基准。

## 用法与风险

    -e SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1

风险是数值：稀疏 prefill 只看 top-512 键，长上下文下若索引器选得不好，
表现为 needle 检索退化而不是崩溃。所以判据必须是 correctness 四项 **加** needle 多档，
只看"能起服、快了"会误判。
"""
MARKER = "PATCH_SPARSE_PREFILL"
BASE = "/usr/local/lib/python3.10/dist-packages/sglang/srt"
HOOK = f"{BASE}/arg_groups/deepseek_v4_hook.py"
BACKEND = f"{BASE}/layers/attention/deepseek_v4_backend.py"

OLD_HOOK = """    if is_hip():
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
"""
NEW_HOOK = """    # PATCH_SPARSE_PREFILL: 原来是无条件 set(False)，而 EnvField.set() 直接写 os.environ，
    # 会盖掉用户显式设的值——日志里那句 "set it explicitly to override" 与代码不符。
    # 改为只在用户未显式设置时才强制关闭；未设时行为与原来完全一致。
    if is_hip() and not envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set():
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    elif is_hip():
        logger.warning(
            "PATCH_SPARSE_PREFILL: SGLANG_OPT_FLASHMLA_SPARSE_PREFILL="
            f"{envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()} 由使用者显式指定，不覆盖。"
        )
"""

OLD_BACKEND = """        if _is_hcu:
            try:
                from flash_mla.flash_mla_interface import flash_mla_sparse_fwd
            except ImportError:
                from flash_mla import flash_mla_sparse_fwd
        else:
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd
"""
NEW_BACKEND = """        # PATCH_SPARSE_PREFILL: 运行时凭证。初始化/配置日志证明不了内核真被调用，
        # 判稀疏 prefill 是否生效只认这一行。
        if not getattr(self, "_sparse_prefill_probe_logged", False):
            self._sparse_prefill_probe_logged = True
            logger.info(
                "[SPARSE_PREFILL] PATCH_SPARSE_PREFILL first sparse prefill call: "
                "q=%s compress_ratio=%s layer=%s",
                tuple(q.shape), compress_ratio, layer_id,
            )
        if _is_hcu:
            try:
                from flash_mla.flash_mla_interface import flash_mla_sparse_fwd
            except ImportError:
                from flash_mla import flash_mla_sparse_fwd
        else:
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd
"""


def _hunk(path, old, new, tag, hint):
    s = open(path, encoding="utf-8").read()
    if tag in s:
        print(f"  {tag}: 已打过，跳过")
        return
    assert s.count(old) == 1, f"{tag} 锚点命中 {s.count(old)} 处（应为 1）——{hint}"
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print(f"  {tag}: ✓")


if __name__ == "__main__":
    print(f"[{MARKER}] 解禁 ROCm 上的 FlashMLA 稀疏 prefill（仅在显式设 env 时生效）")
    assert "logger" in open(BACKEND, encoding="utf-8").read()[:4000], "backend 无 module 级 logger"
    _hunk(HOOK, OLD_HOOK, NEW_HOOK, MARKER + "_HOOK", "请核对 deepseek_v4_hook.py 第 32 行附近")
    _hunk(BACKEND, OLD_BACKEND, NEW_BACKEND, MARKER + "_PROBE", "请核对 _forward_prefill_sparse 开头的 import 段")
    for f in (HOOK, BACKEND):
        assert MARKER in open(f, encoding="utf-8").read(), f"✗ {f} 未见标记"
    import py_compile
    py_compile.compile(HOOK, doraise=True)
    py_compile.compile(BACKEND, doraise=True)
    print(f"[{MARKER}] 两处均已生效；开启方式：-e SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1")
