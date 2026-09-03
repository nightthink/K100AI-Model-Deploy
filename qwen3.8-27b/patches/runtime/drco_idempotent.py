import torch
from torch.library import Library, infer_schema
from collections.abc import Callable


DEFAULT_DISPATCH_KEY = "CUDA"


# 从vllm中迁移过来的，用于注册自定义op
#
# ★ 本地补丁（2026-08-28，DaoTechAi）：注册幂等化。
#   镜像里 `lightop/_lmslim_native/quantize/quant_ops.py` 与
#   `lmslim/quantize/quant_ops.py` 各带一份本文件，且都用
#   Library("lmslim", "FRAGMENT") 注册同名算子（gptq_gemm1 等）。
#   加载 compressed-tensors INT8 模型时两者都会被导入，后到者抛：
#       RuntimeError: Tried to register an operator (lmslim::gptq_gemm1(...))
#   导致任何 INT8 模型无法启动。此处改为：已注册则跳过，保留先到者的实现。
#   模块其余符号（BlockSize 等）照常导出，不影响其它导入方。
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    if mutates_args is None:
        mutates_args = []

    if target_lib is None:
        raise TypeError(
            "direct_register_custom_op(...): target_lib must not be None. "
            "Create once: `from torch.library import Library` then "
            "`lmslim_lib = Library('lmslim', 'FRAGMENT')` and pass `target_lib=lmslim_lib`."
        )

    if dispatch_key is None:
        dispatch_key = DEFAULT_DISPATCH_KEY

    # 先查：该算子是否已在 torch.ops.lmslim 下存在
    try:
        _ns = getattr(torch.ops, "lmslim", None)
        if _ns is not None and hasattr(_ns, op_name):
            return
    except Exception:
        pass

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib
    try:
        my_lib.define(op_name + schema_str, tags=tags)
    except RuntimeError as e:
        msg = str(e).lower()
        if "register" in msg or "already" in msg or "exist" in msg:
            return          # 重复注册：静默跳过
        raise
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
