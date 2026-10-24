"""候选⑪：把 DSA 索引器的 top-k 由 512 提到可配值（默认不改行为）。

## v2 —— 第一版补错了地方，此为改写

v1 补的是 `model_config.get_dsa_index_topk()`。**DSv4 根本不调用它**：
`deepseek_v4.py` / `deepseek_v4_dspark.py` / `deepseek_v4_backend.py` / `dsv4/` 全无引用，
五处调用方（dsa_backend、deepseek_v2、server_args、disaggregation）没有一处在 DSv4 路径上。
运行时实证：服务 `fired up=1` 而 `[INDEX_TOPK]=0`，装机行却为 1
—— 补丁改到了源码，但改的是一段永不执行的代码。

## 真正的读取点（逐字查实）

`c4_sparse_topk` 不是独立配置项：`deepseek_v4_backend.py:2259  c4_sparse_topk=self.c4_topk`，
由 `self.c4_topk` 填充。故唯一源头是 HF 配置属性，经两处读入：

    deepseek_v4_backend.py:554-556   self.c4_topk = getattr(hf_text_config, "index_topk", C4_TOPK)
    dsv4/indexer.py:921              self.index_topk = config.index_topk

## 为什么只补 deepseek_v4_backend.py，不补 _hip_radix 版

`attention_registry.py:152` 是 `elif _is_hip and not _is_hcu:` —— HCU 上 `_is_hcu` 为真，
该支跳过，落到 161 行 `else` ⇒ 实际用 `DeepseekV4AttnBackend`（deepseek_v4_backend.py）。
补 `_hip_radix` 又是死代码。

## 硬约束

`deepseek_v4_backend.py:390` 有 `assert self.c4_sparse_topk in (512, 1024)`
（注释：small 512 / large 1024）⇒ **1024 是官方支持值，2048 会直接撞断言**。
故本补丁只接受 512 / 1024。

## 开关

未设 `DSV4_INDEX_TOPK` 或值不在 {512,1024} 时**行为与原版逐字相同**。
生效时在两处各打一行 `[INDEX_TOPK]` 运行时探针——**只有它出现才算真生效**。
"""

MARKER = "PATCH_INDEX_TOPK"
B = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/deepseek_v4_backend.py"
I = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/dsv4/indexer.py"

HELPER = '''

def _patch_index_topk_override(_v, _where):  # PATCH_INDEX_TOPK
    import os as _os
    _ov = _os.environ.get("DSV4_INDEX_TOPK", "")
    if _ov in ("512", "1024") and int(_ov) != _v:
        import logging as _pl
        _pl.getLogger(__name__).info(
            "[INDEX_TOPK] PATCH_INDEX_TOPK active @%s: %s -> %s", _where, _v, _ov
        )
        return int(_ov)
    return _v
'''

OLD_B = '''        self.c4_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )
'''
NEW_B = '''        self.c4_topk = _patch_index_topk_override(  # PATCH_INDEX_TOPK
            getattr(model_runner.model_config.hf_text_config, "index_topk", C4_TOPK),
            "backend.c4_topk",
        )
'''

OLD_I = "        self.index_topk = config.index_topk\n"
NEW_I = '        self.index_topk = _patch_index_topk_override(config.index_topk, "indexer")  # PATCH_INDEX_TOPK\n'

if __name__ == "__main__":
    for path, old, new in ((B, OLD_B, NEW_B), (I, OLD_I, NEW_I)):
        s = open(path, encoding="utf-8").read()
        if MARKER in s:
            print(f"[{MARKER}] {path.split('/')[-1]} 已打过，跳过"); continue
        n = s.count(old)
        assert n == 1, f"[{MARKER}] {path.split('/')[-1]} 锚点匹配 {n} 次（应为 1），拒绝改动"
        s = s.replace(old, new, 1)
        # 把 helper 插到文件末尾之前的模块级（放最前会撞 import 顺序）
        s = s + HELPER
        open(path, "w", encoding="utf-8").write(s)
        print(f"[{MARKER}] ✓ 已打 {path.split('/')[-1]}")
    print(f"[{MARKER}] 完成（未设 DSV4_INDEX_TOPK∈{{512,1024}} 时行为不变）")
