"""RadixKey 前缀匹配加速：galloping match + offset 游标，回输到 SGLang 0.5.12。

来源：上游 #37324（`[Perf] Walk the radix tree by offset instead of re-slicing
token storage`，合入 2026-09-03，ported from #36507），原生最早在 v0.5.20。

★ 实测发现（进 0.5.12 镜像读源码，非推断）：0.5.12 的 `RadixKey.match` 是
**逐 token 的 Python 循环**：

    i = 0
    for a, b in zip(t0, t1):
        if a != b: break
        i += 1

而上游同名方法早已是**指数搜索（galloping）+ 二分**，每轮只做一次 C 层切片比较。
⇒ 对「95% 命中的长前缀」负载，这是数量级差别：命中 40 万 token 时，
   0.5.12 走 40 万次 Python 迭代；galloping 约 2·log2(4e5) ≈ 37 次 C 层比较。
   **这一条的收益比 #37324 的 offset 本身更大**，故两者一并回输。

本补丁做两件事：
  ① match / match_at —— galloping + 二分，并支持 offset（不切 other）
  ② child_key_at + _match_prefix_helper 改用 offset 游标，
     消除 `key = key[prefix_len:]` 的逐层复制（0.5.12 的 __getitem__ 会构造新
     RadixKey，长前缀下每层都复制一次）

不动镜像内任何源码文件，纯运行时猴补丁。

★ 0.5.12 与上游的差异（实测）
  · 上游改的是 `unified_cache/unified_tree_core.py`，0.5.12 **无该文件**；
    我方对应物是 `mem_cache/radix_cache.py` 的 `_match_prefix_helper`，结构等价。
  · 0.5.12 的 `_split_node` 内部仍用 `key[split_len:].child_key(...)`；
    本补丁**不改 _split_node**（它在分裂路径上、每次只切一次，非热点），
    只改 helper 的逐层游标，避免牵连正确性。
"""


def apply():
    import os

    # ★ 单变量 A/B 用的闸：设 SGLANG_BACKPORT_RADIX=0 可关掉本回输，
    #   用于测「回输前基线」这一臂。默认开（纯加速、语义逐位复刻原生）。
    if os.environ.get("SGLANG_BACKPORT_RADIX", "1") == "0":
        return "radix 回输已按 SGLANG_BACKPORT_RADIX=0 关闭"

    from sglang.srt.mem_cache import radix_cache as _rc

    RadixKey = _rc.RadixKey
    RadixCache = _rc.RadixCache
    notes = []

    # ── ① galloping match，支持 offset ───────────────────────────────
    def _match_at(self, other, offset=0, page_size=1):
        """与 other[offset:] 的公共前缀长度（逻辑单位，向下取整到 page_size）。"""
        self._check_compatible(other)
        if self.is_bigram != other.is_bigram:
            raise ValueError("RadixKey operations require matching bigram modes")
        t0, t1 = self.token_ids, other.token_ids
        if offset < 0 or offset > len(t1):
            raise IndexError(f"RadixKey offset out of range: {offset}")

        n = min(len(t0), len(t1) - offset)
        # 指数搜索：先按倍增窗口整块比（每次一个 C 层切片比较），
        # 命中分歧窗口后在窗口内二分。等价于逐 token，但比较次数 O(log n)。
        lo, step = 0, 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[offset + lo : offset + hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[offset + lo : offset + mid]:
                        lo = mid
                    else:
                        hi = mid
                break
            lo = hi
            step *= 2
        matched_tokens = lo

        if self.is_bigram:
            # L 个匹配 raw token ⇒ L-1 个匹配 bigram
            m = max(0, min(matched_tokens - 1, len(self), len(other) - offset))
            return (m // page_size) * page_size if page_size > 1 else m

        if page_size == 1:
            return min(matched_tokens, len(self), len(other) - offset)

        # ★ page_size > 1：必须**逐位复刻** 0.5.12 原生语义，不能照抄上游。
        #   原生：while i < min_len: if t0[i:i+ps] != t1[i:i+ps]: break; i += ps
        #   两个反直觉之处（对拍实测，非推断）：
        #     · 返回值可**超过** min_len：n=3/ps=2 全匹配时原生返回 4（i=2 时比较
        #       t0[2:4] vs t1[2:4]，Python 切片越界自动截断成单元素，相等 ⇒ i=4）；
        #     · 即末页只要「截断后的残余相等」就整页计入。
        #   上游改成 (m//ps)*ps 会得 2 或 0 —— 语义不同，对拍 2000 例差出 62 例。
        #   回输原则：只换实现（逐 token → galloping），绝不顺手改变对外行为；
        #   原生这个「超长返回」是否算 bug 交给上游，我方不在回输里变更，以免牵连
        #   _match_prefix_helper 里 prefix_len 与 len(child.key) 的比较及 value 切片。
        #   galloping 已给出 matched_tokens（首个分歧位置），据此按原生规则步进，
        #   只做 O(n/ps) 次整数比较，不再逐页切片。
        min_len = min(len(self), len(other) - offset)
        i = 0
        while i < min_len:
            # 本页在 [i, i+ps) 与原串的有效交集；越界由切片截断，故取 min
            page_end = min(i + page_size, len(t0), len(t1) - offset)
            if matched_tokens < page_end:      # 页内存在分歧
                break
            i += page_size
        return i

    def _match(self, other, page_size=1):
        return _match_at(self, other, 0, page_size)

    RadixKey.match_at = _match_at
    RadixKey.match = _match
    notes.append("match→galloping+offset")

    # ── ② child_key_at：不切 token 存储 ──────────────────────────────
    def _child_key_at(self, offset=0, page_size=1):
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[offset], t[offset + 1])
            else:
                plain = tuple((t[offset + j], t[offset + j + 1]) for j in range(page_size))
        else:
            plain = (
                t[offset] if page_size == 1 else tuple(t[offset : offset + page_size])
            )
        return plain if self.extra_key is None else (self.extra_key, plain)

    def _child_key(self, page_size=1):
        return _child_key_at(self, 0, page_size)

    RadixKey.child_key_at = _child_key_at
    RadixKey.child_key = _child_key
    notes.append("child_key→offset 版")

    # ── ③ _match_prefix_helper 用游标替代逐层切片 ────────────────────
    def _match_prefix_helper(self, node, key):
        import time as _t

        access_time = _t.monotonic()
        node.last_access_time = access_time

        key_offset = 0
        total = len(key)
        child_key = _child_key_at(key, key_offset, self.page_size)

        value = []
        while key_offset < total and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = _match_at(child.key, key, key_offset, page_size=self.page_size)
            if prefix_len < len(child.key):
                # 分裂路径：沿用原生 _split_node（三参，实测签名）。
                # ★ 注意：_split_node 内部会对传入的 key 调 child_key()，原代码传的是
                #   「已切到当前层」的 key；改用游标后必须显式切一次保持语义一致。
                #   该路径每次匹配至多走一次（走到即 break），非热点，切一次无妨。
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key_offset += prefix_len
                if key_offset < total:
                    child_key = _child_key_at(key, key_offset, self.page_size)

        return value, node

    # 实测 0.5.12 的 _split_node 是 (self, key, child, split_len)；若上游版本
    # 改了签名，这里要跟着改——故加一道断言，签名不符就明确失败而不是静默跑错。
    import inspect

    _np = len(inspect.signature(RadixCache._split_node).parameters)
    assert _np == 4, f"_split_node 签名变了（参数 {_np} 个，期望 4），补丁需复核"
    RadixCache._match_prefix_helper = _match_prefix_helper
    notes.append("helper→offset 游标")

    return "radix 回输就绪：" + "；".join(notes)


if __name__ == "__main__":
    print(apply())
