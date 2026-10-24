"""开启 custom all-reduce（PCIe 降级模式）—— 移植自海光给 Qwen 0828 底座的改法。

## 为什么本镜像默认开不了 custom AR

DSv4 的 0811 镜像用的是上游原版判定：

    custom_all_reduce_utils.py:478
        if world_size > 2 and not full_nvlink:
            logger.warning("... is disabled because it's not supported on
                            more than two PCIe-only GPUs ...")
            return                      # ← 裸 return，函数给出 None

    custom_all_reduce.py:99
        if full_nvlink is None:
            return                      # ← __init__ 早退，custom AR 根本不初始化

而 K100-AI 八卡实测 `is_full_nvlink([0..7], 8) = False`（链路全是 PCIe 2~3 跳、
零 XGMI），TP8 必然撞上这条禁用。

★ 2026-09-12 更正：上面这段只对 **legacy CustomAllreduce** 成立。本镜像默认
  `custom_all_reduce_backend=auto`，而 server_args 会把 auto **提升为 aiter**，
  实际走的是 `aiter.dist...CustomAllreduce`（日志：`[AR] Using AiterCustomAllreduce
  (HCU/DTK, transport=ipc)`），它**根本不读** can_use_custom_all_reduce_with_nvlink，
  本补丁对它毫无作用。

  我曾据此写下「加不加 --disable-custom-all-reduce 在本镜像上没有区别」——**这句是错的**，
  已由实验证伪：摘掉该开关后（无论挂不挂本补丁）八个 rank 的 aiter AR 都能初始化成功，
  随即 **Rank 0 scheduler SIGSEGV（exit -11）**，服务起不来。

  根因推断：aiter 在 custom_all_reduce.py:485-486 把 `fully_connected` **硬编码为 True**
  （真正的 is_full_nvlink 探测被注释掉），于是拿 XGMI 全连通的假设去跑 PCIe 2~3 跳的
  K100-AI。这与历史记录里那条「custom allreduce 挂死，AR 初始化后彻底无日志推进」
  是同一个病灶。

  **结论：产线那行 `--disable-custom-all-reduce` 不是历史包袱，是刚需**——
  它挡住的正是这个 SIGSEGV。要用本补丁，必须同时加
  `--custom-all-reduce-backend native` 强制走 legacy，绕开 aiter。

★ 2026-09-12 23:52 该路线已实测跑通（环境 ~/dsv4_native）：

    摘 --disable-custom-all-reduce
  + --custom-all-reduce-backend native      ← 绕开只编了 gfx938 的 aiter
  + 本补丁                                   ← 解禁 legacy 门控并把缓冲降到 512KB

    PCIe 降级警告   8 行（八个 rank 全部启用）
    Segmentation fault  0
    max_total_num_tokens  1,017,344（较 P2P-only 的 1,017,600 少 256，AR 缓冲开销）
    冒烟数值正确    37 × 89 − 156 = 3137 ✓

  三条缺一不可：只摘开关（走 aiter）必 SIGSEGV；不摘开关则 AR 根本不初始化；
  不打补丁则 legacy 在非全连通下 return None 早退。

★ 2026-09-13 03:10 更正：上面「跑通」只是**初始化跑通，运行时一次都没用上**。

  2h 标准 harness 基准（101ar）与 P2P-only（101p2p）逐项重合在噪声内：
    单流 decode 35.2 vs 35.8 ｜ 16 路聚合 decode 99 vs 95 ｜ 冷启 prefill 810 vs 803
  追到 custom_all_reduce.py 的 should_custom_ar，HIP 分支是：

      if _is_hip:
          if self.use_amd_deterministic_impl: return True
          if self.full_nvlink: return inp_size <= self.max_size
          return False          # ← full_nvlink=False 时任何张量都进不了 CA

  ① ② 只让对象建起来（disabled=False、日志 8 行 PCIe 降级警告），而每次 all_reduce
  分发（parallel_state.py:729）问 should_custom_ar 都得到 False，全部回落 NCCL。
  **「8 行警告」证明的是初始化，不是使用**。以后判 AR 生效只认本补丁在
  _all_reduce_impl 首次调用打的那行 `[AR] PATCH_CUSTOM_AR_PCIE first custom AR call`。

  修法 ③：HIP 分支去掉 full_nvlink 判定，只按 max_size（已是 512KB）筛。
  修法 ④：_all_reduce_impl 首调打一行日志（含字节数）——运行时生效的唯一凭证。
  于是变成四处缺一不可：①utils 不早退 ②缓冲降 512KB ③should_custom_ar 放行 ④探针。

★ 2026-09-13 04:05 判死：③ 放行后 **内核算错**，与拓扑无关。

  服务级：TP8 起服后 correctness 0/4，输出全是 <｜begin▁of▁sentence｜> 复读（探针 8 行、
  首调 480KB registered=True，证明 AR 确实在跑）。
  单元级（~/ca_unit_test.py，直接构造 CustomAllreduce 与 NCCL 对照，三种卡组）：
    8 卡跨双 socket / 4 卡同 socket(0-3) / 4 卡跨 socket(0,1,4,5) —— **三组全错**：
    8KB~480KB 档输出全 0；512KB 档每 rank 只得到自己的输入（未归约）；
    同一缓冲连续 20 次调用 bad_elems 呈 [0, 全错, 半错, 全错, 半错…] 一拍延迟交替；
    图注册路径 5 次 replay 全错。
  这是"读到对端尚未写完的缓冲"的典型症状——上游用 full_nvlink 把 PCIe 挡掉不是
  经验之谈，是这个内核的同步语义在 PCIe 上不成立。
  同 socket 四卡也错 ⇒ 与 Qwen 0828 镜像（海光改法实测正确、decode 2.15×）的差别
  在镜像内核层，不在拓扑。

  因此 ③ **默认关闭**，只有 CUSTOM_AR_PCIE_FORCE=1 才打（留给日后换内核后复验）。
  ①②④ 保留：无害，且 ④ 探针是判定"AR 是否真在跑"的唯一凭证。
  默认状态下本补丁的净效果 = AR 对象建起但从不使用 = 与不打补丁等价（101ar 基准已证）。

## 海光在 Qwen 0828 底座上的改法

同一个判定，海光把禁用逻辑注释掉，改成「不禁用，只把缓冲压小继续用」：

    # if world_size > 2 and not fully_connected:
    if not fully_connected:
        max_size = 32 * 8192 * 2        # 512 KB
        logger.warning("We are using PCIe's custom allreduce. ...")
    # 不 return，继续往下走

Qwen 侧单变量 A/B 实测 decode **2.15×**（见 Qwen 项目
docs/bench-2026-09-09-选线指南/custom-all-reduce-AB定案.md）。

## 为什么不能照搬，要改写

两个镜像的 sglang 不同代：Qwen 0828 把判定写在 `CustomAllreduce.__init__`
内部（用 self.group、末尾直接 self.disabled = False）；DSv4 0811 已经把它抽成
`custom_all_reduce_utils.can_use_custom_all_reduce_with_nvlink()` 独立函数，
靠返回值 None 让 __init__ 早退。照抄那段代码会落在错误的位置上。

按 DSv4 的结构改写两处：

  ① utils：非全连通时不再裸 return，继续走到函数末尾的 `return full_nvlink`，
     于是 __init__ 拿到的是 False 而不是 None，不再早退。
  ② __init__：full_nvlink 为 False 时把 max_size 从 16 MB 压到 512 KB。
     HIP 下 _MAX_CAR_SIZE = 2*8192*1024 = 16 MB，而 PCIe 非全连通下大缓冲
     不可靠——Qwen 版正是降到 32*8192*2。只解禁而不降缓冲是危险的半吊子。

## 512 KB 的含义（选型时要知道）

缓冲上限 512 KB 意味着 custom AR **只对 decode 生效、对 prefill 无效**——
prefill 的张量远大于此，走不了这条路。所以预期收益集中在 decode 与并发，
prefill 不会变快。
"""
import os

MARKER = "PATCH_CUSTOM_AR_PCIE"
BASE = "/usr/local/lib/python3.10/dist-packages/sglang/srt/distributed/device_communicators"
UTILS = f"{BASE}/custom_all_reduce_utils.py"
CAR = f"{BASE}/custom_all_reduce.py"


def patch_utils():
    s = open(UTILS, encoding="utf-8").read()
    if MARKER in s:
        print("  utils: 已打过，跳过")
        return
    old = '''    if world_size > 2 and not full_nvlink:
        logger.warning(
            f"{cls_name} is disabled because it's not supported on"
            " more than two PCIe-only GPUs. To silence this warning, "
            "specify disable_custom_all_reduce=True explicitly."
        )
        return
'''
    new = '''    # PATCH_CUSTOM_AR_PCIE: 非全连通不再禁用，降级为 PCIe 模式继续启用。
    # 原上游在此裸 return（→None→__init__ 早退）；K100-AI 八卡全 PCIe、
    # is_full_nvlink 恒 False，那等于 custom AR 永远无法启用。
    # 改为继续走到末尾 return full_nvlink（False），由 __init__ 把缓冲压到 512KB。
    if world_size > 2 and not full_nvlink:
        logger.warning(
            f"{cls_name}: PCIe-only topology, using degraded custom allreduce "
            "(512KB buffer, decode-only). Add --disable-custom-all-reduce to opt out."
        )
'''
    assert old in s, "utils 锚点未命中——镜像版本可能已变，请重新核对第 478 行附近"
    open(UTILS, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print("  utils: ✓ 非全连通不再 return None")


def _hunk(path, old, new, tag, hint):
    """同一文件里有多处改动，跳过判定要按每处自己的标记，不能按文件级 MARKER。"""
    s = open(path, encoding="utf-8").read()
    if tag in s:
        print(f"  {tag}: 已打过，跳过")
        return
    assert old in s, f"{tag} 锚点未命中——镜像版本可能已变，{hint}"
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print(f"  {tag}: ✓")


TAG_CAR_BUF = MARKER + "_BUF"
TAG_CAR_SHOULD = MARKER + "_SHOULD"
TAG_CAR_PROBE = MARKER + "_PROBE"


def patch_car():
    # ② __init__：PCIe 下缓冲降 512KB
    _hunk(CAR, '''        self.group = group
        self.max_size = max_size
''', '''        self.group = group
        # PATCH_CUSTOM_AR_PCIE_BUF: 非全连通（PCIe-only）时把缓冲从 16MB 压到 512KB。
        # HIP 下 _MAX_CAR_SIZE = 2*8192*1024 = 16MB，PCIe 拓扑下大缓冲不可靠；
        # 512KB 也意味着 custom AR 只对 decode 生效、对 prefill 无效。
        if not full_nvlink:
            max_size = 32 * 8192 * 2
        self.max_size = max_size
''', TAG_CAR_BUF, "请重新核对 __init__ 第 101 行附近")

    # ③ should_custom_ar：HIP 分支不再以 full_nvlink 一票否决，只按 max_size 筛。
    #    不打这一处，①② 只是让对象建起来，运行时每次分发都回落 NCCL（101ar 基准零效的根因）。
    #    ⚠ 2026-09-13 判死：打上后内核在任何卡组都算错（见文件头），默认不打。
    if os.environ.get("CUSTOM_AR_PCIE_FORCE") != "1":
        print("  PATCH_CUSTOM_AR_PCIE_SHOULD: 跳过（内核在 K100-AI 上算错，需 CUSTOM_AR_PCIE_FORCE=1 强开）")
    else:
      _hunk(CAR, '''        if _is_hip:
            if self.use_amd_deterministic_impl:
                return True
            if self.full_nvlink:
                return inp_size <= self.max_size
            return False
''', '''        if _is_hip:
            if self.use_amd_deterministic_impl:
                return True
            # PATCH_CUSTOM_AR_PCIE_SHOULD: 非全连通也放行，上限由 __init__ 压成的 512KB 兜底。
            return inp_size <= self.max_size
''', TAG_CAR_SHOULD, "请重新核对 should_custom_ar 的 `if _is_hip:` 分支")

    # ⑤ deterministic（SGLANG_USE_1STAGE_ALLREDUCE=1）分支：上游对任何尺寸都 return True，
    #    ≥max_size 的张量会走 register_buffer + deterministic_all_reduce_reg——单测证实该路径算错
    #    （524288B 常量得到部分和）；<max_size 的 unreg 路径与图注册路径在 8 卡/4 卡上全对。
    #    改成严格 < max_size：把 1STAGE 内核限制在 decode 张量，prefill 大张量继续走 NCCL。
    _hunk(CAR, '''        if _is_hip:
            if self.use_amd_deterministic_impl:
                return True
''', '''        if _is_hip:
            if self.use_amd_deterministic_impl:
                # PATCH_CUSTOM_AR_PCIE_DET: 只放行 < max_size（512KB），reg 路径在 K100-AI 上算错
                return inp_size < self.max_size
''', MARKER + "_DET", "请重新核对 should_custom_ar 的 use_amd_deterministic_impl 分支")

    # ④ 首次真实调用打一行日志：运行时生效的唯一凭证（初始化警告证明不了使用）
    _hunk(CAR, '''    def _all_reduce_impl(self, inp: torch.Tensor, registered: bool):
        out = torch.empty_like(inp)
''', '''    def _all_reduce_impl(self, inp: torch.Tensor, registered: bool):
        if not getattr(self, "_pcie_probe_logged", False):
            self._pcie_probe_logged = True
            logger.info(
                "[AR] PATCH_CUSTOM_AR_PCIE_PROBE first custom AR call: %d bytes, registered=%s",
                inp.numel() * inp.element_size(), registered,
            )
        out = torch.empty_like(inp)
''', TAG_CAR_PROBE, "请重新核对 _all_reduce_impl 开头两行")


if __name__ == "__main__":
    print("[patch_custom_ar] 开启 PCIe 降级模式的 custom all-reduce")
    patch_utils()
    patch_car()
    # 自检：四处都必须带上标记，否则后面会拿着"以为打过"的镜像去跑
    u = open(UTILS, encoding="utf-8").read()
    c = open(CAR, encoding="utf-8").read()
    assert MARKER in u, f"✗ {UTILS} 未见标记"
    need = [TAG_CAR_BUF, TAG_CAR_PROBE, MARKER + "_DET"]
    if os.environ.get("CUSTOM_AR_PCIE_FORCE") == "1":
        need.append(TAG_CAR_SHOULD)
    for t in need:
        assert t in c, f"✗ {CAR} 未见 {t}"
    print("[patch_custom_ar] 生效：utils 不早退 / 缓冲 512KB / 1STAGE 限 <512KB / 首调探针"
          + (" / should 强制放行" if TAG_CAR_SHOULD in need else ""))
    print("[patch_custom_ar] 提示：真正启用 AR 需 -e SGLANG_USE_1STAGE_ALLREDUCE=1（走 deterministic 内核）")
