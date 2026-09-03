# HIP agent 列表过滤 —— 缺陷 1 的真正修复

> **这是本次调查唯一一个「修好了根因」的补丁**，而不是规避。
> 其余四个 RCCL 补丁均已实测否决（见各自文件头的标注）。

> ## ⚠️ 2026-08-23 重大更新，请先读这三条
>
> 1. **改用 `dlhook2.c`，不要再用 `dlhook.c`。**
>    `dlhook.c` 按**下标**过滤 agent，而 HIP 传给 `allow_access` 的列表**顺序不固定**
>    （101 次调用里 2 次把当前设备提到队首、1 次只传 7 个），会挑错卡 → 段错误/挂死。
>    `dlhook2.c` 改为按 **agent 句柄**识别卡号。详见 §八。
> 2. **§7.3 的「④ PXB + 按 socket 过滤 → 挂死」是假结果**，就是上面那个 bug 造成的。
>    用 `dlhook2.c` 重做后，**TP8 集合带宽 1.5 → 5.7 GB/s（3.75×），正确性 27/27 通过**。详见 §九。
> 3. **根因已追到内核**：跨 socket 的 VRAM IPC 映射会让 hycu 陷入 evict↔restore 活锁，
>    中毒进程白丢 22% 的 GPU 时间。完整证据链见 `docs/复盘三-52ms活锁根因与TP8混合传输.md`。

---

## 一、根因

HIP（`libgalaxyhip.so`）在**每一次显存分配**时，都调用

```c
hsa_amd_agents_allow_access(num_agents, agents, flags, ptr)
```

把这块内存授权给**进程可见的全部 GPU**。

Hook 观察到的实际调用：

```
可见 0,1     →  allow_access num_agents=2   agents=16a3c750,16a42920
可见 0,1,4   →  allow_access num_agents=3   agents=17181750,17187920,17199e00
```

**agent 列表一旦横跨两个 socket，驱动就为这块内存建立一个「对两个 socket 都有效」的
保守映射** —— 跨进程 IPC 的可见性延迟从 **1 µs 变成 52 ms**。

这解释了为什么「只是让一张从不使用的卡可见」就会中招：
**那张卡进了 agent 列表。**

## 二、修复

拦截 `hsa_amd_agents_allow_access`，只把**实际会用到的 GPU** 传下去。

### 为什么必须拦 `dlsym`

`libgalaxyhip.so` 的 `DT_NEEDED` 里**没有** `libhsa-runtime64`，
466 个未定义符号里**含 hsa 的是 0 个** —— 它是 `dlopen` + `dlsym` 动态取符号的。

所以普通 `LD_PRELOAD` 拦不到（早先试过，一次都没被调用）。
本补丁改为**拦截 `dlsym` 本身**，在它返回 HSA 符号时换成我们的包装函数。

## 三、实测

### 纯 HIP（跨进程 IPC 标志位往返）

| 配置 | 往返延迟 |
|---|---|
| 可见 `0,1`（基线）| 1.03 µs |
| 可见 `0,1,4`，无 hook | **51,999.65 µs** |
| **可见 `0,1,4` + `DLHOOK_MAXAG=2`** | **1.03 µs** |
| **可见 `0-7` + `DLHOOK_KEEP=0,1`** | **1.05 µs** |
| **可见 `0-7` + `DLHOOK_KEEP=0,1,2,3`** | **1.04 µs** |

### 真实 RCCL（8 字节 all-reduce）

| 配置 | 延迟 |
|---|---|
| TP4 可见 `0,1,2,3`（基线）| 65.7 µs |
| TP4 可见 `0-7`，无 hook | **415,873.8 µs** |
| **TP4 可见 `0-7` + `KEEP=0,1,2,3`** | **63.0 µs** ← **快 6600 倍** |
| TP4 可见 `4,5,6,7`（基线）| 64.0 µs |
| **TP4 可见 `0-7` 用后四张 + KEEP** | **73.6 µs** |

### 正确性

`verify_ar.py`（bf16 / fp32 / fp16 × 1 / 16 / 80 MB 的 all_reduce + reduce_scatter）：

| 配置 | 结果 |
|---|---|
| TP4 可见 `0,1,2,3` 基线 | **ALL CORRECT** |
| TP4 可见 `0-7` + `KEEP=0,1,2,3` | **ALL CORRECT** |
| TP4 可见 `0-7` + 后四张 + KEEP | **ALL CORRECT** |

**没有 `NCCL_SHM_USE_CUDA_MEMCPY=1` 那种静默算错的问题。**

## 四、用法

```bash
gcc -shared -fPIC -O2 -o dlhook.so dlhook.c -ldl

# 观察：打印每次调用的 agent 列表
DLHOOK_LOG=1 LD_PRELOAD=/path/dlhook.so ./your_app

# 修复：只保留下标 0,1,2,3 的 agent（按枚举序，通常等于 GPU 序）
DLHOOK_KEEP=0,1,2,3 LD_PRELOAD=/path/dlhook.so torchrun --nproc_per_node=4 ...

# 或按个数截断
DLHOOK_MAXAG=4 LD_PRELOAD=/path/dlhook.so ./your_app
```

**`DLHOOK_KEEP` 里的下标必须覆盖作业实际使用的全部 GPU**，
否则未被授权的卡访问时会失败。

## 五、能做什么，不能做什么

### ✅ 能做

**在无法控制 `HIP_VISIBLE_DEVICES` 的场景下，恢复同 socket 作业的全速。**
例如调度器强制给了全部 8 张卡，而作业只用其中 4 张。

### ❌ 不能做：TP8 仍然无解

实测 `TP8 + KEEP=0,1,2,3` → **失败**。原因很直接：
TP8 的 8 张卡**真的需要互相访问**，agent 列表不能缩。
而只要列表跨 socket，那块内存就一定拿到保守映射。

即使做成「每块缓冲只授权给环上的左右邻居」，
8 卡环里仍有 2 条边跨 socket（`3→4` 和 `7→0`），
这两条边的缓冲照样是 52 ms/次。**环形 TP8 在这个缺陷下没有出路。**

### 当前生产是否需要它

**不需要。** DP2×TP4 的两个副本本来就分别传
`GPUS=0,1,2,3` 和 `GPUS=4,5,6,7`，可见集合天然不跨 socket。

本补丁的价值在于：
1. **它证明了根因的确切位置** —— 从「某处慢」变成「就是这个 API 的这个参数」
2. **它是一份可交付的证明** —— 改一个参数就能让 6600 倍的差距消失
3. 作为纵深防御：万一有人传了 `0-7`，加上 hook 仍能保住性能

## 六、给海光的一句话

> `hsa_amd_agents_allow_access` 在 agent 列表横跨两个 socket 时，
> 为该内存建立的映射会使跨进程 IPC 的可见性延迟从 1 µs 退化到 52 ms。
> 把列表限制到单个 socket 内即恢复正常（我们用 `dlsym` hook 验证过，
> 纯 HIP 与 RCCL 两个层面都成立，且结果正确）。
> **请检查该调用在多 NUMA / 多 socket agent 列表下的映射属性（MTYPE / 页表）决策。**

---

# 七、追查：能否骗过「跨 socket」判断以救回 TP8

## 7.1 两种「骗」是不同的

| 思路 | 做法 | 结果 |
|---|---|---|
| **骗 agent 列表** | 把列表缩短 | ❌ 已证伪 —— TP8 + `KEEP=0,1,2,3` **失败**，卡 4-7 真的访问不了。**授权是真的，不是装饰** |
| **骗拓扑** | 列表保持 8 个，但让它认为都在同一 socket | ❌ **骗不了** —— 见下 |

## 7.2 ★ 决定在内核，不在用户态

拦截 `ioctl`（`kfdtrace.c`），对比快慢两种情况传给内核的参数：

| | 慢（可见 `0,1,4`）| 快（同上 + `KEEP=0,1`）|
|---|---|---|
| 往返 | 51,997 µs | **1.05 µs** |
| ALLOC 次数 / MAP 次数 | 124 / 126 | 124 / 126 |
| **ALLOC flags 分布** | **完全相同**（10 种 flag，逐个计数一致）| **完全相同** |
| **MAP `n_devices` 分布** | 22×1、**42×3**、62×8 | 22×1、**42×2**、62×8 |

**用户态没有传递任何「跨不跨 socket」的标志** —— ALLOC 的 flags 一位都没变。
唯一变化是 `MAP_MEMORY_TO_GPU` 的 gpu_id 列表长度（3 → 2）。

> **结论：内核是看着 gpu_id 列表自己推断 socket 归属的。**
> 骗 sysfs 拓扑也没用 —— 内核不读 sysfs，它是 sysfs 的提供方。

顺带两个观察：
1. **`HIP_VISIBLE_DEVICES` 并不减少 KFD 层的映射范围** —— 即使只可见 2 张卡，
   仍有 62 次 `MAP_MEMORY_TO_GPU` 用 `n_devices=8` 映射给全部 8 张物理卡
2. 那 62 次 8 路映射**不导致变慢** —— 只有 HIP 按「可见 agent 列表」做的那 42 次才敏感

## 7.3 试过的 TP8 组合

| 配置 | 8 字节 all-reduce |
|---|---|
| ① 基线 | 1,529,196 µs |
| ② 仅 `NCCL_P2P_DISABLE=1`（现有规避）| **87.7 µs** |
| ③ 仅 `NCCL_P2P_LEVEL=PXB` | 1,249,117 µs |
| ④ PXB + 按 socket 过滤 agent | ~~挂死~~ → **见下方更正** |
| ⑤ 仅按 socket 过滤 | ~~挂死~~ → **见下方更正** |

> ### ⚠️ 2026-08-23 更正：④⑤ 的「挂死」是工装 bug，不是结论
>
> 挂死的真因是 `dlhook.c` **按下标过滤**，而 HIP 的 agent 列表顺序不固定：
> `KEEP=4,5,6,7` 在被重排的列表上实际挑到了 GPU3/4/6/7，**漏掉了 GPU5**，
> 该卡的缓冲未被授权 → 段错误。
>
> 换成按句柄过滤的 `dlhook2.c` 后，④ **成立且效果显著**（§九）。
> 原先「跨 socket 那两对的 SHM 缓冲需要双方都能访问」的解释也是错的 ——
> SHM 走的是主机内存，根本不需要 GPU agent 授权。

## 7.4 ★ 为什么这条路不值得再走：环形的算术

即使把过滤做得更精细（比如放过主机内存、只过滤显存），
**环形算法的吞吐由最慢的那条链路决定**：

- 8 卡环里有 **2 条边跨 socket**（`3→4` 和 `7→0`）
- 这两条边只能走 SHM（2.8 GB/s）
- 环形 all-reduce 每条链路都要搬 `2(n−1)/n × N ≈ 1.75N` 的数据

**所以让另外 6 条边跑到 24 GB/s，整体仍然被那 2 条 2.8 GB/s 的边卡住**
—— 与「全部走 SHM」几乎没有区别。实测 ② 的 87.7 µs 已经接近这个上限。

唯一能赢的结构是**分层**（socket 内规约 → 一次跨 socket 交换 → socket 内广播），
因为跨 socket 只搬一次规约后的数据，而不是让它在环里反复穿过。
粗算 1 MB 场景：分层约 400–700 µs vs 全 SHM 约 1500 µs，**2–3 倍**。

**但这需要在框架层自己实现分层 all-reduce**，
而 **DP2×TP4 已经用更简单的方式拿到了更好的结果**
（64 路并发 571.6 输出 tok/s，且完全不需要跨 socket 集合通信）。

## 7.5 定论 ⚠️ 已被 §九 推翻

~~**TP8 在这个缺陷下没有值得投入的出路。**~~
~~`patches/hip-agent-filter/` 的价值仍然成立，但**限于同 socket 作业**。~~

**2026-08-23 更正**：§7.4 的环形算术推理本身没错（跨 socket 那 2 条边确实是瓶颈），
但它假设了跨 socket 边只能跑 2.8 GB/s 的 SHM，而**同 socket 那 6 条边可以跑 P2P** ——
`NCCL_P2P_LEVEL=PXB` 正好做出这个分工。之前之所以没验证成功，是 §7.3 的工装 bug。
实测混合传输的整体带宽是 **5.7 GB/s，而不是全 SHM 的 1.5 GB/s**。见 §九。

---

# 八、`dlhook2.c` —— 按句柄过滤（取代 `dlhook.c`）

## 8.1 为什么必须换

Hook 观察到的实际调用（8 卡可见，用设备 4/5）：

```
 98 次  agents=1c093750,1c099920,1c09fad0,1c0a5c60,1c0abe00,1c0b2320,1c0b87d0,1c0bece0   ← 规范顺序
  1 次  agents=1c0b2320,1c093750,...                     ← GPU5 被提到队首
  1 次  agents=1c0abe00,1c093750,...                     ← GPU4 被提到队首
  1 次  agents=1c093750,...,1c0b87d0,                    ← 只有 7 个，缺 GPU7
```

**HIP 传给 `allow_access` 的 agent 列表顺序不固定，长度也不固定。**
按下标过滤在这些调用上会挑错卡。`DLHOOK_KEEP=0,1,2,3` 侥幸没炸（GPU0 几乎总在低位），
`DLHOOK_KEEP=4,5,6,7` 必炸。

## 8.2 做法

agent 句柄按 GPU 序单调递增。维护一张**全局升序句柄表**，句柄在表中的**名次 = GPU 号**；
首次调用用完整列表播种，句柄表还没认全时不过滤（避免名次漂移）；
**过滤后为空时也不过滤**（宁可不改也不能传空表）。

```bash
gcc -shared -fPIC -O2 -o dlhook2.so dlhook2.c -ldl -lpthread

DLHOOK2_KEEP=0,1,2,3 LD_PRELOAD=/path/dlhook2.so ./your_app   # 只保留这些 GPU 号
DLHOOK2_LOG=1        LD_PRELOAD=/path/dlhook2.so ./your_app   # 打印每次调用及解析出的 GPU 号
```

---

# 九、★ TP8 混合传输（2026-08-23，已验证）

## 9.1 思路

`NCCL_P2P_LEVEL=PXB` 让 RCCL 做出正确分工 —— 实测传输选择是
**96 条 `P2P/IPC`（同 socket）+ 24 条 `SHM/direct`（跨 socket）**。
它之所以仍然慢，是因为那 96 条同 socket 的 IPC 映射，**agent 列表仍是全部 8 个**（缺陷 1）。
用 `dlhook2` 按 socket 把 agent 列表切开，两者一起就成了。

## 9.2 用法

**torchrun（能逐 rank 设环境变量）**

```bash
gcc -shared -fPIC -O2 -o dlhook2.so dlhook2.c -ldl -lpthread
export NCCL_P2P_LEVEL=PXB NCCL_ALGO=Ring
torchrun --no-python --nproc_per_node=8 wrap_persocket.sh your_script.py
```

`wrap_persocket.sh` 按 `LOCAL_RANK` 设 `DLHOOK2_KEEP`（<4 → `0,1,2,3`，否则 `4,5,6,7`）。

**★ vLLM / sglang（worker 由框架自己 spawn，设不了逐 rank 环境变量）→ 用 AUTO**

```bash
-e NCCL_P2P_LEVEL=PXB -e NCCL_ALGO=Ring
-e LD_PRELOAD=/w/dlhook2.so -e DLHOOK2_AUTO=1
```

AUTO 模式拦 HSA 层的 `hsa_queue_create`（第一个参数就是该 worker 用的 agent），
据此判断本进程归属哪个 socket。分组默认 `0,1,2,3|4,5,6,7`，可用 `DLHOOK2_SOCKETS` 改。

> **不要去拦 `hipSetDevice`。** 试过，拿不到真函数时返回 -1 →
> `torch.AcceleratorError: HIP error: unknown error`，8 个 rank 全崩。
> `hsa_queue_create` 在我们本来就在安全 hook 的那一层。

现成脚本：`scripts/serve_tp8_hybrid.sh`（C 线 vLLM）、`scripts/serve_sg8_hybrid.sh`（A 线 sglang）。
**`dlhook2.so` 必须在各自镜像里编译**（两个镜像的 glibc 不同，不能混用）。
**`--disable-custom-all-reduce` 必须保留** —— 自研 all-reduce 自己跨全部 8 rank 做 IPC，绕不过活锁。

## 9.3 实测

| 消息大小 | `PXB + 按socket过滤 + Ring` | 之前最好 `P2P_DISABLE + Ring` | 倍数 |
|---|---|---|---|
| 8 字节（延迟）| **75.9 µs** | 80.3 µs | — |
| 1 MB | **4.59 GB/s** | 1.13 GB/s | **4.1×** |
| 16 MB | **5.74 GB/s** | 1.46 GB/s | **3.9×** |
| 256 MB | **5.67 GB/s**（77.2 ms）| 1.51 GB/s（289.2 ms）| **3.75×** |

**正确性 27/27 `ALL CORRECT`**（bf16/fp32/fp16 × 1/16/80 MB × all_reduce/reduce_scatter/all_gather）。
`ALGO=Tree` 更差（175.5 µs），用 Ring。中毒探测（hycu 中断增量）：**无**。

## 9.4 端到端实测（真实推理负载，同配置只换传输）

> **测量协议**：每组第一次运行必须丢弃（首次 16 并发在 46–248 之间乱跳，
> 第二次起稳定到 ±1%）；`ttft98k.py` 按 `TAG` 变 prompt 前缀，
> 不给 tag 会命中前缀缓存，测冷启必须给唯一 tag。
> 下表并发取预热后 4–6 次均值，冷启用唯一 tag 各 2 次。

| | 混合 | 基线 `NCCL_P2P_DISABLE=1` | 倍数 |
|---|---|---|---|
| C 线 vLLM · 16 并发 | **297.0 tok/s** | 158.3 tok/s | **1.88×** |
| C 线 vLLM · 98K 冷启 TTFT | **114.2 s**（859 tok/s）| 215.6 s（455 tok/s）| **1.89×** |
| C 线 vLLM · 单流（5 次）| 48.76 tok/s | 45.26 tok/s | 1.08× |
| C 线 KV 池 | 1,141,964 tok | 1,144,422 tok | 相同 |
| A 线 sglang · 16 并发 | **223.5 tok/s** | 133.6 tok/s | **1.67×** |
| A 线 sglang · 单流（5 次）| 30.19 tok/s | 26.68 tok/s | 1.13× |
| A 线 KV 池 | 2,125,888 tok（context 1M）| 2,130,240 tok | 相同 |

并发与长 prefill 两个独立口径在 C 线上都落在 **1.88×**。
`dlhook2.so` 确认映射进了全部 8 个 `VLLM::Worker_TP*` / `sglang::scheduler_TP*` 进程。

**两条已证伪的路**（详见 `docs/复盘三` §5.6–5.7）：
- `--disable-custom-all-reduce` **不能去掉** —— 去掉后 TP8 卡死在 CUDA graph 捕获，
  8 张卡中断均匀锁在 398/秒的活锁特征。custom all-reduce 是 8 rank 两两 IPC 的全连通网格。
- `NCCL_ALGO=Tree` **微基准赢、端到端输**（256 MB 带宽 9.02 vs 5.70 GB/s，
  但 16 并发 281 vs 297）。保持 Ring；也别调 `NCCL_*_NCHANNELS`（负收益）。

**注意**：历史文档里「TP8 16 并发 = 102」不能再当基线用，
今天同配置重测基线是 158.3 / 133.6。

## 9.5 仍然不如 DP2×TP4

| 配置 | 16 并发 | KV 池 |
|---|---|---|
| DP2×TP4（生产）| 642 tok/s | 2 × ~512K |
| TP4 单副本 | 315 tok/s | ~512K |
| **TP8 混合 · C 线** | **297 tok/s** | 1,141,964 |
| **TP8 混合 · A 线** | **223 tok/s** | **2,125,888 / context 1M** |
| TP8 基线 | 134–158 tok/s | 同上 |

**生产线继续用 DP2×TP4。** 本节的价值是把 TP8 从「用两倍硬件换 1/3 吞吐」
变成「换 0.83 倍吞吐、得 4 倍 KV 池」——**对需要 1M 上下文的场景这是可接受的交换。**
