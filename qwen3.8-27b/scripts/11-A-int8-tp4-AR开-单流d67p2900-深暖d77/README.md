# 11 · INT8 + DFlash2 + custom-AR —— 深上下文冠军档（2026-09-02）

> ## ⛔ 2026-09-11：INT8 全系暂不可投产
>
> **VMFault 两次复现**：2026-09-10 存活 19 分 26 秒、2026-09-11 存活 37 分钟。
> 两次都发生在**低负载、显存充裕**时（KV 池占用 4%、单请求解码），
> **无一致前兆**——首次伴随 `pool memory leak` 告警 104 次，第二次该告警 0 次，
> 故该线索已被撤回。v4 驱动补丁不是解法（dmabuf 与 INT8 内核是两个子系统）。
>
> 嫌疑收敛到 **INT8 W8A8 权重 + 镜像的 INT8 内核路径**本身，尚未定位。
> 本线拉起包本身可正常启动并通过自证，但**不建议投入生产**。
> bf16 主线（01 / 13）不受影响。


> ### ⚠ 2026-09-09 更正：custom AR 实际从未启用
>
> 实测（四个 rank 一致）：
> ```
> disable_custom_all_reduce=False                        ← 我们确实没禁用
> [AR] All-reduce call path: NCCL (custom AR disabled)   ← SGLang 因无 XGMI 自行关闭
> ```
> 全程 `gpu_gpukfd_gpuvm_import_dmabuf = 0`，**没有建立任何进程间 IPC 映射**。
>
> ⇒ 本线历史上归因于「AR 开」的性能收益（短 +40% / 深暖 +64%），
> **实际来自同批改动的其他参数**（DocPang 模板、pack min-q 2048、mem 0.95、
> graphs 1-8、mamba 32）。**实测数据仍有效，错的是原因。**
>
> ⇒ 「同 socket 门禁是因为 AR/IPC」这一理由不成立；门禁暂时保留，
> 但依据改为「跨 socket 的 NCCL 传输侧已知问题」（见 13/14 线 README）。
>
> ⇒ 由此也说明：**hycu.ko 的 v2 补丁对本线无收益**——它修的是进程间 IPC
> 映射中毒，而本线不触发该路径。



4 卡（同 socket 组），Qwen3.8-27B 海光官方 Channel-INT8 + z-lab DFlash2 投机 + **custom all-reduce 开启**。
09 的直系升级：同权重同镜像同补丁链，差异全在参数体系（DocPang v30 吸收，dp30 系列 A/B 定界）。

## 与 09 对比（同探针同卡组）

| 口径 | 09 | **11** |
|---|---|---|
| decode 短 | ~60 | **58-67** |
| decode 64K | ~54 | **63** |
| decode 120K 暖 | 52.3 | **53-77（瞬时 108）** |
| 关键差异 | AR 关 | **AR 开（主引擎：短 +40% / 深暖 +64%）** |

其余参数差异：DocPang chat 模板、pack min-q 2048（09=4096）、mem 0.95（09=0.85）、
graphs 1-8（09=1-4/8）、mamba 32（09=16）、max-total-tokens 1M、mamba-track-interval 16384。

## 镜像与权重

- 镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828`
- 目标：ModelScope `hygon/Qwen3.8-27B-Channel-INT8-w8a8`（S2 自动拉取）
  ——**切勿换 Freaksterz SmoothQuant 版**：实测在本树上使 DFlash2 accept 归零（dp30d/e 定界）
- 草稿：HF `z-lab/Qwen3.8-27B-DFlash2`（serve.sh 自动派生 v2 目录）

## 硬性边界

1. **卡组必须同 socket**（0,1,2,3 或 4,5,6,7）——AR/P2P 跨 socket 触发 52ms 活锁
2. 就绪探测用 `/model_info`（`/health` 会注入生成请求）
3. bs16×投机=调度器崩：graphs 止于 8
4. temperature>0 依赖 v122/ 三件套挂载（本包已带）

## 用法

拉起包统一入口：`bash up.sh`（体检→S1→S2..S8→冒烟）；`status` / `stop` 同 09。
调参：`GPUS=4,5,6,7 PORT=8112 bash up.sh 11`。

## 聚合与边界补充（2026-09-02 验收）

- 8 路 × 512：**8/8 全通，聚合 145.7 tok/s**（与 09 的 148 同级——聚合受 verify 限制，AR 主要提单流）
- 已知记账瑕疵：8 路投机突发后 idle 审计差 384/1M 槽，会触发严格检查自杀；
  本线以 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` 降为告警（DocPang v1.3.1
  rootfix 是根修，待兼容性验证后可换根修）
- 120K 暖轮 decode 存在 53-77 波动（accept 内容依赖 + 实例态），冷轮 23-32

## 能力四项验证（2026-09-03 实测）

| 项 | 结果 |
|---|---|
| 1M 上下文 | 配置 1048576；350K 循环全通；1M 单发同树验证（09 线，99.7万 tok） |
| Think | 默认思考先行；`chat_template_kwargs.enable_thinking=false` 可关（实测） |
| Tool call | `tool_choice=auto` 实测通过（选函数/参数/finish_reason 全对）；`required` 未声明支持 |
| 温度采样 | temp/top_p/默认采样全通（v122）；`sampling_seed` 精确复现未声明支持 |
