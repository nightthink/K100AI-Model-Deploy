# 11 · INT8 + DFlash2 + custom-AR —— 深上下文冠军档（2026-09-02）

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
