# 10 · INT8-W8A8 无投机 · TP4（高 QPS 批量档）

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


**目录名里的数字**：单流 decode 33 tok/s / prefill 3990 tok/s；聚合最高 decode 243 tok/s（8 路）/ prefill 同 3990（分块 prefill 串行流水，并发下不叠加）。

| 项 | 值 |
|---|---|
| 卡数 | **4**（同 socket 卡组）|
| 引擎 | sglang 0.5.12+das（与 09 同镜像）|
| 网关 | 无（与 09 双姿态部署时由外部网关分流）|
| 镜像 | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-K100AI-qwen3.8-0828`（自动 pull）|
| 权重 | ModelScope `hygon/Qwen3.8-27B-Channel-INT8-w8a8`（58G，S2 自动 git clone；无 draft）|

## 实测（验证机B）

| 场景 | 值 | 备注 |
|---|---|---|
| **8 路聚合** | **242.6 tok/s** | 比投机档高 64%——批量下 draft 算力反成负担 |
| 单流 decode | 33.1 tok/s | 低延迟场景请用 09 |
| 96K prefill | **3990 tok/s**（服务端 25s；TTFT 35.2s 含 tokenize）| 全场最快 prefill |
| 并发上限 | 9 路（mamba 48 槽 ÷ 每请求约 5 槽）| 再高要挤 KV 池，不划算 |

## 一键

```bash
bash common/launch.sh 10-A-int8-tp4-nospec-单流d33p3990-聚合d243p3990
```

## 硬性前提 / 边界

同 09 号线第 1-3、5 条（/opt/hyhal、禁 /health、TTFT 暖场、勿挂厂商整链）。
本线无投机，cuda-graph 到 bs16 安全；`--max-mamba-cache-size 48` 已按 9 路并发配平。

## 选择指引

吞吐优先（批处理、离线蒸馏、RAG 批量摘要）选本线；交互流量选 **09**。
