# opencode 编程后端 · 长时综合基准（多线共用）

复刻 opencode 的真实请求形态，覆盖六个场景，按分位数记录。
**对模型中立**：同一套 harness、同一份语料，可用于任意 OpenAI 兼容服务；
只要 seed 与线档案对齐，不同模型、不同配置线之间的数就可以直接横向比。

```bash
# 单条线
python3 run_bench.py --profile qwen3.8-27b --container <容器名> \
                     --port 8100 --hours 6 --out ./bench_out

# 多条线串行（推荐；串行是硬要求，同机并发实例会互相污染测量）
bash run_lines.sh <profile> <线清单文件> [每线小时数]
```

产出 `requests.jsonl`（每请求一行）、`report.md` 与 `container.log`。

## 线档案：模型差异只收在一处

`engine.py` / `corpus.py` / `ocproto.py` / `report2.py` / `cachestat.py`
五个模块**与模型无关**，一行都不必改：指标是 vLLM `bench serve` 口径，
语料按 seed 现生成，工具是标准 OpenAI function 格式，
命中率取自 SGLang 容器日志的 `Prefill batch` 行。

真正随模型而变的只有三件事，全在 `profiles.py`：

| 字段 | 作用 |
|---|---|
| `model` | 请求体的 `model` 字段，即服务端 `--served-model-name` |
| `thinking_off` / `thinking_on` | 开关思考的 `chat_template_kwargs`——**各模型不同名，语义也不同** |
| `ready_token` | 就绪探针在 `/v1/models` 响应里匹配什么 |

### thinking 键为什么必须参数化

实测自镜像内 `sglang/srt/parser/reasoning_parser.py` 的 `DetectorMap`：

```
DetectorMap["qwen3"]       → Qwen3Detector        reasoning_default = "enable_thinking"
DetectorMap["deepseek-v4"] → _DeepSeekV3Detector  reasoning_default = "explicit_thinking"
```

两者**不同名**：一个看 `enable_thinking`，一个看 `thinking`。更要紧的是
`explicit_` 前缀意味着**不显式传就不思考**——所以 DeepSeek-V4 的 `thinking_off`
是**空字典**（不传即关），而不是 `{"thinking": False}`。

给 DeepSeek-V4 传 `enable_thinking` 是无效键。它虽然碰巧也不思考，
但那是巧合而非正确——换个模型就会踩空。

### 新增一个模型

在 `profiles.py` 的 `PROFILES` 里加一项（四个字段齐全即可），
`--profile` 与 `run_lines.sh` 会自动认得。写错名字会直接报错并列出可选值。
**不给默认值是刻意的**：默认值会造成「打错模型名却照跑两小时」这种最贵的失败。

## 为什么要这套，而不是简单压测

| | 朴素压测 | 本套 |
|---|---|---|
| 上下文内容 | 中文散文填充 | **真实代码语料**（60 文件 / 648K 字符，按 seed 现生成） |
| 请求形态 | 裸 user 消息 | **opencode 系统提示 + 10 个工具定义 + 多轮工具循环** |
| 指标 | 单流 tok/s 均值 | **TTFT / ITL / TPOT / E2EL 的 P50/P95/P99**（对齐 vLLM bench serve） |
| 覆盖 | 单流 + 前缀 | 六场景：多轮会话 / 长上下文 / 并发 / 工具调用 / PPT / 长输出 |

中文约 1 字/token、代码约 3–4 字符/token——分词形态与模型负担完全不同，
**用散文测出来的数不能代表编程助手负载**。

## 调研依据（2026-08）

- opencode 是 agentic loop：约 10 个内置工具（每个描述约 150 词）、流式、
  多轮往返、可并行调工具、文件按 `cat -n` 注入、上下文 90–96% 触发压缩
- 指标定义对齐 vLLM `bench serve`：TPOT = (E2EL − TTFT) / (输出 token − 1)
- PPT 生成以 **Marp Markdown** 最适合 LLM（错误率近零、可导出 pptx）

## 六个场景

| | 场景 | 测什么 |
|---|---|---|
| S1 | 多轮编程会话 | 长仓库上下文 + 多轮工具循环；缓存命中随轮次的效果 |
| S2 | 长上下文冷启 | 60K/180K/340K 字符；冷启 TTFT + needle 准确率 |
| S3 | 并发混合 | 4/8/16 路独立会话；聚合吞吐与分位数 |
| S4 | 工具调用密集 | 5 个探针：单工具/并行多工具/复杂参数/选工具/任务规划 |
| S5 | PPT 生成 | marp / python-pptx / reveal.js，**agentic 与直出双路径** |
| S6 | 长输出 | 8K token 连续生成 |

## 关于缓存命中率

SGLang **不打** `Prefix cache hit rate` 这个字符串（实测计数 0），
命中率不能从那里取。`cachestat.py` 改从每条 `Prefill batch` 行的
`#new-token` / `#cached-token` 如实汇总：

```bash
python3 cachestat.py <容器名> [起始HH:MM:SS] [结束HH:MM:SS]
```

这件事很要紧：在 95% 命中的生产口径下，`prompt_tok / TTFT` 量的**不是 prefill 做的功**
——绝大部分 token 根本没被重新计算。真实 prefill 速率必须用 `#new-token` 算。

## 设计中踩到的坑（都已写在代码注释里）

1. **S4 的「单工具」探针原本问上下文里已有的文件**——好的 agent 本就不该调工具，
   测的是题目不对。改成问上下文里没有的文件。
2. **S5 原本要求一次性输出完整 PPT**——但 opencode 的系统提示明确要求
   「少于 4 行文本输出」，模型会先探索再用 `write` 落文件。实测对照：同一请求
   「opencode 提示 + 带工具」只输出 61 token 并调 `list`/`glob`；换中性提示才
   直接吐 5,977 字符 HTML。已改为 agentic 形态，并保留「直出」作为能力基线。
3. **S6 长输出带着 opencode 系统提示会得到 content=0**（预算全进 reasoning）。
   长文必须用中性提示 + 关思考——「怎么关」由线档案给出。
4. **工具调用参数可能被 max_tokens 截断成半截 JSON**，原样回灌会让服务端返回
   HTTP 400。真实客户端也会丢弃或修复，故此处丢弃并计数，不记作服务故障。
5. **服务不可达必须退避**：连接被拒会以毫秒级空转——某次容器崩溃重启的 8 分钟里，
   无退避的驱动产生了 10 万条垃圾记录，把真实数据淹没。

## 起跑前必查

缺一项就可能白跑数小时：

1. 远端 `bench/*.py` 与本地的 md5 **逐项一致**
   ——曾因 `scenarios.py` 未同步，在 27 秒时才截住，差点白跑 6.5 小时
2. 加速卡状态正常、链路宽度满速、故障计数为 0
3. 若该线依赖驱动补丁，确认补丁在**运行中**的驱动里生效
   （符号地址每次开机都变，必须动态查 `/proc/kallsyms`，不能只查文件）
4. 必须分离式启动，否则 ssh 一断就全丢（历史上已因此损失三次长跑）：
   ```bash
   ssh <host> 'setsid nohup bash run_lines.sh <profile> lines.txt 2.0 \
                 > /tmp/runlines.boot 2>&1 < /dev/null &'
   ```

## 输出

- `requests.jsonl`：每请求一行，含 ttft / itl_p50 / itl_p99 / tpot / e2el /
  prompt_tok / completion_tok / 工具调用数 / 各场景专属校验字段
- `report.md`：分场景分位数、S1 缓存随轮次、S3 并发、S4 工具正确性、
  S5 PPT 质量、S2 needle、逐小时退化检测、错误分布
- `container.log`：该线的容器日志（`cachestat.py` 依赖的 `Prefill batch` 行在此）
