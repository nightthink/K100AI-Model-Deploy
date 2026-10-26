# 19 · Qwen3.8-27B BF16 · 厂商vLLM 0.28.1 · TP8性能配置

> **验证状态：性能实验配置，完整标准周期未完成。** 初始化与有限 API 的历史通过结果，不等于当前源码包入口验收、满 1M 请求、合格性能矩阵或生产适用性已经通过。

## 合同与来源

用于与13的八卡标准负载作历史对照，保留17双卡配置不变。复用既有打包器和 `up.sh → S3–S8`；须接入现有后台 Worker、状态与归属互证，不另建生命周期，保留01／13及旧配置线的既有行为。

固定同一厂商镜像摘要，保留HCU插件、`TRITON_ATTN`、gfx928数学库、64-token块对齐、MTP3与细粒度前缀缓存。厂商原配方只明确TP2／32K，因此本配置必须先通过真实初始化、有限功能及完整标准周期，不能以配置文件存在宣称已支持或更快。

来源：<https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/vllm/qwen3.8.md#qwen38-27b-ifb-k100_ai-2x-vllm-0281>。实际版本以发行元数据为准，不根据镜像tag推断。

镜像按摘要固定，不随源码分发镜像或权重：

```text
harbor.sourcefind.cn:5443/hcu/admin/base/custom@sha256:2eff683dd747db3484a1b523c05e49d75bb45ed6724b4d1238344385c7e88c9d
```

## 配置与前提

| 项目 | 选择 |
|---|---|
| GPU／TP | `0,1,2,3,4,5,6,7`／TP8，跨socket |
| 模型 | 现有 `$MODELS_ROOT/Qwen3.8-27B-1M`，只读；模型根同时只读挂载以解析链接 |
| Tokenizer | 与13相同的hygon目录两文件只读覆盖，保留农场聊天模板，不换量化权重 |
| 精度 | BF16权重与显式BF16 KV；4个KV heads由核心原生复制，不假定容量随卡数线性增长 |
| 上下文 | `CTX=1000000`，使用现有农场的YaRN参数，不修改文件或加新缩放覆盖 |
| Prefill／显存 | chunked prefill开启，`MAX_BATCHED_TOKENS=16384`，`MEM_FRAC=0.90` |
| 请求并发 | 标准量具串行；引擎 `MAX_NUM_SEQS=16` |
| API | `qwen38`，`PORT=8117`，`HOST=127.0.0.1`；与其他配置串行运行 |
| 缓存 | 本包按实例独立，不借用01／13／17／18的JIT缓存 |

使用前必须准备上述只读 1M 模型农场、hygon tokenizer 两文件、固定镜像及已生效的运行驱动 v4 前提。缺项失败；本线不自动生成农场、下载权重或安装依赖。公共 S3 继续检查跨 socket P2P，不挂旧 agent 过滤库，不以关闭 custom all-reduce 或热重载驱动绕过前提。默认仅 loopback 监听，无需开放 UFW；改为非 loopback 前须先限制访问来源。

模型根 `MODELS_ROOT` 与端口 `PORT` 可由既有入口覆盖，表中的模型及 tokenizer 子目录名仍按本线配置解释，不增加任意模型路径参数。默认端口与17／18相同，起服前须按原包身份停止占用目标卡组或端口的实例，不能按名称批量清场。

## 已有历史结果与限制（2026-10-25）

- 厂商运行栈已完成 TP8 模型初始化，六项有限 API 检查通过，不是“模型不能运行”。
- 一次标准负载在 7200 秒执行时限到达时结束，完整标准周期未完成，未取得合格的完整报告及四指标矩阵。
- 到达执行时限是已观察到的结束条件，不足以确定超时根因。不能把部分样本或13的历史数字作为本线的完整性能结果，也不按卡数从18外推。
- `CTX=1000000` 是配置上限，不证明满 1M 请求、长文本精度或长期稳定性。没有已验证的相对 SGLang 性能收益，不作为生产推荐。
- 上述属于历史证据；后续源码包须重新证明正式 `start/state/stop`、Worker快照与实际容器身份互证。直接执行 `serve.sh` 或注入 `LAUNCH_ID` 不能替代入口验收。

## 有限验证与标准性能

先核对模型配置、索引及tokenizer摘要与历史参照一致；链接同文件不等于重做全部大权重哈希。READY、容器／包／launch身份、实际TP／上下文／BF16配置必须互证，再检查语义、SSE、usage与标准请求字段。

只使用包内原字节 `bench/standard_load.py`，固定原默认seed7、温度0.7、top_p0.8及强制满额输出，不注入thinking=false。输入45K–350K，输出500–4K，按实际长度分桶；必须完整周期、自证全部达标后才使用四指标矩阵。MTP与NEXTN、TRITON与FA3等引擎差异单列，不宣称逐请求配对或同时段因果对照。

```bash
# 先进入部署目录：私仓根执行 cd deploy-kit；公开仓根执行 cd qwen3.8-27b。
# 只构建19。
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/19-C-vllm0281-tp8 -p 'test_*.py'
OUT_DIR="$(mktemp -d)"
bash build_package.sh 19 "$OUT_DIR"

# 打包后；先按原包正式停止归属互证的占卡模型，不能按名称扫停。
bash up.sh start GPUS=0,1,2,3,4,5,6,7 PORT=8117 READY_TIMEOUT=2400 MAX_RESTARTS=0 KEEP_ON_FAIL=1
bash up.sh state PORT=8117
bash up.sh benchmark PORT=8117 CYCLES=1 OUT=/tmp/owned-standard-19
bash up.sh stop PORT=8117
```

上述命令描述使用与复现方法，不表示本次文档更新已重跑实机。性能检查须设置执行时限并保留归属与失败证据，之后由本包正式 stop 收尾；不得为通过而降上下文、关 MTP／前缀、换精度或修改量具。标准周期未完成时不填写合格性能矩阵，350K周期本身也不证明满1M精度、长期稳定性或A3最终保障门。内部现场日志与执行预算不随公开源码发布。
