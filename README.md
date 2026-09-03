# K100AI 模型部署实践

海光 **K100-AI（gfx928，8×64GB）** 上大模型部署的**成功配置线**发布仓。
只收录实机验收通过的**完整拉起包**（自包含、可观察、幂等、可完整撤场），
且只发布有独立优点的精选线；调优过程与研究性中间版本不在本仓。定期更新。

当前收录：

| 模型 | 配置线 | 说明 |
|---|---|---|
| **Qwen3.8-27B** | 六条精选线 → [`qwen3.8-27b/`](qwen3.8-27b/) | 27B 稠密（GDN 混合），INT8/bf16 双谱系 |
| **DeepSeek-V4-Flash** | 两条生产线 → [`deepseek-v4-flash/`](deepseek-v4-flash/) | 284B MoE（激活 13B），W8A8+DSpark |

其他模型的拉起包就绪后陆续发布。

## 六条线怎么选（详见 [qwen3.8-27b/scripts/README.md](qwen3.8-27b/scripts/README.md)）

| 线 | 定位 | 卡数 | 关键实测 |
|---|---|---|---|
| **11** ★ | 深上下文主力（INT8+DFlash2+custom-AR） | 4 | 120K 暖 decode 53-77（峰 108）；1M 上下文 |
| **10** ★ | 高 QPS（INT8 无投机） | 4 | 8 路聚合 242.6 tok/s；prefill 3990 |
| **12** | 深 prefill 冠军（社区 v1.3.1 整包，离线镜像） | 4 | 120K 冷 TTFT 59s；350K decode 34.5 |
| 09 | 低延迟（11 的前身，短上下文最快） | 4 | 单流 86 tok/s（代码类，瞬时 105-113） |
| 01 | bf16 全精度 + 1M 上下文 | 4 | prefill 密集型场景；批量段 1028-2739 tok/s |
| 02 | bf16 双副本 + 粘性网关 | 8 | 聚合 prefill 2.91× / decode 2.55× |

## DeepSeek-V4-Flash 两条线（详见 [deepseek-v4-flash/scripts/](deepseek-v4-flash/scripts/) 各线 README）

| 线 | 定位 | 卡数 | 关键实测 |
|---|---|---|---|
| **01** ★ | 贪心主线（W8A8+DSpark 投机） | 8 | 单流 33.2 tok/s；8/10 并发稳定；⚠ 仅 temperature=0 |
| 02 | 任意温度稳线（无投机） | 8 | temp0.7×8 并发全通，聚合 52.2 tok/s |

⚠ 权重为自量化产物（包内 `quant/` 含完整配方，从公开 bf16 原模型生成）；
01 线务必阅读其 README 的四条硬边界（贪心限制/kv-dtype/Think kwargs/热身）。

## 拉起包怎么用

```bash
# 直接取 Releases 里的现成包（或仓库侧 cd qwen3.8-27b && bash build_package.sh 11）
tar xzf q38-kit-11-*.tar.gz && cd q38-kit-11
bash up.sh          # 体检 → 机器准备 → 镜像/权重自动获取 → 起服 → 真实请求冒烟
bash up.sh status   # 观察 S1-S10 进展与实例状态
bash up.sh stop     # 完整撤场（按标记前缀精确识别本包资产）
```

硬性前提只有两个：DCU 驱动已装载（/dev/kfd、/opt/hyhal）；网络可达
harbor.sourcefind.cn:5443 与 hf-mirror.com/ModelScope（不可达走各线 README 的离线兜底）。

## 发布纪律

- 只发布通过「解包 → up → S1-S8 → 真实请求 → 完整停止」验收、且有独立优点的配置线
- 每线 README 写明镜像 URL、权重来源、参数小结、实测数据与硬边界
- 版本以 git tag + Releases（含预打好的 kit tar.gz）发布
