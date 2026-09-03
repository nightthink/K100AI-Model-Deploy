# K100AI 模型部署实践

海光 **K100-AI（gfx928，8×64GB）** 上大模型部署的**成功配置线**发布仓。
只收录实机验收通过的**拉起包**（自包含、可观察、幂等、可完整撤场）；
调优过程与实践记录不在本仓。定期更新。

## 覆盖模型

| 模型 | 状态 | 目录 |
|---|---|---|
| **Qwen3.8-27B** | ✅ 12 条配置线，全部包级验收通过 | [`qwen3.8-27b/`](qwen3.8-27b/) |
| DeepSeek-V4-Flash | ✅ 已跑通（w8a8+DSpark 单流 33.2 / int4 线 64 并发），拉起包化迁移中 | [`deepseek-v4-flash/`](deepseek-v4-flash/) |
| Qwen3.8-Flash-Next | ⏳ 上游阻塞：官方 vllm0.29 镜像 wheel 仅编 gfx936，等厂商重编 | [`qwen3.8-flash-next/`](qwen3.8-flash-next/) |
| GLM-5.3-Flash | ❌ 判定不可行：原生 FP8（本卡慢 4×），bf16 反量化 ~640GB 超 8 卡显存 | [`glm-5.3-flash/`](glm-5.3-flash/) |

## 拉起包怎么用

```bash
# 仓库侧：打包某条线（或 all）
cd qwen3.8-27b && bash build_package.sh 11
# 现场（任意主机任意目录；也可直接取 Releases 里的现成包）
tar xzf q38-kit-11-*.tar.gz && cd q38-kit-11
bash up.sh          # 体检 → 机器准备 → 镜像/权重自动获取 → 起服 → 真实请求冒烟
bash up.sh status   # 观察 S1-S10 进展与实例状态
bash up.sh stop     # 完整撤场（按标记前缀精确识别本包资产）
```

硬性前提只有两个：DCU 驱动已装载（/dev/kfd、/opt/hyhal）；网络可达
harbor.sourcefind.cn:5443 与 hf-mirror.com/ModelScope（不可达走各线 README 的离线兜底）。

## Qwen3.8-27B 推荐速览（详见 [qwen3.8-27b/scripts/README.md](qwen3.8-27b/scripts/README.md)）

| 线 | 定位 | 关键实测 |
|---|---|---|
| **11** ★ | 深上下文主力（INT8+DFlash2+custom-AR，4卡） | 120K 暖 decode 53-77（峰 108）；1M 上下文 |
| **10** ★ | 高 QPS（INT8 无投机，4卡） | 8 路聚合 242.6 tok/s；prefill 3990 |
| **12** | 深 prefill 冠军（社区 v1.3.1 整包，离线镜像） | 120K 冷 TTFT 59s；350K decode 34.5 |
| 09 | 低延迟后备（11 的前身） | 单流 86（代码类） |
| 01/02 | bf16 全精度线（单副本 / 双副本+粘性网关） | prefill 密集型场景 |

## 更新纪律

- 只发布通过「解包 → up → S1-S8 → 真实请求 → 完整停止」验收的线
- 每线 README 写明镜像 URL/digest、权重来源、参数小结、实测数据与硬边界
- 版本以 git tag + Releases（含预打好的 kit tar.gz）发布
