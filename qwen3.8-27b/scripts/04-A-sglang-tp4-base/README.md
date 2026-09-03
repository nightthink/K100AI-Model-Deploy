# 04 · A 线 · 4 卡 · TP4 基线（对照/回退用）

| 项 | 值 |
|---|---|
| 卡数 | 4 |
| 引擎 | sglang |
| 网关 | 无 |
| 上下文 | 1M |

未调优的基准配置（无 chunked-prefill 调参、无投机旋钮）。SLA 扫描 `a-tp4` 用的就是它。
日常部署用 **01**；本配置留作性能回归的对照组与最简回退。

```bash
bash ../common/launch.sh 04-A-sglang-tp4-base
```

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
农场  bash ../patches/model-1m-farm/mk_1m_farm.sh   # launch 会自动做
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。
