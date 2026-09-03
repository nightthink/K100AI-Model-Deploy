# 06 · C 线 · 4 卡 · TP4 参数化副本（C 线 DP2 构件）

| 项 | 值 |
|---|---|
| 卡数 | 4（`GPUS` 可传）|
| 引擎 | vLLM |
| 网关 | 本体无；与 `common/serve_router.sh` 组合可拼 C 线 DP2×TP4 |
| 上下文 | `MAXLEN`/`YARN_FACTOR` 可传 |

`GPUS`/`PORT`/`NAME`/`MAXLEN`/`YARN_FACTOR`/`MAMBA_ALL` 全部参数化，
起两份 + 网关即成 C 线版的 02。**A 线的 02 更强**，此配置仅当必须用 vLLM 时用。

```bash
GPUS=0,1,2,3 PORT=8101 NAME=q38c-a bash ../common/launch.sh 06-C-vllm-tp4-replica
GPUS=4,5,6,7 PORT=8102 NAME=q38c-b bash ../common/launch.sh 06-C-vllm-tp4-replica
UPSTREAMS="http://127.0.0.1:8101,http://127.0.0.1:8102" bash ../common/serve_router.sh
```

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。
