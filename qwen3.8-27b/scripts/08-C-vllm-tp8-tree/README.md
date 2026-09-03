# 08 · C 线 · 8 卡 · TP8 · NCCL_ALGO=Tree（取舍版）

| 项 | 值 |
|---|---|
| 卡数 | 8 |
| 引擎 | vLLM |
| 网关 | 无 |

与 07 唯一差别：`NCCL_ALGO=Tree`。**取舍不是提升**：prefill 快 14%、并发慢 5.6%
（护栏 9）。只在「重 prefill、轻并发」的批处理场景下选它。

```bash
bash ../common/launch.sh 08-C-vllm-tp8-tree
```

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。
