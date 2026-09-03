# 07 · C 线 · 8 卡 · TP8 混合传输

| 项 | 值 |
|---|---|
| 卡数 | **8**（跨 socket）|
| 引擎 | **vLLM** |
| 网关 | 无 |
| 特殊 | `dlhook2.so` agent 过滤；`--disable-custom-all-reduce` 必须保留；Tool Call ✓ |

C 线的 8 卡形态。与 A 线同理：并发场景不如 DP2×TP4，长输入同样受 GDN 内核拖累。

```bash
bash ../common/launch.sh 07-C-vllm-tp8
```

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。
