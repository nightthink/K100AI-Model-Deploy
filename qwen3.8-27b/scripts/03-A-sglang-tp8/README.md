# 03 · A 线 · 8 卡 · TP8 混合传输（仅单用户长输出时考虑）

| 项 | 值 |
|---|---|
| 卡数 | **8**（跨 socket）|
| 引擎 | **sglang** |
| 网关 | 无 |
| 上下文 | 1M |
| 特殊 | 同 socket 走 VRAM P2P、跨 socket 走 SHM（`dlhook2-sg.so` 按 socket 过滤）；需 `NCCL_P2P_DISABLE=1` |

**定位**：只有「单用户 + 长输出」时比 02 强（单流 decode 34.8 tok/s @64K 曾测）。
**并发一上来就输**：16 并发 102 vs TP4 的 315。并发场景一律用 02。

```bash
bash ../common/launch.sh 03-A-sglang-tp8
```

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
农场  bash ../patches/model-1m-farm/mk_1m_farm.sh   # launch 会自动做
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。
