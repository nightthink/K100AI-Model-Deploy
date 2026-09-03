# 05 · C 线 · 4 卡 · TP4 生产（512K）

| 项 | 值 |
|---|---|
| 卡数 | **4**（同 socket）|
| 引擎 | **vLLM**（0.21.0 + vllm_hcu，hy3-0706 镜像）|
| 网关 | 无 |
| 上下文 | **512K**（C 线 TP4 做不到 1M：需 61.06 GiB KV/卡，仅有 41.01；vLLM 自算上限 671,744）|
| 投机 | MTP |
| 特殊 | GDN all 模式 + 自研补丁（`dlhook2.so`）；`--disable-custom-all-reduce` 必须保留 |

**已知短板**：长输入崩塌 —— 2k→64k 输入 decode 43.06→6.65 tok/s
（根因：aiter GDN 内核缺 gfx928 配置，护栏 1·八）。短输入场景仍是 C 线最优。

```bash
bash ../common/launch.sh 05-C-vllm-tp4
```

详细依据（参数理由、实测过程、踩坑）见同目录 [NOTES.md](NOTES.md)。

## 镜像与权重（首次拉起自动补齐 = S2；也可手工执行）

```
镜像  docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm-ubuntu22.04-dtk26.04-hy3-0706
权重  bash ../common/fetch_hf_model.sh Qwen/Qwen3.8-27B $MODELS_ROOT/Qwen3.8-27B
```
完整镜像 URL 即上行；`launch.sh` 的 S2 会检测本地有无，缺才拉。`SKIP_S2=1` 可跳过。
