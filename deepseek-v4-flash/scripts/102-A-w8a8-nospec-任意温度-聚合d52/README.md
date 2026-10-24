# DSv4-02 · 0731-w8a8 无投机 · TP8（任意温度稳线）

101 线的安全后备：同镜像同权重谱系（剔除 DSpark 张量），不开投机。

## 实测

- temperature=0.7 × 8 并发：**8/8 全通，聚合 52.2 tok/s**（任意温度稳定）
- 单流 ~12.3 tok/s（无投机的代价）

以上为 2026-08-14 定稿时的数（P2P 禁用）。

### 2026-09-12：P2P 解禁 + 卖点复验

本线自 2026-09-12 起默认 `NCCL_P2P_LEVEL=SYS`。

**复验本线卖点**（P2P 开启，故意不加 `top_k` 护栏——加了就等于没验"任意温度"）：
temperature=0.7 + top_p=0.95 × 8 并发 **8/8 成功，聚合 50.1 tok/s**，
与定稿时的 52.2 同量级。「任意温度稳定」这一定位仍然成立。

> ⚠ **本线未单独做 P2P 的 A/B**。改用 P2P 的依据来自 101 线的单变量对照
> （整机吞吐 +26.8%、并发 16 路 decode +28.7%、冷启 prefill +47.9%），
> 本线与它同镜像、同权重谱系、同 inner 补丁，差别只在不开投机，故一并采用。
> **要引用本线自己的 P2P 收益数字，必须另跑一轮，不能套用 101 的。**

## 定位

必须支持采样（temperature>0）+ 高并发的场景；或作为 101 线故障时的降级线。
其余边界（Think kwargs、kv-cache-dtype、就绪时长、热身）同 101 线 README。

## 硬边界

0. **驱动必须已打 v4 补丁**（`hycu.ko` 的 `gpu_dma_buf_attach` 一字节改）。
   未打补丁而开 P2P，跨 socket 会触发驱逐活锁。无法打补丁时，请把 `serve.sh`
   里的 `NCCL_P2P_LEVEL=SYS` 改回 `NCCL_P2P_DISABLE=1`。
1. custom AR 开不了，与 P2P 无关（本镜像是上游原版，TP8 且无 XGMI 时直接
   return；QuickAllReduce 白名单无 gfx928），`--disable-custom-all-reduce` 保留。

权重：`$DSV4_MODELS_ROOT/dsv4-0731-w8a8`（quant/ 配方与 101 线权重同批产出）。
