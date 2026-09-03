# DSv4-02 · 0731-w8a8 无投机 · TP8（任意温度稳线）

01 线的安全后备：同镜像同权重谱系（剔除 DSpark 张量），不开投机。

## 实测

- temperature=0.7 × 8 并发：**8/8 全通，聚合 52.2 tok/s**（任意温度稳定）
- 单流 ~12.3 tok/s（无投机的代价）

## 定位

必须支持采样（temperature>0）+ 高并发的场景；或作为 01 线故障时的降级线。
其余边界（Think kwargs、kv-cache-dtype、就绪时长、热身）同 01 线 README。

权重：`$DSV4_MODELS_ROOT/dsv4-0731-w8a8`（quant/ 配方与 01 线权重同批产出）。
