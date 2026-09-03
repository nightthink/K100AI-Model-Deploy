# 量化方案（Qwen3.8-27B 线）—— 本目录自足，无需外查

## 一句话现状
本线用**第三方 SmoothQuant/GPTQ W8A8 INT8**（Freaksterz），**方案全文已抄录在本目录**；
我方贡献两个修正件（见下）。当前被镜像打包缺陷挡在正确性上
（CLAUDE.md 护栏 1·十一：能跑、每步快 89%、但输出乱码），**不用于生产**，材料齐备待镜像修复。

## 目录内容

```
quant/
├── README.md                        ← 本文件
├── dl_int8.sh                       权重下载（hf-mirror，8 路并发，断点续传，索引校验）
└── third-party/
    ├── freaksterz-w8a8/             ★ 量化方案完整抄录
    │   ├── MODEL_CARD.md            方法论全文（旋转+SmoothQuant+GPTQ 三段式，含精度对照表）
    │   ├── recipe.yaml              llmcompressor 配方（GPTQModifier / W8A8 / actorder:static）
    │   ├── quantization/*.py ×9     完整实现：absorb-rotate / apply-smoothing-rot /
    │   │                            build-gptq-rot / capture-actmax-rot / alpha-proxy-rot /
    │   │                            graft-mtp / make-corpus / patch-vision-rotation /
    │   │                            verify-rotation-cpu   —— 可从 bf16 权重完整复现量化
    │   ├── config.json.orig         量化产物的原始 config（含会被 compressed-tensors
    │   │                            0.15.0.1 拒绝的 actorder 字段）
    │   ├── config.json.fixed        我方修正后的版本（可直接用）
    │   └── LICENSE
    └── docpang-int8-docs/           DocPang 的 K100AI INT8 实践文档
        （README/TP1/TP2/TP4/PERFORMANCE/RELEASE_NOTES + LICENSE；
         其 .hip 源码与预编译 .so 在仓库 lab/third-party/docpang-dflash2/native_ext/，研究素材不随包）
```

## 从零到起服务（机器上只缺镜像与 bf16 权重时）

```bash
bash dl_int8.sh /data/models/Qwen3.8-27B-W8A8-INT8        # ① 下权重（或用 quantization/ 自产）
python3 ../patches/runtime/fix_actorder_int8_config.py    # ② 修 config（等价于直接用 config.json.fixed）
# ③ 起服务：drco 幂等补丁挂两处 + dtype 对齐（护栏 1·十一 的全部三道坎）
P=/usr/local/lib/python3.10/dist-packages
Q=../patches/runtime/drco_idempotent.py
MODEL_PATH=/data/models/Qwen3.8-27B-W8A8-INT8 CTX=262144 \
  EXTRA_ARGS='--dtype float16' \
  EXTRA_ENV_IN='-e SGLANG_MAMBA_CONV_DTYPE=float16' \
  EXTRA_MOUNT="-v $Q:$P/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py:ro \
               -v $Q:$P/lmslim/vllm_compat/direct_register_custom_op.py:ro" \
  bash ../scripts/common/launch.sh 01-A-sglang-tp4
```

复现记录与失败边界：仓库 `docs/bf16优化穷举-2026-08-28-通宵.md` §5.5（不随包）。

## 相关但不属于本线
DeepSeek-V4-Flash 的自研 W8A8/INT4 方案在其自己的仓库（scripts/quant_w8a8_0731.py、scripts/int4/）。
