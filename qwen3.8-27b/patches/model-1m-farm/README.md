# Qwen3.8-27B 1M 软链农场（A 线全部配置的 @requires）

原生 262144 → 1M 上下文的方法：**不改权重**，只换 `config.json` 的 rope 段
（`rope_type: yarn, factor 4.0, original_max_position_embeddings: 262144`）。
`mk_1m_farm.sh <原模型目录> <农场目录>` 一键重建。
`yarn_override.json` 是同一 text_config 的独立 JSON 形态（C 线早期 `--hf-overrides` 用法留档）。
