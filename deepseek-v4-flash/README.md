# DeepSeek-V4-Flash on K100-AI —— 拉起包迁移中

已在 8×K100-AI 实机跑通四条线路（结论数据）：
- **w8a8 + DSpark 主线**：单流 decode 33.2 tok/s（编程类），DSpark accept 4.38/0.68，KV 池 1.03M
- **int4 线（自研 AWQ 全量 284B 专家量化）**：KV 池 2.97M，64 并发全通（聚合 56-65 tok/s）
- 护栏：DSpark 仅贪心稳定（temperature>0 需强制 top_k=1）；fp8 KV 为该线必需

拉起包化整理完成后在此发布。
