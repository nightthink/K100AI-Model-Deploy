# Qwen3.8-Flash-Next on K100-AI —— 上游阻塞（2026-09-02 定界）

- 官方镜像 `custom:vllm0.29.0-…-qwen3.8flashnext` 的 vLLM wheel 构建仅含 **gfx936** 目标，
  在 gfx928 上 C 算子**静默空跑**（rms_norm 哨兵实验：返回无异常、输出 buffer 零写入），
  引发 MoE 垃圾索引 → GPU 页错误（XID:81）
- 我方 fix928 补丁（torch 版 moe_align/topk_softmax/moe_sum）可压住崩溃，但残余静默数值错误
  无法在用户态穷尽——需厂商以 `PYTORCH_ROCM_ARCH=gfx928;gfx936` 重编 wheel
- FL 家族（vllm-plugin-FL 0.24/0.26）另有 MPClient IPC 缺陷，同样阻塞
- 权重与复现材料已备齐，厂商修复后即可发布拉起包
