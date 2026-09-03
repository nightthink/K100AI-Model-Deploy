# 第三方声明

本仓库不再分发基础 Docker 镜像与模型权重（拉起时自动获取，或按各线 README 离线获取）。

- **SGLang**（Apache-2.0）：各配置线在 SourceFind SGLang 镜像上以运行时补丁形式运行，
  被修改/挂载覆盖的 SGLang 文件保留上游许可证。
- **DocPang/qwen38-k100ai-int8-optimization**（MIT）：09/11 线的 raw-q8 verifier 思路与
  v1.2.2 非贪婪修复三文件、chat 模板源自该项目；12 线直接运行其 v1.3.1 成品镜像
  （镜像获取方式与 SHA256 见 12 线 README）。特此致谢。
- **z-lab/dflash**（MIT）：DFlash2 草稿模型与算法上游。
- **vLLM**（Apache-2.0）：C 线（05-08）补丁的上游。
- 海光/SourceFind 的 DTK、闭源算子库与基础镜像版权归其所有；本仓库仅含用户态配置与补丁，
  不含任何官方闭源组件的再分发。
