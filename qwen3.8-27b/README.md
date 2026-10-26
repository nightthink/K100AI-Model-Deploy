# deploy-kit —— 工程现场部署包

## ★ 拉起包（现场部署单位，一线一包）

源码可按实际验证状态公开；目录存在、能够打包、有限功能通过、完整性能通过与生产适用性是不同结论。既有精选线保持原选型与警示，新增的 **17／18／19** 分别为厂商 vLLM 0.28.1 的 TP2 有限功能配置、TP4／TP8 性能实验配置，不替代 13 线的推荐地位。

```bash
bash deploy-kit/build_package.sh 09       # 仓库侧：打 09 号线一个包（28K-152K/包）
bash deploy-kit/build_package.sh all      # 对全部编号目录打包，含实验线；不表示逐线验收通过
# 现场（任意主机任意目录）：
tar xzf q38-kit-09-*.tar.gz && cd q38-kit-09
bash up.sh                # 拉起本线（体检→S1→S2..S8→真实请求冒烟，全自动）
bash up.sh GPUS=4,5,6,7   # 带参拉起（KEY=VAL 透传）
bash up.sh status         # 观察      bash up.sh stop   # 停止(S9)

# ★ 同机多实例（同一个包、同一个脚本，只改卡与端口）
bash up.sh GPUS=0,1,2,3 PORT=8101      # 实例一
bash up.sh GPUS=4,5,6,7 PORT=8102      # 实例二（名字按端口自动派生）
bash up.sh status/logs/restart/stop PORT=8102   # 管理时以端口为句柄，定点到该实例
```

每包只含本线所需（02 包内嵌其依赖的 01 副本配置与网关件）；
我方运行脚本随包提供，镜像、模型及派生资产按各线 README 准备。S2 仅补齐该线元数据明确声明的内容；18／19 不自动生成 1M 模型农场或准备 tokenizer。目录中立，复用十步契约，不另建启动流程。
升级=新包覆盖解包（勿 rm -rf，保留 triton 缓存）。

**每条线的参数小结、实测数据、验证限制与选择指引**（含 17／18／19）：
见 [`scripts/README.md`](scripts/README.md) 总索引。


**只含拉起与运行所需**。已有配置按其 S2 声明获取镜像和模型权重
（本地有则跳过；`SKIP_S2=1` 可关闭获取），不能据此推断所有实验配置都支持从零自举。
每个配置 README 都写明完整镜像 URL 与手工命令。调测记录、测量工装、研究散件在仓库根部，不随包。

| vLLM 配置 | 默认范围 | 已有历史结果与限制 |
|---|---|---|
| [17 · TP2](scripts/17-C-vllm0281-tp2/README.md) | BF16，32K 上下文 | 七项有限功能检查通过；不承诺生产全量验收、1M 或速度收益 |
| [18 · TP4](scripts/18-C-vllm0281-tp4/README.md) | BF16 权重／KV，1M 配置上限 | 初始化及六项有限 API 检查通过；标准周期在执行时限内未完成，无合格完整性能矩阵 |
| [19 · TP8](scripts/19-C-vllm0281-tp8/README.md) | 同上，跨 socket | 同上；额外依赖已生效的运行驱动 v4 前提 |

上述为历史验证，不是当前运行健康状态，也不自动证明后续源码包已通过正式入口验收。

下面是配置源树布局示意；公开副本只保留所选择公开的编号目录。
```
deploy-kit/
├── build_package.sh        ★ 打包器：`bash build_package.sh <NN|all>` → q38-kit-<NN>-<日期>.tar.gz
├── scripts/
│   ├── up.sh               ★ 拉起包统一入口（打包时置于包根；拉起/停止/状态/S1-S10）
│   ├── 13-A-bf16-tp8-…/    ★ 推荐主力（bf16 TP8 全P2P + custom-AR；01 的八卡版）
│   ├── 01-A-sglang-tp4/    ★ 4 卡推荐（sglang·1M·无网关）    serve.sh + README
│   ├── 02-A-dp2tp4-router/ ★ 8 卡推荐（DP2×TP4 + 粘性网关）  up.sh 一键全套
│   ├── 03…08-*/            其余配置（TP8 / 基线 / C 线 vLLM）
│   ├── common/             launch.sh（十步契约）、machine_prep（S1）、网关、部署映射
│   ├── lib/                stages.sh（S3/S5/S7/S8/S9）、carrier.sh（载体适配）
│   ├── 09-A-int8-tp4-…/    ⛔ INT8 低延迟档（单流d86p2300 聚合d148）—— INT8 全系暂不可投产
│   ├── 10-A-int8-tp4-…/    ⛔ INT8 高QPS档（聚合d243p3990 单流d33）—— 同上
│   ├── 11/12/14-*int8*/    ⛔ 其余 INT8 线（VMFault 未解，14 未发布）
│   ├── 17-C-vllm0281-tp2/   厂商 vLLM 0.28.1，TP2 有限功能配置
│   ├── 18-C-vllm0281-tp4/   TP4 性能实验配置，完整标准周期未完成
│   ├── 19-C-vllm0281-tp8/   TP8 性能实验配置，完整标准周期未完成
│   └── README.md           ★ 总索引：配置一览、推荐编号、契约逐阶段说明
├── patches/
│   ├── hip-agent-filter/   dlhook2.c 源码 + prebuilt/dlhook2.so（SHA256）
│   ├── rccl-acs-topo/      ACS 清除脚本 + acs-clear.service（S1 用）
│   ├── vllm-gdn-cache/ vllm-triton-attn/   C 线挂载的 7 个补丁
│   ├── runtime/            drco 算子幂等注册、INT8 config 修正（quant 用）
│   └── model-1m-farm/      1M 上下文农场配方（mk_1m_farm.sh + config.json）
├── quant/                  量化方案（完整自足：第三方方案全文抄录 + 下载脚本 + 双版 config）
└── docs/方案脚本规范-设计文档.md   十步契约（S1-S10）的定义本体 —— 本包脚本的方法论规范
```

## 通用前提与各线额外准备

1. **DCU 驱动已装载**（`/dev/kfd` 存在、hycu 绑定 ≥4 卡、`/opt/hyhal` 在位）。
   **装驱动超出本包边界**——体检失败会直接指明："先按海光官方文档安装 DTK/驱动，本包不负责装驱动"。
2. **网络可达 `harbor.sourcefind.cn:5443`（镜像）与 `hf-mirror.com`（权重）**。
   不可达时走**离线兜底**：在有网机器按各配置 README「镜像与权重」节的完整 URL
   `docker pull` + `docker save` 镜像、拉好权重目录，拷到现场后
   `docker load` / 放入 `$MODELS_ROOT`——S2 检测到本地已有即跳过拉取，其余流程不变。

各线额外前提以其 README 和 `serve.sh` 的声明为准：18／19 要预先具备既有 1M 模型农场及指定的 hygon tokenizer，19 还要求跨 socket 所需的驱动条件已在运行时生效；不通过新装或热重载驱动、重启宿主来完成本次源码公开。

17／18／19 默认仅监听 `127.0.0.1`，不需要开放 UFW 端口。若操作者改为非 loopback 监听，须先限制访问来源。三线复用正式 `start/state/stop` 与归属互证；发布前须验证当前包的入口，不能用直接运行 `serve.sh` 或历史 READY 记录代替。

## 旧式流程（保留，仓库内开发用）

`quickstart.sh`（整仓五步一键）与 `deploy_to_host.sh <目录>` + `launch.sh <配置名>`
仍可用；**现场交付一律走上方拉起包**（一线一包，up.sh 已内含体检/S1/S2-S8/冒烟/停止/状态）。

**目录中立**：包可铺在任意路径（脚本以自身位置定位，`WORK` 由 launch.sh 导出）。
仅两个外部路径约定，均可覆盖：`MODELS_ROOT`（默认 `/data/models`，权重根）、
`/opt/hyhal`（宿主驱动，硬约定）。已验证：铺到全新目录后 01 配置 S3→S9 全绿。
