# deploy-kit —— 工程现场部署包

## ★ 拉起包（唯一交付物，一线一包，2026-08-31 起）

```bash
bash deploy-kit/build_package.sh 09       # 仓库侧：打 09 号线一个包（28K-152K/包）
bash deploy-kit/build_package.sh all      # 十条线各打一包
# 现场（任意主机任意目录）：
tar xzf q38-kit-09-*.tar.gz && cd q38-kit-09
bash up.sh                # 拉起本线（体检→S1→S2..S8→真实请求冒烟，全自动）
bash up.sh GPUS=4,5,6,7   # 带参拉起（KEY=VAL 透传）
bash up.sh status         # 观察      bash up.sh stop   # 停止(S9)
```

每包只含本线所需（02 包内嵌其依赖的 01 副本配置与网关件）；
除 **镜像** 与 **模型权重**（S2 自动拉取）外全自包含；目录中立；十步契约内建。
升级=新包覆盖解包（勿 rm -rf，保留 triton 缓存）。

**每条线的参数小结、实测数据与选择指引**（含 09/10/01/02 的循环负载对比表）：
见 [`scripts/README.md`](scripts/README.md) 总索引。


**只含拉起与运行所需**。打包本目录送到现场，解包后**镜像与模型权重由首次拉起自动补齐**
（S2 自举：本地有则跳过，缺则从 harbor / hf-mirror 拉；`SKIP_S2=1` 可关）。
每个配置 README 都写明完整镜像 URL 与手工命令。调测记录、测量工装、研究散件在仓库根部，不随包。

```
deploy-kit/
├── build_package.sh        ★ 打包器：`bash build_package.sh <NN|all>` → q38-kit-<NN>-<日期>.tar.gz
├── scripts/
│   ├── up.sh               ★ 拉起包统一入口（打包时置于包根；拉起/停止/状态/S1-S10）
│   ├── 01-A-sglang-tp4/    ★ 4 卡推荐（sglang·1M·无网关）    serve.sh + README
│   ├── 02-A-dp2tp4-router/ ★ 8 卡推荐（DP2×TP4 + 粘性网关）  up.sh 一键全套
│   ├── 03…08-*/            其余配置（TP8 / 基线 / C 线 vLLM）
│   ├── common/             launch.sh（十步契约）、machine_prep（S1）、网关、部署映射
│   ├── lib/                stages.sh（S3/S5/S7/S8/S9）、carrier.sh（载体适配）
│   ├── 09-A-int8-tp4-…/    ★ INT8 低延迟档（单流d86p2300 聚合d148）
│   ├── 10-A-int8-tp4-…/    ★ INT8 高QPS档（聚合d243p3990 单流d33）
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

## 硬性前提（只有两个，体检会逐条明说）

1. **DCU 驱动已装载**（`/dev/kfd` 存在、hycu 绑定 ≥4 卡、`/opt/hyhal` 在位）。
   **装驱动超出本包边界**——体检失败会直接指明："先按海光官方文档安装 DTK/驱动，本包不负责装驱动"。
2. **网络可达 `harbor.sourcefind.cn:5443`（镜像）与 `hf-mirror.com`（权重）**。
   不可达时走**离线兜底**：在有网机器按各配置 README「镜像与权重」节的完整 URL
   `docker pull` + `docker save` 镜像、拉好权重目录，拷到现场后
   `docker load` / 放入 `$MODELS_ROOT`——S2 检测到本地已有即跳过拉取，其余流程不变。

其余一切（内核参数、ACS、systemd 自愈、补丁、1M 农场、拉起、验证）都由本包自动完成。

## 旧式流程（保留，仓库内开发用）

`quickstart.sh`（整仓五步一键）与 `deploy_to_host.sh <目录>` + `launch.sh <配置名>`
仍可用；**现场交付一律走上方拉起包**（一线一包，up.sh 已内含体检/S1/S2-S8/冒烟/停止/状态）。

**目录中立**：包可铺在任意路径（脚本以自身位置定位，`WORK` 由 launch.sh 导出）。
仅两个外部路径约定，均可覆盖：`MODELS_ROOT`（默认 `/data/models`，权重根）、
`/opt/hyhal`（宿主驱动，硬约定）。已验证：铺到全新目录后 01 配置 S3→S9 全绿。

