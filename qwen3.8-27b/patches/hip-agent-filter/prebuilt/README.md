# 预编译产物

| 文件 | 来源 | 说明 |
|---|---|---|
| `dlhook2.so` | 本目录上级 `dlhook2.c` | 机器上的 `dlhook2-sg.so` 与它 **md5 相同**，只是 sglang 线的别名 |

校验：`sha256sum -c SHA256SUMS`。重编译按上级目录脚本（hipcc/gcc 命令见 `dhfix.sh` 等）。
部署位置：两台机器 `/data/q38-work/dlhook2.so`（及别名 `dlhook2-sg.so`）。
