# repair.py 提取件（0.5.18 基版专用）

15 / 16 线以 SGLang **0.5.18** 为基版，但 gfx928 的 paged-varlen 正确性修复
**上游至今未修**，仍需沿用 0.5.12 镜像里的 `repair.py`。

该文件出自海光 SourceFind 的 K100 补丁树，**不是我方原创，故不随包发布**；
包内只提供提取脚本，由现场从自己已有的 0.5.12 镜像取出：

```bash
bash patches/repair-from-0512/extract_repair.sh            # 取到脚本同目录
SRC_IMG=<自定义0.5.12镜像> bash patches/repair-from-0512/extract_repair.sh /目标目录
```

取出后放到本线 `minichain5n/` 同级即可（`sitecustomize.py` 按镜像内绝对路径加载，
现场若路径不同，改 `sitecustomize.py` 第 16 行的 `run_path` 参数）。

**⇒ 15 / 16 线在现场需要两个镜像**：0.5.18 跑服务、0.5.12 仅供提取此文件。
