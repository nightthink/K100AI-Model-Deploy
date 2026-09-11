# 注入方式说明（fix928-v2 与 layerprobe 如何共存）

## 坑：PYTHONPATH 下只能有一个 sitecustomize.py

Python 在 `PYTHONPATH` 里只会导入**第一个**找到的 `sitecustomize` 模块。
所以不能把 `fix928-v2` 和 `layerprobe` 各做成一个 `sitecustomize.py` 并列放，
必须由一个入口文件串联。现场目录结构：

```
/data/kq0828/fnv2probe/
  sitecustomize.py    ← 入口：runpy 依次执行下面两个
  _fix928v2.py        ← fix928-v2 本体（算子改道）
  _layerprobe.py      ← 逐层发散检测（forward hook）
```

顺序有讲究：**先装 fix928-v2**（否则模型根本起不来，VMFault 会复现），
再挂 layerprobe 的 hook。

## 各文件对应关系

| 仓库文件 | 现场落点 | 作用 |
|---|---|---|
| `sitecustomize-v2.py` | `_fix928v2.py` | 11 个 gfx928 静默空跑算子改道 |
| `layerprobe.py` | `_layerprobe.py` | 逐层 hidden/mlp_out/injection 统计 |
| `launch_flashnext_fix928v2.sh` | — | 只跑 fix928-v2（正常服务） |
| `launch_flashnext_layerprobe.sh` | — | fix928-v2 + layerprobe（诊断） |

## 开关

- `FIX928=0` 停用算子改道
- `LAYERPROBE=0` 停用逐层检测
- `LAYERPROBE_PASSES=N` 只打印前 N 次前向（默认 2，避免刷屏）
