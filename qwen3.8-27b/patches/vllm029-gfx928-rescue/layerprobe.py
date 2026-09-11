"""路线乙：Flash-Next 逐层发散检测（无参考实现）。

## 为什么不做逐层对拍

本机没有能跑起来的 qwen4_exp 参考实现（336G 权重 + 自定义架构，CPU 放不下）。
所以不比对绝对值，而是找**自身统计量发生突变**的位置：
正常前向的逐层 hidden state 范数应平稳变化，爆炸 / 塌缩 / NaN 都会明确指向某一层。

## 被测对象的形态（已从源码核实）

  入口   hidden_states = embed_tokens(input_ids).repeat(1, hc_count)   → [T, 4*2560=10240]
  每层   layer(...) -> (hidden_states[T,10240], mlp_out[T,2560], injection[T,4])
  出口   hyper_connection_mixer.combine_and_mix(...) 的第二项 sample_hidden_states

## ★ 判读时必须记住的陷阱

HyperConnection 采用**延迟合并**：layer 返回的 hidden_states **尚未合并本层的 mlp_out**，
要到下一层的 combine_and_mix 才合并。所以第 i 层打印的 hidden_states
反映的是「第 i-1 层已合并 + 第 i 层未合并」的状态，**不能直接当作第 i 层的输出**。
injection 经 2*sigmoid(·/HC) 决定残差注入强度，若它异常会逐层放大，故单独记录。

用法：作为 sitecustomize 注入（PYTHONPATH），只在 rank 0、只对前 N 次前向打印。
"""
from __future__ import annotations

import os

_on = os.getenv("LAYERPROBE", "1").strip().lower() in ("1", "true", "yes", "on")
_max_passes = int(os.getenv("LAYERPROBE_PASSES", "2"))

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
)


def _install() -> None:
    import torch
    from vllm.models.qwen4_exp.amd import model as m

    Layer = m.Qwen4ExpDecoderLayer
    orig_forward = Layer.forward
    state = {"pass": 0, "seen": set()}

    def _stat(t):
        if t is None:
            return "None"
        if not isinstance(t, torch.Tensor):
            return type(t).__name__
        f = t.detach().float()
        n_nan = int(torch.isnan(f).sum().item())
        n_inf = int(torch.isinf(f).sum().item())
        finite = f[torch.isfinite(f)]
        if finite.numel() == 0:
            return "全部非有限! nan=%d inf=%d" % (n_nan, n_inf)
        return "mean|x|=%.4f max|x|=%.3f std=%.4f%s" % (
            finite.abs().mean().item(),
            finite.abs().max().item(),
            finite.std().item(),
            ("  ★nan=%d inf=%d" % (n_nan, n_inf)) if (n_nan or n_inf) else "",
        )

    def patched(self, hidden_states, prev_block_output, prev_injection,
                positions, **kw):
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0

        idx = getattr(self, "_probe_idx", None)
        if idx is None:
            idx = state["seen"].__len__()
            state["seen"].add(id(self))
            self._probe_idx = idx

        # ★ 计数修正（2026-09-12）：原先把 state["pass"] += 1 放在 idx==0 分支里，
        #   而 show 在其之前求值 —— 第 2 次前向刚进 L00 就把自己关掉，
        #   结果只打印了 L00 一行。改为「进入 L00 时先记次数，再据此决定是否显示」。
        if idx == 0:
            state["pass"] += 1
        cur = state["pass"]
        show = (rank == 0 and cur <= _max_passes)

        if show and idx == 0:
            print("[layerprobe] ===== 第 %d 次前向（T=%s）=====" %
                  (cur, tuple(hidden_states.shape)), flush=True)
            print("[layerprobe]  入口 hidden: %s" % _stat(hidden_states), flush=True)

        out = orig_forward(self, hidden_states, prev_block_output, prev_injection,
                           positions, **kw)

        if show:
            h, blk, inj = out
            lt = getattr(self, "layer_type", "?")
            # block_input（即 xn 经 hc_gate_mix 后送进 attn/mlp 的那一路）不在返回值里，
            # 但 mlp_out 与 inj 已足以定位；inj 单独看符号分布，因为
            # 2*sigmoid(inj/HC) 才是真正的注入系数，mean|x| 会高估饱和程度。
            extra = ""
            if isinstance(inj, torch.Tensor):
                g = 2.0 * torch.sigmoid(inj.detach().float() / 4.0)
                extra = "  gate: mean=%.3f min=%.3f max=%.3f 饱和占比=%.1f%%" % (
                    g.mean().item(), g.min().item(), g.max().item(),
                    100.0 * (g > 1.9).float().mean().item())
            print("[layerprobe]  L%02d %-16s h: %-52s | mlp_out: %-46s | inj: %s%s"
                  % (idx, lt, _stat(h), _stat(blk), _stat(inj), extra), flush=True)
        return out

    Layer.forward = patched
    print("[layerprobe] 已挂载：Qwen4ExpDecoderLayer.forward（rank0，前 %d 次前向）"
          % _max_passes, flush=True)


if _on and not _helper:
    try:
        _install()
    except Exception as _e:
        print("[layerprobe] 安装失败: %r" % (_e,), flush=True)
