"""把本机调优的 fused_moe triton 配置装进 sglang 的配置目录。

## 为什么

服务日志明写：

    Using default MoE kernel config. Performance might be sub-optimal! Config file not found at
    .../moe_runner/triton_utils/configs/triton_3_5_0/E=256,N=256,device_name=K100_AI,dtype=int8_w8a8,per_channel_quant=True.json

即 W8A8 的 triton fused_moe 一直在用默认块配置。2026-09-13 profiler：fused_moe_kernel 占 decode 一步的
24.5%（17 ms/步，197 µs/次），是 bf16fp32 GEMM 补丁之后的头号单点。

## 配置从哪来

`tests/prof/moe_tune/`（上游 sglang benchmark/kernels/fused_moe_triton 的 tuning_fused_moe_triton.py +
common_utils.py，原样）在本机 8 卡上以 ray 并行跑：

    python3 tuning_fused_moe_triton.py --model /models --tp-size 8 --dtype int8_w8a8 --per-channel-quant \
        --disable-shared-experts-fusion --tune --batch-sizes 1 2 4 6 8 12 16 24 32 48 64 96 128 256 512 1024 2048 4096

产物 json 放在本目录 moe_configs/ 下，容器启动时由本脚本按运行中的 triton 版本拷到 sglang 配置目录。
sglang 找不到当前 triton 版本目录时会退而搜其它版本目录，但同名文件在正确版本目录里最稳。

未提供 json 时本脚本什么都不做（打印提示），不影响启动。
"""
import glob
import os
import shutil

MARKER = "PATCH_MOE_CONFIG"
SRC = "/patches/moe_configs"

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "*.json")))
    if not files:
        print(f"[{MARKER}] {SRC} 下无 json，跳过（fused_moe 走默认配置）")
        raise SystemExit(0)
    import triton
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config as m
    cfg_dir = os.environ.get("SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(m.__file__)))
    dst = os.path.join(cfg_dir, "configs", f"triton_{triton.__version__.replace('.', '_')}")
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(dst, os.path.basename(f)))
        print(f"[{MARKER}] ✓ {os.path.basename(f)} → {dst}")
    print(f"[{MARKER}] {len(files)} 个配置就位；起服日志应出现 'Using MoE kernel config from'")
