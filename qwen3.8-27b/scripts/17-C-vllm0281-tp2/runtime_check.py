"""在模型进程启动前核验真实发行身份，不改写任何安装包或版本字段。"""

import importlib.metadata as metadata
import json
import os
import sys

from packaging.version import Version


def runtime_versions():
    versions = {}
    for name in ("vllm", "vllm_hcu"):
        value = metadata.version(name)
        version = Version(value)
        if version.epoch != 0 or version.release != (0, 28, 1):
            raise RuntimeError(f"17线要求{name}的release为0.28.1，实际为{value}")
        versions[name] = value
    for name in ("torch", "triton", "aiter", "transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    print("Q38_RUNTIME_VERSION=" + json.dumps(runtime_versions(), ensure_ascii=False), flush=True)
    from vllm.platforms import current_platform

    platform = type(current_platform).__module__ + "." + type(current_platform).__qualname__
    if not type(current_platform).__module__.startswith("vllm_hcu."):
        raise RuntimeError(f"17线要求厂商HCU平台，实际选择{platform}")
    print("Q38_HCU_PLATFORM=" + platform, flush=True)
    # 参数逐项传给厂商CLI，不拼接Shell，也不替换CLI对配置的校验。
    os.execvp("vllm", ["vllm", "serve", *sys.argv[1:]])


if __name__ == "__main__":
    main()
