import json, shutil, os
import sys
P = (sys.argv[1] if len(sys.argv) > 1 else "/data/models/Qwen3.8-27B-W8A8-INT8") + "/config.json"
B = P + ".orig_hf"
if not os.path.exists(B):
    shutil.copy2(P, B)
c = json.load(open(P))
q = c["quantization_config"]
n = 0
for g in (q.get("config_groups") or {}).values():
    for k in ("weights", "input_activations", "output_activations"):
        v = g.get(k)
        # actorder 只对 group/tensor_group 策略有意义；本模型 weights 是 channel、
        # activations 是 token，actorder 是量化工具留下的无效字段。
        # 镜像里的 compressed-tensors 0.15.0.1 会因此在 pydantic 校验时直接拒绝加载。
        if isinstance(v, dict) and v.get("actorder") is not None and \
           v.get("strategy") not in ("group", "tensor_group"):
            v.pop("actorder"); n += 1
json.dump(c, open(P, "w"), indent=2)
print("  移除了 {} 处无效的 actorder；备份 {}".format(n, B))
