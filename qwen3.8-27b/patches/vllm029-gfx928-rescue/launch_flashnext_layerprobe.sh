#!/bin/bash
# Qwen3.8-Flash-Next 路线乙（2026-09-12 凌晨）：逐层发散检测
#
# 背景：fix928-v2 已把 VMFault 修到归零（前八次恒为 Invalid=16/VMFault=24），
# 服务能起能出 token，但输出是纯数字乱码，且 prompt_logprobs 证明
# **prefill 阶段即失真**（" Paris" logprob −12.9，正常应接近 0）。
#
# 逐个算子排查已排除全部已知路径而无一命中：权重加载 / tokenizer / 线性层 /
# rms_norm / 激活 / MoE / QSA / GDN 两端 / KV cache / device capability / PDL /
# HyperConnection 五个 triton kernel / 基础 GEMM 与 softmax / all-reduce 通信 /
# triton 本身（四项独立证据）。再逐个猜就是穷举。
#
# 乙的做法：给 Qwen4ExpDecoderLayer.forward 挂 hook，逐层记录
# hidden_states / mlp_out / injection 的 mean|x| / max|x| / std / NaN，
# 找**第一个统计量突变的层**。不做逐层对拍——本机没有能跑起来的 qwen4_exp
# 参考实现（336G 权重 + 自定义架构）。
#
# 与 09-11 前八次的根本区别：前八次是在**猜参数**（六个假说，五个被干预实验证伪），
# 而本次针对的是 09-02 就已坐实、但 09-11 未被启用的根因：
#
#   镜像的 vLLM 编译扩展只含 gfx936（实测 _C.abi3.so / _moe_C_stable / _C_stable），
#   在 gfx928 上**不报错、不写出**（静默空跑）→ 输出 buffer 保持 torch.empty 的
#   未初始化内容 → 垃圾索引 → 越界读显存 → VMFault。
#
# 2026-09-11 哨兵实测确认静默空跑的算子共 11 个，本次全部覆盖：
#   · 7 个 CustomOp 家族  → --compilation-config '{"custom_ops":["none"]}' 走 forward_native
#   · 3 个 MoE 辅助算子   → fix928 v1 的 torch 替换
#   · top_k_per_row_decode（QSA 在 ROCm 上实际走的那个）→ fix928 **v2** 新增
#
# 参数逐个核实过（engine/arg_utils.py 源码，--help 在无 GPU 容器里不可用）：
#   已确认存在：language-model-only / compilation-config / kernel-config / moe-backend /
#               enforce-eager / enable-prefix-caching / block-size / max-num-batched-tokens
#   **已剔除**：--no-enable-flashinfer-autotune（本镜像 arg_utils 命中 0，会启动失败）
#   **暂不用**：--load-format fastsafetensors（取值合法性存疑，只影响加载速度）
#
# 首跑刻意保留 NCCL_P2P_DISABLE=1：本次已同时更换「权重 / 架构 / 补丁」三项变量，
# 再放开 P2P 就是第四项。v4 补丁虽已在 13 线验证稳定，起来之后再单独放开测增益。
set -u
IMG=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext
M=/data/models/Qwen3.8-Flash-Next
W=/data/kq0828/fnv2probe_work
PORT=8133
NAME=q38-fnv2probe
CFGDIR=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/configs

mkdir -p "$W/tritoncache" "$W/vllmcache"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

sudo docker rm -f "$NAME" >/dev/null 2>&1; sleep 3

log "起服：fix928-v2 + custom_ops=none + eager + language-model-only"
sudo docker run -d --name "$NAME" \
  --network host --privileged --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$M":/models/target:ro \
  -v "$W/tritoncache":/tritoncache -e TRITON_CACHE_DIR=/tritoncache \
  -v "$W/vllmcache":/root/.cache/vllm \
  -v /data/kq0828/fnv2probe:/fix928:ro -e PYTHONPATH=/fix928 \
  -e LAYERPROBE=1 -e LAYERPROBE_PASSES=2 \
  -v "/data/kq0828/E512N80_K100AI.json":"$CFGDIR/E=512,N=80,device_name=K100_AI.json":ro \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_PLUGINS= \
  -e VLLM_CUDART_SO_PATH=/opt/dtk/lib/libamdhip64.so \
  --entrypoint bash "$IMG" -c "
    vllm serve /models/target --served-model-name fn \
      --host 0.0.0.0 --port $PORT \
      --dtype bfloat16 --tensor-parallel-size 8 \
      --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.90 --enforce-eager --language-model-only \
      --compilation-config '{\"custom_ops\":[\"none\"]}' \
      --kernel-config '{\"ir_op_priority\":{\"rms_norm\":[\"native\"],\"fused_add_rms_norm\":[\"native\"]}}' \
      --trust-remote-code" \
  >/dev/null

log "等就绪（≤50 分钟；每 15s 探一次，容器死则提前退出）"
for i in $(seq 1 200); do
  curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -aq '"id"' && { log "READY（第 $i 次探测）"; break; }
  sudo docker ps --format '{{.Names}}' | grep -q "$NAME" || { log "✗ 容器已退出"; break; }
  sleep 15
done

log "----- 关键判据 -----"
L=$(sudo docker logs "$NAME" 2>&1)
echo "  fix928-v2 装载:  $(grep -ac 'fix928-v2' <<<"$L")"
echo "  Invalid address: $(grep -ac 'Invalid address access' <<<"$L")   ← 前八次恒为 16"
echo "  KERNEL VMFault:  $(grep -ac 'KERNEL VMFault' <<<"$L")           ← 前八次恒为 24"
echo "  MoE 缺配置告警:  $(grep -ac 'Using default MoE config' <<<"$L")"
echo "  使用自造配置:    $(grep -ac 'E=512,N=80' <<<"$L")"
sudo docker logs "$NAME" > /data/kq0828/fnv2/log.txt 2>&1
log "完整日志: /data/kq0828/fnv2/log.txt（$(wc -l < /data/kq0828/fnv2/log.txt) 行）"

if curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -aq '"id"'; then
  log "----- 首光探测：先看能否出字，再看字是否正常 -----"
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"fn","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":64,"temperature":0}' \
    2>/dev/null | head -c 1200
  echo
  log "----- 第二问：检验是否复读机（09-02 的残留症状）-----"
  curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"fn","messages":[{"role":"user","content":"写一个 Python 快速排序函数。"}],"max_tokens":200,"temperature":0}' \
    2>/dev/null | head -c 2000
  echo
else
  log "✗ 未就绪，最后 40 行日志："
  tail -40 /data/kq0828/fnv2/log.txt
fi
log "FNV2_DONE"
