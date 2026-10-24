#!/bin/bash
# A 臂对照链：就绪等待 → 判据一 → 端到端等价 → A@8000 → A@32000
# 远端脱离运行，不挂在 ssh 会话上（本地 ssh 被 OOM 杀过，教训已记两次）
set -u
W=$HOME/patchwork
echo "===== A 臂对照链开始 $(date '+%F %T') ====="

READY=0
for i in $(seq 1 40); do
  H=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8113/v1/models 2>/dev/null)
  A=$(sudo docker ps --format '{{.Names}}' | grep -cx q38-armA2)
  [ "$H" = "200" ] && { echo "✓ 就绪于 $(date '+%T')（第 $i 次）"; READY=1; break; }
  [ "$A" != "1" ] && { echo "✗ 容器已退出于 $(date '+%T')"; sudo docker logs --tail 30 q38-armA2 2>&1|tail -30; exit 1; }
  echo "  [$i] $(date '+%T') HTTP=$H $(sudo docker logs q38-armA2 2>&1 | grep -aoE 'Load weight end|Capture cuda graph' | tail -1)"
  sleep 30
done
[ "$READY" != "1" ] && { echo "✗ 超时未就绪"; exit 1; }

echo; echo "═══ 判据一 ═══"
sudo docker exec q38-armA2 bash -c 'grep -o "num_v_heads // self.num_k_heads in \[[0-9, ]*\]" /usr/local/lib/python3.10/dist-packages/sglang/srt/models/qwen3_5.py' 2>&1 | sed 's/^/  白名单: /'
curl -s -m 120 -o /dev/null -w "  冒烟 HTTP %{http_code}\n" \
  -X POST http://127.0.0.1:8113/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"1+1=?只回数字"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
L=$(sudo docker logs q38-armA2 2>&1)
AC=$(grep -ac 'GDNFUSE\] ACTIVE' <<<"$L"); FB=$(grep -ac 'GDNFUSE\] FALLBACK' <<<"$L")
echo "  ACTIVE=$AC  FALLBACK=$FB  （A 臂须 ACTIVE=0 且 FALLBACK>0）"
[ "$AC" -ne 0 ] || [ "$FB" -eq 0 ] && { [ "$AC" -ne 0 ] && { echo "  ✗✗ 判据一未过，中止"; exit 1; }; }
[ "$FB" -eq 0 ] && { echo "  ✗✗ 未见 FALLBACK，中止"; exit 1; }
echo "  ✓✓ 判据一通过"

echo; echo "═══ 端到端等价性（固定 prompt，与 B 臂同参）═══"
python3 $W/fixed_probe.py 8113 armA 2>&1
HA=$(sha256sum $W/fixed_armA.txt 2>/dev/null | cut -d' ' -f1)
HB=$(sha256sum $W/fixed_armB.txt 2>/dev/null | cut -d' ' -f1)
echo "  armA sha256: ${HA:0:40}"
echo "  armB sha256: ${HB:0:40}"
if [ "$HA" = "$HB" ]; then echo "  ✓✓ 两臂端到端输出逐字节相同"
else echo "  ⚠⚠ 两臂输出不同:"; diff $W/fixed_armA.txt $W/fixed_armB.txt | head -14 | sed 's/^/     /'; fi

echo; echo "═══ A@8000（复现基线，对照首次 步/s 24.89）═══"
python3 $W/q38_measure.py 8113 8000 512 5 1 2>&1

echo; echo "═══ ★ A@32000（决定性对照，判定量=步/s）═══"
python3 $W/q38_measure.py 8113 32000 512 3 1 2>&1

echo; echo "═══ 复核 ═══"
L=$(sudo docker logs q38-armA2 2>&1)
echo "  ACTIVE=$(grep -ac 'GDNFUSE\] ACTIVE' <<<"$L")  FALLBACK=$(grep -ac 'GDNFUSE\] FALLBACK' <<<"$L")"
echo "  OOM=$(grep -aci 'out of memory' <<<"$L")  Traceback=$(grep -ac Traceback <<<"$L")  非200=$(grep -aF 'POST /v1' <<<"$L"|grep -vc '200 OK')"
echo "  下载 $(du -sb /data1/models/qwen38-flashnext-int8 2>/dev/null | awk '{printf "%.1f GB", $1/1e9}')/183.4 GB"
echo "ARMA2_CHAIN_DONE $(date '+%T')"
