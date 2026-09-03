#!/bin/bash
# ============================================================================
# 阶段契约实现 —— 见 docs/方案脚本规范-设计文档.md §二
# 依赖 lib/carrier.sh。所有函数只调 carrier_*，不直接写载体命令。
# 每个函数对应文档里的一个阶段，注释标明「保证 / 判据 / 失败」。
# ============================================================================
. "$(dirname "${BASH_SOURCE[0]}")/carrier.sh"
_say(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- S3 前提校验
# 保证：所有前提在**此刻**成立（S1 做过不代表现在还成立）
# 判据：逐项读回实际值    失败：拒绝启动
# 用法： stage3_gate "0,1,2,3" 8 /data/models/X img:tag /w/dlhook2.so ...
stage3_gate(){
  local gpus="$1" expect_gpu="${2:-8}"; shift 2
  local fail=0
  _say "S3 前提校验"

  # -- 通用 --
  local nb; nb=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo ?)
  [ "$nb" = "0" ] && _say "  ✓ numa_balancing=0" || { _say "  ✗ numa_balancing=$nb（应为 0）"; fail=1; }

  # -- 硬件耦合：海光 / PCIe ACS --
  local acs; acs=$(sudo lspci -vvv 2>/dev/null | grep -c 'ACSCtl:.*SrcValid+')
  [ "$acs" = "0" ] && _say "  ✓ ACS 置位 0 个" || { _say "  ✗ ACS 仍有 $acs 个桥置位，P2P 会被 IOMMU 重定向且不报错"; fail=1; }

  # -- 硬件耦合：海光 hycu 绑定 --
  local bound; bound=$(ls -d /sys/bus/pci/drivers/hycu/0000:* 2>/dev/null | wc -l)
  [ "$bound" -ge "$expect_gpu" ] && _say "  ✓ hycu 已绑定 $bound 张卡" || { _say "  ✗ hycu 只绑定 $bound/$expect_gpu"; fail=1; }

  # -- 硬件耦合：本机异质，同 socket 且同型号才算干净 --
  if [ -n "$gpus" ]; then
    local ids; ids=$(for h in /sys/class/hwmon/hwmon*; do
      [ "$(cat $h/name 2>/dev/null)" = hycu ] || continue
      cat "$(readlink -f $h/device)/device" 2>/dev/null; done | sort -u | tr '\n' ' ')
    _say "  · 本机在册芯片型号：$ids（0x6210=480SIMD/400W，0x6211=512SIMD/350W）"
  fi

  # -- 通用：依赖产物 --
  local p
  for p in "$@"; do
    if [ -e "$p" ]; then _say "  ✓ 存在 $p"
    else _say "  ✗ 缺 $p"; fail=1; fi
  done

  # -- 通用：无他人占卡、无孤儿上下文 --
  local orphan; orphan=$(for pid in $(ls /sys/class/kfd/kfd/proc/ 2>/dev/null); do [ -d /proc/$pid ] || echo x; done | wc -l)
  [ "$orphan" = "0" ] && _say "  ✓ 无孤儿 KFD 上下文" || { _say "  ✗ 有 $orphan 个孤儿 KFD，显存未回收"; fail=1; }

  [ "$fail" = 0 ] || { _say "S3 未通过，拒绝启动"; return 1; }
  _say "S3 通过"; return 0
}

# ---------------------------------------------------------------- S5 清理上一实例
# 保证：无残留、资源已释放   判据：按真实标识查 + 资源回落   失败：拒绝启动
stage5_cleanup(){
  local name="$1" wait_s="${2:-300}"
  carrier_exists "$name" || { _say "S5 无残留"; return 0; }
  _say "S5 清理 $(carrier_name "$name")"
  local base; base=$(kfd_others "$name")   # 停之前先数清同机他者
  carrier_unmanage "$name"        # ★ 先解自动重启，否则 stop 完又被拉起
  carrier_stop "$name" 60
  carrier_remove "$name"
  stage9_wait_resources "$wait_s" "$base"
}

# ---------------------------------------------------------------- S7 就绪/失败/进度
# 保证：能在实例外部判定 就绪 / 失败 / 进行中
# 判据：就绪=HTTP 200；失败=重启计数超限、或实例已退出（★退出码 0 也算失败）
# 失败：立刻返回并打印根因，不等满超时
stage7_wait_ready(){
  local name="$1" url="$2" timeout_s="${3:-3900}" max_restarts="${4:-2}"
  local t0=$SECONDS
  _say "S7 等待就绪（≤$((timeout_s/60)) 分钟；重启 >$max_restarts 次判失败）"
  while [ $((SECONDS-t0)) -lt "$timeout_s" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null)" = "200" ] && {
      _say "S7 就绪，用时 $(( (SECONDS-t0)/60 )) 分 $(( (SECONDS-t0)%60 )) 秒"; return 0; }
    local rc; rc=$(carrier_restarts "$name")
    if [ "${rc:-0}" -gt "$max_restarts" ]; then
      _say "S7 ✗ 已重启 $rc 次，判定起不来"; stage7_root_cause "$name"; return 1
    fi
    if ! carrier_alive "$name"; then
      # ★ 退出码 0 不等于成功：sglang 内部健康检查超时后会干净退出
      _say "S7 ✗ 实例已退出（退出码 $(carrier_exitcode "$name")）。注意：退出码 0 也是失败，"
      _say "     因为就绪判据从未满足"; stage7_root_cause "$name"; return 1
    fi
    sleep 15
  done
  _say "S7 ✗ 超时未就绪"; stage7_root_cause "$name"; return 1
}
stage7_root_cause(){
  _say "  根因（已滤掉 HIP 异步噪声）："
  carrier_logs "$1" | grep -iE 'error|assert|Traceback|RuntimeError|Timeout|not compatible|ValueError' \
    | grep -viE 'TORCH_USE_HIP_DSA|asynchronously|Search for|Compile with|multimem|AMD_SERIALIZE' \
    | tail -6 | sed 's/^/      /'
}

# ---------------------------------------------------------------- S8 运行时自证
# 保证：声明的优化确实生效     判据：从日志读**证据**，不信任配置
# 失败：判定为「配置未生效」，数据不可用于结论
stage8_attest(){
  local name="$1"; shift
  local L; L=$(carrier_logs "$name"); local fail=0
  _say "S8 运行时自证"

  # KV 池：必须实读，不写死（通用；两种引擎两种写法）
  local pool
  pool=$(grep -oE 'GPU KV cache size: *[0-9,]+' <<<"$L" | tail -1 | grep -oE '[0-9,]+$' | tr -d ,)
  [ -n "$pool" ] || pool=$(grep -oE 'max_total_num_tokens[=: ]+[0-9]+' <<<"$L" | tail -1 | grep -oE '[0-9]+$')
  [ -n "$pool" ] && _say "  ✓ KV 池实读 $pool token" || { _say "  ✗ 读不到 KV 池"; fail=1; }
  echo "$pool" > /tmp/.attest_kvpool

  # 逐项断言（调用方传入，形如 "agent过滤:→保留4个"）
  local spec k v
  for spec in "$@"; do
    k="${spec%%:*}"; v="${spec#*:}"
    if grep -qF "$v" <<<"$L"; then _say "  ✓ $k（证据：$v）"
    else _say "  ✗ $k —— 日志里找不到「$v」，该优化很可能没生效"; fail=1; fi
  done

  # 内核回落告警（硬件耦合：aiter 缺 gfx928 配置）
  local n; n=$(grep -c "config not found" <<<"$L")
  [ "$n" = 0 ] && _say "  ✓ 无内核缺配置回落" || _say "  ⚠ $n 条内核缺配置告警（回落到 num_warps=1，会拖慢长上下文 decode）"

  [ "$fail" = 0 ] || { _say "S8 未通过：起来的不是我们要的那个配置"; return 1; }
  _say "S8 通过"; return 0
}

# ---------------------------------------------------------------- S9 收尾与资源归还
stage9_teardown(){
  local name="$1"
  _say "S9 收尾 $(carrier_name "$name")"
  local base; base=$(kfd_others "$name")
  carrier_unmanage "$name"; carrier_stop "$name" 60; carrier_remove "$name"
  stage9_wait_resources "${2:-300}" "$base"
}
# 资源归还判据属于**硬件/驱动维度**，与载体无关。
# ★ 2026-08-27 修：原来等「全机 KFD 上下文归零」是错的 —— 同机可能有别的
#   合法实例（另一副本、router）。那会让 stage5_cleanup 每次白等满超时后判失败，
#   投机解码旋钮扫描四个变体挂了三个，全是这个原因。
#   正确判据：**本实例的上下文消失**，而不是全机归零。
#   基线由调用方传入（清理前的计数），归还即「回落到基线」。
stage9_wait_resources(){
  local wait_s="${1:-300}" baseline="${2:-0}" t0=$SECONDS
  local now
  while [ $((SECONDS-t0)) -lt "$wait_s" ]; do
    now=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | wc -l)
    if [ "$now" -le "$baseline" ]; then
      [ "$baseline" -gt 0 ] && _say "  ✓ 本实例资源已归还（KFD $now，同机他者基线 $baseline）" \
                            || _say "  ✓ 显存已归还"
      return 0
    fi
    sleep 5
  done
  _say "  ⚠ 等了 $((wait_s/60)) 分钟未回落到基线 $baseline（当前 $now）"; return 1
}
# 数一下「不属于本实例」的 KFD 上下文，作为归还基线
kfd_others(){
  local name="$1" mine total
  total=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | wc -l)
  mine=$(carrier_alive "$name" 2>/dev/null && carrier_pids "$name" 2>/dev/null | wc -l || echo 0)
  echo $(( total > mine ? total - mine : 0 ))
}
