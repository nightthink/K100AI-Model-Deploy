#!/bin/bash
# 生命周期合同 · 纯逻辑模块（R24–R34）
#
# 本文件只含**纯函数**：输入字符串、输出字符串或返回码，不碰 docker、不碰文件系统、
# 不读环境前提。因此它可以被测试直接 source，在无 GPU / 无 Docker 的机器上完整验证。
#
# ★ 设计约束（2026-09-22）：
#   · 不得在生产路径上提供任何「跳过驱动/设备检查」的开关。本模块之所以可测，
#     是因为它本身不依赖那些前提，而不是因为它能绕过它们。
#   · 调用方（up.sh）负责采集事实（容器表、端口、GPU 占用），本模块只做判定。
#   · 其它线不受影响：本模块只被 01/13 的 up.sh 主流程调用，不改变既有分支行为。

# ── 身份规范化（R25）──────────────────────────────────────────────

# R25.1 模型身份 = realpath(model_path)
#   软链、相对路径、结尾斜杠差异归一后视为同一模型。
#   realpath 不可用或路径不存在时回退到「去尾斜杠的原串」，不静默当成空。
ct_norm_model(){
  local p="${1:-}"
  [ -z "$p" ] && { echo ""; return 1; }
  local r
  if r=$(realpath -m "$p" 2>/dev/null) && [ -n "$r" ]; then echo "$r"
  else echo "${p%/}"; fi
}

# R25.2 GPU 集合排序去重
#   "4,5,6,7" 与 "7,6,5,4,5" → "4,5,6,7"。空输入返回空串并回非零。
ct_norm_gpus(){
  local g="${1:-}"
  [ -z "$g" ] && { echo ""; return 1; }
  echo "$g" | tr ',' '\n' | sed 's/[[:space:]]//g' | grep -E '^[0-9]+$' \
    | sort -n -u | paste -sd, -
}

# R25.4–R25.6 launch_mode 只表达执行机制：docker / systemd / process
#   不表达引擎、TP 规模、精度、投机算法或调优方案。
ct_norm_launch_mode(){
  local m="${1:-docker}"
  m=$(echo "$m" | tr 'A-Z' 'a-z' | sed 's/[[:space:]]//g')
  case "$m" in docker|systemd|process) echo "$m";; *) echo "$m"; return 1;; esac
}

# 身份四元组（R25.3），字段顺序固定：model|gpus|launch_mode|port
ct_identity(){ printf '%s|%s|%s|%s' \
  "$(ct_norm_model "${1:-}")" "$(ct_norm_gpus "${2:-}")" \
  "$(ct_norm_launch_mode "${3:-docker}")" "${4:-}"; }

# GPU 集合关系：equal / disjoint / overlap
ct_gpu_relation(){
  local a b; a=$(ct_norm_gpus "${1:-}"); b=$(ct_norm_gpus "${2:-}")
  [ "$a" = "$b" ] && { echo equal; return; }
  local g inter=0
  for g in ${a//,/ }; do case ",$b," in *",$g,"*) inter=1; break;; esac; done
  [ "$inter" = 1 ] && echo overlap || echo disjoint
}

# ── 归属证明（R26.1 / R32）────────────────────────────────────────
# 归属必须由包内稳定标签与包状态互证，**不得仅凭名称前缀**。
# 标签形如 "line=01,pkg=<version>,launch_id=<id>"。
ct_label_get(){ # labels key
  echo "${1:-}" | tr ',' '\n' | sed -n "s/^${2}=//p" | head -1
}
# 返回 0 = 可证明属于本包本线；非 0 = 无法证明（按未知占用处理）
ct_is_ours(){ # labels expect_line expect_pkg
  local lb="${1:-}" line="${2:-}" pkg="${3:-}"
  local l p i
  l=$(ct_label_get "$lb" line); p=$(ct_label_get "$lb" pkg); i=$(ct_label_get "$lb" launch_id)
  [ -n "$l" ] && [ -n "$i" ] || return 1          # 缺标签 ⇒ 无法证明
  [ "$l" = "$line" ] || return 1
  [ -n "$pkg" ] && [ -n "$p" ] && [ "$p" != "$pkg" ] && return 1
  return 0
}

# ── 冲突分类与决策（R26 / R27 / R28）──────────────────────────────
#
# ct_decide <自身四元组> <既有实例四元组> <既有状态> <既有是否可证明归属>
#   自身/既有四元组：ct_identity 的输出
#   既有状态：READY / STARTING / FAILED / STOPPED / NONE
#   可证明归属：own / unknown
#
# 输出一行：<决策> <原因码>
#   决策：REUSE / ATTACH / RECREATE / MIGRATE_PORT / FAIL
ct_decide(){
  local me="${1:-}" ex="${2:-}" exstate="${3:-NONE}" own="${4:-unknown}"

  [ -z "$ex" ] || [ "$exstate" = NONE ] && { echo "CREATE new_instance"; return 0; }

  # 未知归属一律失败关闭，且不得清理（R26 情形 5 / R32）
  [ "$own" != own ] && { echo "FAIL unknown_ownership"; return 3; }

  local m1 g1 l1 p1 m2 g2 l2 p2
  IFS='|' read -r m1 g1 l1 p1 <<<"$me"
  IFS='|' read -r m2 g2 l2 p2 <<<"$ex"

  local rel; rel=$(ct_gpu_relation "$g1" "$g2")

  # R26 情形 1：GPU 相交但不完全相同
  [ "$rel" = overlap ] && { echo "FAIL gpu_partial_overlap"; return 3; }

  if [ "$rel" = equal ]; then
    # R26 情形 2：同 GPU 异模型
    [ "$m1" != "$m2" ] && { echo "FAIL same_gpu_different_model"; return 3; }
    # R26 情形 3：同模型同 GPU 异 launch_mode
    [ "$l1" != "$l2" ] && { echo "FAIL same_gpu_different_launch_mode"; return 3; }

    if [ "$p1" = "$p2" ]; then
      # R27 四项全等
      case "$exstate" in
        READY)    echo "REUSE ready_same_identity"; return 0;;
        STARTING) echo "ATTACH starting_same_identity"; return 0;;
        FAILED)   echo "RECREATE failed_cleanup_own_residue"; return 0;;
        STOPPED)  echo "RECREATE stopped_same_identity"; return 0;;
        *)        echo "FAIL unexpected_state"; return 3;;
      esac
    else
      # R28 仅端口不同 —— 调用方须先验新端口可用，再定点停旧
      echo "MIGRATE_PORT port_changed_verify_new_first"; return 0
    fi
  fi

  # GPU 完全不相交 ⇒ 是另一个实例，与本次无关
  echo "CREATE disjoint_gpu"; return 0
}

# R26 情形 4：目标端口被「别的实例」占用
#   调用方传入该端口持有者的四元组与归属；持有者即本次目标实例时不算占用。
ct_port_conflict(){ # me_identity holder_identity holder_own
  local me="${1:-}" ho="${2:-}" own="${3:-unknown}"
  [ -z "$ho" ] && return 0
  [ "$me" = "$ho" ] && return 0
  [ "$own" != own ] && { echo "FAIL port_held_by_unknown"; return 3; }
  echo "FAIL port_busy"; return 3
}
