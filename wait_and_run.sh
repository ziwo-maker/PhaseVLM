#!/usr/bin/env bash
set -euo pipefail

# ====== 你要运行的命令（对应你的 launch 配置）======
CUDA_VISIBLE_DEVICES_STR="0,1,2,3"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_STR}"

PYTHON_BIN="python"  # 如果你是 conda 环境：改成 /path/to/conda/env/bin/python
SCRIPT="run_open_LLM_with_vllm.py"
ARGS=(--dataset jinan --traffic_file anon_3_4_jinan_real.json)

# ====== 空闲判定阈值（可按需调整）======
# GPU 利用率 < GPU_UTIL_MAX (%) 且 显存使用 < MEM_USED_MAX (MiB) 且 没有 compute 进程
GPU_UTIL_MAX=5
MEM_USED_MAX=800

# ====== 轮询间隔（秒）======
INTERVAL=20

# 将 "0,1,2,3" -> (0 1 2 3)
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES_STR}"

echo "[INFO] Watching GPUs: ${GPUS[*]}"
echo "[INFO] Thresholds: util<${GPU_UTIL_MAX}% mem<${MEM_USED_MAX}MiB and no compute processes"
echo "[INFO] When idle => run: ${PYTHON_BIN} ${SCRIPT} ${ARGS[*]}"
echo

is_all_idle() {
  # 1) 检查每张卡的 util/mem
  local ids_csv
  ids_csv="$(IFS=','; echo "${GPUS[*]}")"

  # 输出格式：index, util, mem_used
  local lines
  lines="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -i "${ids_csv}")"

  # 逐行判断阈值
  while IFS=',' read -r idx util mem; do
    idx="$(echo "$idx" | xargs)"
    util="$(echo "$util" | xargs)"
    mem="$(echo "$mem" | xargs)"

    if [[ "$util" -ge "$GPU_UTIL_MAX" ]]; then
      return 1
    fi
    if [[ "$mem" -ge "$MEM_USED_MAX" ]]; then
      return 1
    fi
  done <<< "$lines"

  # 2) 检查 compute 进程（更严格：只要有 compute 进程就不算空闲）
  # nvidia-smi pmon: type=C 表示 compute
  local pmon
  pmon="$(nvidia-smi pmon -c 1 -s um -i "${ids_csv}" 2>/dev/null || true)"
  # 跳过表头，检查是否存在 type 为 C 且 pid 不是 -
  # 注意：不同驱动输出列略有差异，这里做宽松匹配
  if echo "$pmon" | awk 'NR>2 { if ($2 != "-" && $7 == "C") found=1 } END{ exit(found?0:1) }'; then
    return 1
  fi

  return 0
}

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  if is_all_idle; then
    echo "[$ts] ✅ All GPUs idle. Launching..."
    exec "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}"
  else
    echo "[$ts] ... GPUs busy, check again in ${INTERVAL}s"
  fi
  sleep "${INTERVAL}"
done

