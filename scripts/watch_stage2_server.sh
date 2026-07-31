#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-/tmp/stage2_bc_noleak_server_watch_15m.log}"
INTERVAL_SECONDS="${2:-900}"

mkdir -p "$(dirname "$LOG_FILE")"

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    sshpass -p '1007303561wwc' ssh -p 30022 -o StrictHostKeyChecking=no \
      ziwowang@root@ssh-4e7c1df2c322734067c133835b83d17f.bpkdhfkhdpti@direct.virtaicloud.com \
      "ps -p 346,411 -o pid,ppid,%cpu,%mem,etime,cmd; \
       echo '--- GPU ---'; \
       nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader; \
       echo '--- LOG ---'; \
       tail -n 20 /gemini/code/LLMSTSC/records/pipeline_runs/exp_jinan2000/stage2_bc_noleak/nohup.log"
    echo
  } >> "$LOG_FILE" 2>&1
  sleep "$INTERVAL_SECONDS"
done
