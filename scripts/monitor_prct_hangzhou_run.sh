#!/usr/bin/env bash
# Detached, bounded status monitor for the single-dataset PRCT campaign.
set -u -o pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 OUTPUT_DIR LAUNCHER_PID" >&2
    exit 2
fi

OUT=$1
LAUNCHER_PID=$2
REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
STATUS_LOG="$OUT/monitor_40m.log"
INTERVAL_SECONDS=2400

mkdir -p "$OUT"

snapshot() {
    local notify=$1 stamp active
    stamp=$(date -Is)
    if kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        active=yes
    else
        active=no
    fi

    {
        echo "[$stamp] launcher_pid=$LAUNCHER_PID active=$active"
        timeout --foreground 10s ps -eo pid,ppid,sid,pgid,etime,stat,cmd | \
            rg "${OUT}|run_phase7_policy_responsive.py|relabel_policy_responsive.py|relabel_simexpert.py" || true
        tail -n 120 "$OUT/launch.log" 2>/dev/null || true
        timeout --foreground 10s nvidia-smi \
            --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader || true
        rg -n -i 'traceback|exception|error|oom|killed|retry' "$OUT/launch.log" 2>/dev/null | tail -n 80 || true
        echo
    } >> "$STATUS_LOG"

    if [ "$notify" = yes ]; then
        "$PY" "$REPO/scripts/feishu_push.py" \
            "PRCT Hangzhou 多GPU实验进度：launcher=${active}；最近40分钟状态快照已写入 ${STATUS_LOG}。输出目录：${OUT}" \
            >> "$STATUS_LOG" 2>&1 || true
    fi
    [ "$active" = yes ]
}

snapshot no || exit 0
while sleep "$INTERVAL_SECONDS"; do
    snapshot yes || break
done

echo "[$(date -Is)] launcher exited; detached monitor stopped" >> "$STATUS_LOG"
