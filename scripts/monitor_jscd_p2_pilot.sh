#!/usr/bin/env bash
# Detached, bounded 40-minute status snapshots for a JSCD P2 campaign.
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

snapshot() {
    local notify=$1 active stamp
    stamp=$(date -Is)
    if kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        active=yes
    else
        active=no
    fi
    {
        echo "[$stamp] launcher_pid=$LAUNCHER_PID active=$active"
        timeout --foreground 10s ps -eo pid,ppid,sid,pgid,etime,stat,cmd | \
            rg "${OUT}|launch_jscd_p2_pilot|relabel_policy_responsive|run_phase7_crcd|train_projection_bc" || true
        tail -n 120 "$OUT/launch.log" 2>/dev/null || true
        timeout --foreground 10s nvidia-smi \
            --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader || true
        rg -n -i 'traceback|exception|error|oom|killed|retry|failed' "$OUT/launch.log" 2>/dev/null | tail -n 80 || true
        echo
    } >> "$STATUS_LOG"
    if [ "$notify" = yes ]; then
        "$PY" "$REPO/scripts/feishu_push.py" \
            "JSCD P2 多GPU实验进度：launcher=${active}；最近40分钟快照已写入 ${STATUS_LOG}。" \
            >> "$STATUS_LOG" 2>&1 || true
    fi
    [ "$active" = yes ]
}

snapshot no || exit 0
while true; do
    for _ in $(seq 1 40); do
        sleep 60
        if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
            snapshot no
            exit 0
        fi
    done
    snapshot yes || exit 0
done
