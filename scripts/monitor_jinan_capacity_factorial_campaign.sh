#!/usr/bin/env bash
# Bounded ten-minute monitor for the capacity factorial campaign.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT CAMPAIGN_PID}"
CAMPAIGN_PID="${2:?Usage: $0 RUN_ROOT CAMPAIGN_PID}"
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
FEISHU=/home/code/LLMSTSC/scripts/feishu_push.py
LOG="$RUN_ROOT/monitor_10min.log"
POLL_SECONDS=600
NOTIFY_EVERY_POLLS=4
MAX_POLLS=6

notify() {
    if [[ -f "$FEISHU" ]]; then
        set +e
        timeout --kill-after=5s 15s "$PYTHON_BIN" "$FEISHU" "$1"
        set -e
    fi
    return 0
}

snapshot() {
    local label="$1"
    {
        printf '\n[%s] %s campaign_pid=%s\n' "$(date -Is)" "$label" "$CAMPAIGN_PID"
        timeout --foreground 10s ps -p "$CAMPAIGN_PID" -o pid=,ppid=,stat=,etime=,cmd= || true
        [[ -f "$RUN_ROOT/capacity_factorial_campaign.log" ]] && timeout --foreground 10s tail -n 120 "$RUN_ROOT/capacity_factorial_campaign.log" || true
        [[ -f "$RUN_ROOT/capacity_factorial_campaign.log" ]] && timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|Missing' "$RUN_ROOT/capacity_factorial_campaign.log" | tail -n 50 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
        [[ -f "$RUN_ROOT/capacity_factorial_campaign.status" ]] && timeout --foreground 10s tail -n 20 "$RUN_ROOT/capacity_factorial_campaign.status" || true
        [[ -f "$RUN_ROOT/COMPLETION_AUDIT.json" ]] && timeout --foreground 10s sed -n '1,80p' "$RUN_ROOT/COMPLETION_AUDIT.json" || true
    } >> "$LOG" 2>&1
}

poll_count=0
while kill -0 "$CAMPAIGN_PID" 2>/dev/null && (( poll_count < MAX_POLLS )); do
    sleep "$POLL_SECONDS"
    if kill -0 "$CAMPAIGN_PID" 2>/dev/null; then
        poll_count=$((poll_count + 1))
        snapshot ten_minute_running
        if (( poll_count % NOTIFY_EVERY_POLLS == 0 )); then
            notify "Jinan capacity factorial still running; 10-minute status captured at $RUN_ROOT/monitor_10min.log."
        fi
    fi
done
if kill -0 "$CAMPAIGN_PID" 2>/dev/null; then
    snapshot one_hour_monitor_stop
    notify "Jinan capacity factorial one-hour monitor window ended; campaign still running. Inspect $RUN_ROOT/monitor_10min.log and $RUN_ROOT/capacity_factorial_campaign.status."
else
    snapshot ten_minute_terminal
    notify "Jinan capacity factorial reached terminal state; inspect $RUN_ROOT/capacity_factorial_campaign.status and COMPLETION_AUDIT.json."
fi
