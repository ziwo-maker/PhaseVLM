#!/usr/bin/env bash
# Bounded ten-minute monitor for a detached five-track campaign.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT CAMPAIGN_PID}"
CAMPAIGN_PID="${2:?Usage: $0 RUN_ROOT CAMPAIGN_PID}"
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
FEISHU=/home/code/LLMSTSC/scripts/feishu_push.py
LOG="$RUN_ROOT/monitor_10min.log"
POLL_SECONDS=600

snapshot() {
    local label="$1"
    {
        printf '\n[%s] %s campaign_pid=%s\n' "$(date -Is)" "$label" "$CAMPAIGN_PID"
        timeout --foreground 10s ps -p "$CAMPAIGN_PID" -o pid=,ppid=,stat=,etime=,cmd= || true
        timeout --foreground 10s tail -n 120 "$RUN_ROOT/resume_campaign.log" || true
        timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|Missing' "$RUN_ROOT/resume_campaign.log" | tail -n 50 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
        [[ -f "$RUN_ROOT/resume_campaign.status" ]] && timeout --foreground 10s tail -n 20 "$RUN_ROOT/resume_campaign.status" || true
        [[ -f "$RUN_ROOT/COMPLETION_AUDIT.json" ]] && timeout --foreground 10s sed -n '1,80p' "$RUN_ROOT/COMPLETION_AUDIT.json" || true
    } >> "$LOG" 2>&1
}

while kill -0 "$CAMPAIGN_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    if kill -0 "$CAMPAIGN_PID" 2>/dev/null; then
        snapshot ten_minute_running
        [[ -f "$FEISHU" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU" "Jinan five-track recovery cohort still running; 10-minute status captured at $RUN_ROOT/monitor_10min.log." || true
    fi
done
snapshot ten_minute_terminal
[[ -f "$FEISHU" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU" "Jinan five-track recovery cohort reached terminal state; inspect $RUN_ROOT/resume_campaign.status and COMPLETION_AUDIT.json." || true
