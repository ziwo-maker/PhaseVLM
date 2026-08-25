#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential because each Stage 2 DDP arm reserves all local GPUs.  Each newly
# launched arm receives the project-required five-minute startup observation
# window in a per-arm, bounded snapshot log.

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/hangzhou_real_direct_urgency_20260820}"
REPO_ROOT=/home/code/LLMSTSC
ARM_SCRIPT="$REPO_ROOT/scripts/launch_hangzhou_direct_urgency_arm.sh"
AGGREGATE_SCRIPT="$REPO_ROOT/scripts/aggregate_hangzhou_direct_urgency.py"
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
mkdir -p "$RUN_ROOT"
exec >> "$CAMPAIGN_LOG" 2>&1

ARMS=(hard_bc equal_mixed_rank wait2_mixed_rank wait25_mixed_rank)
SEEDS=(2024 42 123)
LAST_NOTIFICATION_EPOCH=0

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

completed_count() {
    local count=0
    local status_path
    for status_path in "$RUN_ROOT"/arms/*/seed_*/status.txt; do
        [[ -f "$status_path" ]] || continue
        if awk -F= '$1=="status" && $2=="success" {found=1} END {exit !found}' "$status_path"; then
            count=$((count + 1))
        fi
    done
    printf '%s' "$count"
}

maybe_progress_notification() {
    local current_arm="$1"
    local current_seed="$2"
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "Hangzhou_real direct-urgency Stage 2 progress: $(completed_count)/12 arms completed; current=${current_arm}/seed_${current_seed}. All training is sequential 3-GPU DDP; fixed 3600s primary evaluation follows each successful arm."
}

status_of() {
    local status_path="$1/status.txt"
    [[ -f "$status_path" ]] || return 0
    awk -F= '$1=="status" {print $2; exit}' "$status_path"
}

is_success() {
    local arm_root="$1"
    [[ "$(status_of "$arm_root")" == "success" && -s "$arm_root/evaluation/result.json" ]]
}

record_snapshot() {
    local arm_root="$1"
    local child_pid="$2"
    local label="$3"
    local snapshot="$arm_root/logs/startup_monitor.log"
    # The arm refuses to start if its directory exists before its own
    # preflight.  During the first milliseconds, retain the t0 snapshot in
    # the campaign log and switch to the per-arm file once the arm owns it.
    if [[ ! -d "$arm_root/logs" ]]; then
        snapshot="$CAMPAIGN_LOG"
    fi
    {
        printf '\n[%s] %s child_pid=%s\n' "$(timestamp)" "$label" "$child_pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | \
            awk -v pid="$child_pid" 'NR==1 || $1==pid || $2==pid || $3==pid' | head -n 80 || true
        if [[ -f "$arm_root/logs/pipeline.log" ]]; then
            timeout --foreground 10s tail -n 80 "$arm_root/logs/pipeline.log" || true
            timeout --foreground 10s rg -n 'Traceback|ERROR|Exception|NaN|Inf' "$arm_root/logs/pipeline.log" | tail -n 40 || true
        fi
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$snapshot" 2>&1
}

wait_existing() {
    local arm_root="$1"
    local elapsed=0
    while [[ "$(status_of "$arm_root")" == "running" ]]; do
        printf '[%s] waiting for pre-existing run: %s\n' "$(timestamp)" "$arm_root"
        sleep 60
        maybe_progress_notification preexisting 2024
        elapsed=$((elapsed + 60))
        if (( elapsed >= 3600 )); then
            printf '[%s] existing run exceeded one-hour wait: %s\n' "$(timestamp)" "$arm_root" >&2
            return 1
        fi
    done
}

run_arm() {
    local arm="$1"
    local seed="$2"
    local arm_root="$RUN_ROOT/arms/$arm/seed_$seed"
    local child_pid
    local minute
    local errors

    if [[ -e "$arm_root" ]]; then
        wait_existing "$arm_root"
        if is_success "$arm_root"; then
            printf '[%s] already complete: %s seed=%s\n' "$(timestamp)" "$arm" "$seed"
            return 0
        fi
        printf '[%s] pre-existing run not successful; refusing rerun: %s\n' "$(timestamp)" "$arm_root" >&2
        return 1
    fi

    printf '[%s] launching arm=%s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    setsid bash "$ARM_SCRIPT" "$RUN_ROOT" "$arm" "$seed" &
    child_pid=$!
    record_snapshot "$arm_root" "$child_pid" startup_t0
    for minute in 1 2 3 4 5; do
        sleep 60
        record_snapshot "$arm_root" "$child_pid" "startup_t${minute}m"
        if ! kill -0 "$child_pid" 2>/dev/null; then
            printf '[%s] arm exited during startup window: %s seed=%s\n' "$(timestamp)" "$arm" "$seed" >&2
            wait "$child_pid" || true
            return 1
        fi
        errors="$(timeout --foreground 10s rg -n 'Traceback|ERROR|Exception|NaN|Inf' "$arm_root/logs/pipeline.log" | tail -n 1 || true)"
        if [[ -n "$errors" ]]; then
            printf '[%s] startup anomaly for %s seed=%s: %s\n' "$(timestamp)" "$arm" "$seed" "$errors" >&2
            kill -- -"$child_pid" 2>/dev/null || true
            wait "$child_pid" || true
            return 1
        fi
    done
    printf '[%s] startup healthy, waiting for completion: %s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep 60
        maybe_progress_notification "$arm" "$seed"
    done
    wait "$child_pid"
    is_success "$arm_root"
}

printf 'started_at=%s\nstatus=running\n' "$(timestamp)" > "$CAMPAIGN_STATUS"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "Hangzhou_real direct-urgency Stage 2 campaign launched: 12 sequential arms (hard BC, equal, AWTx2, AWTx2.5; training seeds 2024/42/123), direct observed queue/wait supervision only, no counterfactual rollouts."
for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_arm "$arm" "$seed"
    done
done
"$PYTHON_BIN" "$AGGREGATE_SCRIPT" --run-root "$RUN_ROOT"
printf 'completed_at=%s\nstatus=success\n' "$(timestamp)" > "$CAMPAIGN_STATUS"
notify "Hangzhou_real direct-urgency Stage 2 campaign completed: 12/12 arms finished. Aggregate: $RUN_ROOT/AGGREGATE.md and $RUN_ROOT/aggregate.json."
