#!/usr/bin/env bash
set -Eeuo pipefail

# Extends the decisive Stage 2 comparison to nine independent training seeds,
# then evaluates every checkpoint across three demand files and three
# non-overlapping target-flow time slices. Training is serial 3-GPU DDP;
# independent frozen-checkpoint evaluations are scheduled one per GPU.

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/hangzhou_ducpd_robustness_20260820}"
REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
TRAIN_SCRIPT="$REPO_ROOT/scripts/launch_hangzhou_ducpd_robustness_arm.sh"
EVAL_SCRIPT="$REPO_ROOT/scripts/evaluate_hangzhou_ducpd_robustness_checkpoint.sh"
SLICE_SCRIPT="$REPO_ROOT/scripts/build_hangzhou_real_time_slices.py"
AGGREGATE_SCRIPT="$REPO_ROOT/scripts/aggregate_hangzhou_ducpd_robustness.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
TASKS_PATH="$RUN_ROOT/evaluation_tasks.tsv"

ARMS=(hard_bc wait2_mixed_rank)
ORIGINAL_SEEDS=(2024 42 123)
NEW_SEEDS=(7 11 99 314 2718 9001)
ALL_SEEDS=(2024 42 123 7 11 99 314 2718 9001)
SCENARIOS=(primary_real source_5816 synthetic_high time_early time_mid time_late)
LAST_NOTIFICATION_EPOCH=0

mkdir -p "$RUN_ROOT"
exec >> "$CAMPAIGN_LOG" 2>&1

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 20s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

status_of() {
    local path="$1"
    [[ -f "$path" ]] || return 0
    awk -F= '$1=="status" {print $2; exit}' "$path"
}

train_success_count() {
    local count=0
    local arm seed
    for arm in "${ARMS[@]}"; do
        for seed in "${NEW_SEEDS[@]}"; do
            [[ "$(status_of "$RUN_ROOT/arms/$arm/seed_$seed/status.txt")" == success ]] && count=$((count + 1))
        done
    done
    printf '%s' "$count"
}

evaluation_success_count() {
    local count=0
    local status_path
    for status_path in "$RUN_ROOT"/evaluations/*/*/seed_*/status.txt; do
        [[ -f "$status_path" ]] || continue
        [[ "$(status_of "$status_path")" == success ]] && count=$((count + 1))
    done
    printf '%s' "$count"
}

maybe_progress_notification() {
    local stage="$1"
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "Hangzhou DUCPD robustness progress: stage=${stage}; new training=$(train_success_count)/12; evaluations=$(evaluation_success_count)/108. Three-GPU DDP is serialized; frozen-checkpoint evaluations use one worker per GPU."
}

record_startup_snapshot() {
    local arm_root="$1"
    local child_pid="$2"
    local label="$3"
    local snapshot="$arm_root/logs/startup_monitor.log"
    [[ -d "$arm_root/logs" ]] || snapshot="$CAMPAIGN_LOG"
    {
        printf '\n[%s] %s child_pid=%s\n' "$(timestamp)" "$label" "$child_pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v pid="$child_pid" 'NR==1 || $1==pid || $2==pid || $3==pid' | head -n 100 || true
        [[ -f "$arm_root/logs/pipeline.log" ]] && timeout --foreground 10s tail -n 100 "$arm_root/logs/pipeline.log" || true
        [[ -f "$arm_root/logs/pipeline.log" ]] && timeout --foreground 10s rg -n 'Traceback|AssertionError|RuntimeError|NaN|\bInf\b' "$arm_root/logs/pipeline.log" | tail -n 40 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$snapshot" 2>&1
}

run_new_training_arm() {
    local arm="$1"
    local seed="$2"
    local arm_root="$RUN_ROOT/arms/$arm/seed_$seed"
    local child_pid minute anomaly used_sum

    if [[ -e "$arm_root" ]]; then
        if [[ "$(status_of "$arm_root/status.txt")" == success && -s "$arm_root/stage2/best.pt" ]]; then
            printf '[%s] resumed completed train %s seed=%s\n' "$(timestamp)" "$arm" "$seed"
            return 0
        fi
        printf '[%s] refusing to overwrite incomplete train directory: %s\n' "$(timestamp)" "$arm_root" >&2
        return 1
    fi

    printf '[%s] launch 3-GPU DDP train arm=%s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    setsid bash "$TRAIN_SCRIPT" "$RUN_ROOT" "$arm" "$seed" &
    child_pid=$!
    record_startup_snapshot "$arm_root" "$child_pid" startup_t0
    for minute in 1 2 3 4 5; do
        sleep 60
        record_startup_snapshot "$arm_root" "$child_pid" "startup_t${minute}m"
        maybe_progress_notification "training:${arm}:seed_${seed}"
        if ! kill -0 "$child_pid" 2>/dev/null; then
            wait "$child_pid" || true
            [[ "$(status_of "$arm_root/status.txt")" == success && -s "$arm_root/stage2/best.pt" ]] && return 0
            printf '[%s] train exited during startup: %s seed=%s\n' "$(timestamp)" "$arm" "$seed" >&2
            return 1
        fi
        anomaly="$(timeout --foreground 10s rg -n 'Traceback|AssertionError|RuntimeError|NaN|\bInf\b' "$arm_root/logs/pipeline.log" | tail -n 1 || true)"
        if [[ -n "$anomaly" ]]; then
            printf '[%s] startup anomaly for %s seed=%s: %s\n' "$(timestamp)" "$arm" "$seed" "$anomaly" >&2
            return 1
        fi
        if [[ "$minute" -eq 1 ]]; then
            used_sum="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum+0}')"
            if [[ "$used_sum" -lt 1500 ]]; then
                printf '[%s] no meaningful GPU allocation after one minute for %s seed=%s\n' "$(timestamp)" "$arm" "$seed" >&2
                return 1
            fi
        fi
    done
    printf '[%s] five-minute startup healthy: %s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep 60
        maybe_progress_notification "training:${arm}:seed_${seed}"
    done
    wait "$child_pid"
    [[ "$(status_of "$arm_root/status.txt")" == success && -s "$arm_root/stage2/best.pt" ]]
}

write_manifest() {
    "$PYTHON_BIN" - "$RUN_ROOT/campaign_manifest.json" "$REPO_ROOT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
repo = Path(sys.argv[2])
paths = {
    "direct_urgency_train_data": repo / "records/pipeline_runs/hangzhou_real_direct_urgency_20260820/data/train_5816_direct_phase_urgency_v1.jsonl",
    "stage1_checkpoint": repo / "records/pipeline_runs/hangzhou_pt32_hd256_20260813/stage1/best.pt",
    "time_slice_manifest": repo / "data/Hangzhou/4_4/hangzhou_real_time_slices_manifest.json",
}
def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
payload = {
    "schema": "hangzhou_ducpd_robustness_campaign_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "new_training_seeds": [7, 11, 99, 314, 2718, 9001],
    "all_training_seeds": [2024, 42, 123, 7, 11, 99, 314, 2718, 9001],
    "training_arms": ["hard_bc", "wait2_mixed_rank"],
    "scenarios": ["primary_real", "source_5816", "synthetic_high", "time_early", "time_mid", "time_late"],
    "counterfactual_rollouts": False,
    "labels": "existing rule-derived labels; direct observed-state urgency only",
    "inputs": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
}
manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

build_eval_tasks() {
    local task_index=0 scenario arm seed
    : > "$TASKS_PATH"
    for scenario in "${SCENARIOS[@]}"; do
        for arm in "${ARMS[@]}"; do
            for seed in "${ALL_SEEDS[@]}"; do
                printf '%s\t%s\t%s\t%s\n' "$task_index" "$scenario" "$arm" "$seed" >> "$TASKS_PATH"
                task_index=$((task_index + 1))
            done
        done
    done
}

eval_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    local task_index scenario arm seed
    while IFS=$'\t' read -r task_index scenario arm seed; do
        (( task_index % 3 == worker_index )) || continue
        bash "$EVAL_SCRIPT" "$RUN_ROOT" "$arm" "$seed" "$scenario" "$gpu_id"
    done < "$TASKS_PATH"
}

run_parallel_evaluations() {
    local pids=() worker pid failed=0
    for worker in 0 1 2; do
        eval_worker "$worker" "$worker" &
        pids+=("$!")
    done
    while :; do
        local active=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && active=1
        done
        (( active == 1 )) || break
        sleep 60
        maybe_progress_notification evaluation
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 ))
}

on_campaign_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\n' "$(timestamp)" > "$CAMPAIGN_STATUS"
        notify "Hangzhou DUCPD robustness campaign completed: 12 additional 3-GPU training runs and 108 multi-demand/time-slice evaluations. Aggregate: $RUN_ROOT/AGGREGATE.md"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\n' "$(timestamp)" > "$CAMPAIGN_STATUS"
        notify "Hangzhou DUCPD robustness campaign stopped with exit code $code. Inspect $CAMPAIGN_LOG and per-run status files."
    fi
    exit "$code"
}
trap on_campaign_exit EXIT

printf 'started_at=%s\nstatus=running\n' "$(timestamp)" > "$CAMPAIGN_STATUS"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
printf '[%s] building non-overlapping Hangzhou_real time-slice traffic inputs\n' "$(timestamp)"
"$PYTHON_BIN" "$SLICE_SCRIPT" --data-dir "$REPO_ROOT/data/Hangzhou/4_4"
write_manifest
notify "Hangzhou DUCPD robustness campaign launched: 12 additional 3-GPU DDP training arms expand hard-BC and AWT=2 to nine training seeds. After training, 108 independent frozen-checkpoint evaluations will run across three GPUs on three demand files and three target-flow time slices."

for arm in "${ARMS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        run_new_training_arm "$arm" "$seed"
    done
done

build_eval_tasks
printf '[%s] launching 108 frozen-checkpoint evaluations across GPUs 0,1,2\n' "$(timestamp)"
run_parallel_evaluations
"$PYTHON_BIN" "$AGGREGATE_SCRIPT" --run-root "$RUN_ROOT"
