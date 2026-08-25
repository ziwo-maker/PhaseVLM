#!/usr/bin/env bash
# Train Stage 1 only on jinan_real, then rerun direct-urgency Stage 2 evidence arms.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
RUN_ROOT="${1:-$REPO_ROOT/records/pipeline_runs/jinan_real_stage1_only_direct_urgency_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${JINAN_REAL_STAGE1_ONLY_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
STAGE1_SCRIPT="$REPO_ROOT/scripts/run_stage1_direct_urgency_ablation.sh"
ARM_LAUNCHER="$REPO_ROOT/scripts/launch_jinan_wait25_improvement_arm.sh"
AGGREGATOR="$REPO_ROOT/scripts/aggregate_jinan_real_stage1_only_direct_urgency.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
MODEL_PATH=/home/data/model/qwen-vl-8B
STAGE1_DATA="$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
STAGE2_DATA="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl"
STAGE2_SUMMARY="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.summary.json"
FLOW=anon_3_4_jinan_real.json
STAGE1_DIR="$RUN_ROOT/stage1_jinan_real_only"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
STATUS="$RUN_ROOT/campaign.status"
PROVENANCE="$RUN_ROOT/provenance.env"
STARTUP_MONITOR="$RUN_ROOT/startup_monitor.log"
ARMS=(hard_bc wait2_mixed_rank wait225_mixed_rank)
TRAINING_SEEDS=(2024 42 123)
LAST_NOTIFICATION_EPOCH=0

timestamp() { date -Is; }

on_error() {
    local code="$1" line="$2"
    printf '[%s] ERROR line=%s exit=%s\n' "$(timestamp)" "$line" "$code" >> "${CAMPAIGN_LOG:-/tmp/jinan_real_stage1_only_direct_urgency.error.log}" 2>/dev/null || true
}

notify() {
    [[ -f "$FEISHU_SCRIPT" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" || true
}

require_idle_gpus() {
    local gpu used
    for gpu in 0 1 2; do
        used="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
        [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 ]] || {
            echo "GPU $gpu is busy (${used:-unknown} MiB)" >&2
            return 1
        }
    done
}

completed_cells() {
    local count=0 path
    for path in "$RUN_ROOT"/arms/*/seed_*/status.txt; do
        [[ -f "$path" ]] || continue
        awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$path" && count=$((count + 1))
    done
    printf '%s' "$count"
}

latest_metric() {
    local result
    result="$(find "$RUN_ROOT/arms" -path "*/evaluation/${FLOW%.json}/seed_2024/result.json" -type f -print 2>/dev/null | sort | tail -n 1 || true)"
    [[ -s "$result" ]] || { printf 'latest_result=not_available'; return 0; }
    "$PYTHON_BIN" - "$result" <<'PY' 2>/dev/null || printf 'latest_result=unreadable'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
print(
    "latest seed2024 reward={:.3f} AQL={:.3f} AWT={:.3f} ATT={:.3f} P95={:.3f} Max={:.3f}".format(
        float(r.get("test_reward_over")),
        float(r.get("test_avg_queue_len_over")),
        float(r.get("test_avg_waiting_time_over")),
        float(r.get("test_avg_travel_time_over")),
        float(r.get("test_vehicle_wait_p95_over")),
        float(r.get("test_vehicle_wait_max_over")),
    )
)
PY
}

maybe_notify_progress() {
    local label="$1" now
    now="$(date +%s)"
    (( now - LAST_NOTIFICATION_EPOCH >= 2400 )) || return 0
    LAST_NOTIFICATION_EPOCH="$now"
    notify "Jinan real-only Stage-1 direct-urgency progress: phase=${label}; cells=$(completed_cells)/9; $(latest_metric); run_root=${RUN_ROOT}"
}

logs_have_errors() {
    timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|killed' "$RUN_ROOT" --glob '*.log' --glob '*.txt' --glob '*.status'
}

snapshot() {
    local label="$1" pid="$2" log_file="${3:-$CAMPAIGN_LOG}"
    {
        printf '\n[%s] %s pid=%s\n' "$(timestamp)" "$label" "$pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 140 || true
        [[ -f "$log_file" ]] && timeout --foreground 10s tail -n 100 "$log_file" || true
        timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|killed' "$RUN_ROOT" --glob '*.log' --glob '*.txt' --glob '*.status' | tail -n 60 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$STARTUP_MONITOR" 2>&1
}

stop_process_group() {
    local pid="$1"
    kill -- -"$pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- -"$pid" 2>/dev/null || true
}

run_monitored() {
    local label="$1" log_file="$2"
    shift 2
    printf '[%s] START %s\n' "$(timestamp)" "$label"
    require_idle_gpus
    setsid "$@" >> "$log_file" 2>&1 &
    local pid="$!" minute rc
    snapshot "${label}_startup_t0" "$pid" "$log_file"
    for minute in 1 2 3 4 5; do
        sleep 60
        snapshot "${label}_startup_t${minute}m" "$pid" "$log_file"
        if ! kill -0 "$pid" 2>/dev/null; then
            set +e
            wait "$pid"
            rc="$?"
            set -e
            if [[ "$rc" -eq 0 ]]; then
                printf '[%s] DONE %s\n' "$(timestamp)" "$label"
                return 0
            fi
            printf '[%s] FAILED %s exit=%s\n' "$(timestamp)" "$label" "$rc" >&2
            return "$rc"
        fi
        if logs_have_errors; then
            stop_process_group "$pid"
            wait "$pid" || true
            printf '[%s] FAILED %s due to startup log error\n' "$(timestamp)" "$label" >&2
            return 1
        fi
    done
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        maybe_notify_progress "$label"
        if logs_have_errors; then
            stop_process_group "$pid"
            wait "$pid" || true
            printf '[%s] FAILED %s due to log error\n' "$(timestamp)" "$label" >&2
            return 1
        fi
    done
    set +e
    wait "$pid"
    rc="$?"
    set -e
    if [[ "$rc" -eq 0 ]]; then
        printf '[%s] DONE %s\n' "$(timestamp)" "$label"
    else
        printf '[%s] FAILED %s exit=%s\n' "$(timestamp)" "$label" "$rc" >&2
    fi
    return "$rc"
}

write_provenance() {
    local path key git_head git_diff_hash
    set +e
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    file_sha() {
        local item="$1" value
        if [[ -s "$item" ]]; then
            value="$(sha256sum "$item" 2>/dev/null | awk '{print $1}')"
            printf '%s' "${value:-unavailable}"
        else
            printf 'unavailable'
        fi
    }
    : > "$PROVENANCE"
    printf 'schema=jinan_real_stage1_only_direct_urgency_campaign_v1\n' >> "$PROVENANCE"
    printf 'created_at=%s\nrun_root=%s\ngit_head=%s\ngit_diff_sha256=%s\n' "$(timestamp)" "$RUN_ROOT" "$git_head" "$git_diff_hash" >> "$PROVENANCE"
    printf 'stage1_data_mode=jinan_real_only\ntraining_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_only\n' "$FLOW" "$FLOW" >> "$PROVENANCE"
    printf 'stage1_data=%s\nstage1_data_sha256=%s\n' "$STAGE1_DATA" "$(file_sha "$STAGE1_DATA")" >> "$PROVENANCE"
    printf 'stage2_data=%s\nstage2_data_sha256=%s\nstage2_summary=%s\nstage2_summary_sha256=%s\n' "$STAGE2_DATA" "$(file_sha "$STAGE2_DATA")" "$STAGE2_SUMMARY" "$(file_sha "$STAGE2_SUMMARY")" >> "$PROVENANCE"
    printf 'stage1_checkpoint=%s\n' "$STAGE1_DIR/best.pt" >> "$PROVENANCE"
    printf 'arms=%s\ntraining_seeds=%s\neval_seeds=2024 42 123\nrun_counts=3600\n' "${ARMS[*]}" "${TRAINING_SEEDS[*]}" >> "$PROVENANCE"
    for path in "$STAGE1_SCRIPT" "$ARM_LAUNCHER" "$AGGREGATOR" "$REPO_ROOT/projection_train/train_projection_bc.py" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/projection_train/cost_ranking.py" "$REPO_ROOT/projection_train/grouped_dataset.py" "$REPO_ROOT/data_split.py"; do
        key="${path#"$REPO_ROOT"/}"
        key="${key//\//_}"
        printf 'file_%s_sha256=%s\n' "$key" "$(file_sha "$path")" >> "$PROVENANCE"
    done
    set -e
    return 0
}

on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\ncompleted_cells=%s\n' "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" "$(completed_cells)" > "$STATUS"
    notify "Jinan real-only Stage-1 direct-urgency campaign ${outcome}: exit=${code}; cells=$(completed_cells)/9; $(latest_metric); run_root=${RUN_ROOT}"
    trap - EXIT
    exit "$code"
}
trap 'exit 143' HUP INT TERM
trap 'on_error "$?" "$LINENO"' ERR
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" && -x "$STAGE1_SCRIPT" && -x "$ARM_LAUNCHER" && -s "$AGGREGATOR" ]] || { echo "Missing script prerequisite" >&2; exit 1; }
[[ -d "$MODEL_PATH" && -s "$STAGE1_DATA" && -s "$STAGE2_DATA" && -s "$STAGE2_SUMMARY" && -s "$REPO_ROOT/data/Jinan/3_4/$FLOW" ]] || { echo "Missing model/data prerequisite" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
exec >> "$CAMPAIGN_LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\nstage1_data_mode=jinan_real_only\narms=%s\ntraining_seeds=%s\n' "$(timestamp)" "$RUN_ROOT" "${ARMS[*]}" "${TRAINING_SEEDS[*]}" > "$STATUS"
printf '[%s] provenance_start\n' "$(timestamp)"
write_provenance
printf '[%s] provenance_done\n' "$(timestamp)"
if [[ "${JINAN_REAL_STAGE1_ONLY_STOP_AFTER_PROVENANCE:-0}" == "1" ]]; then
    printf '[%s] stop_after_provenance\n' "$(timestamp)"
    exit 0
fi
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "Jinan real-only Stage-1 direct-urgency campaign launched: Stage1 data=jinan_real only; arms=${ARMS[*]}; seeds=${TRAINING_SEEDS[*]}; run_root=${RUN_ROOT}"
if [[ "${JINAN_REAL_STAGE1_ONLY_STOP_BEFORE_STAGE1:-0}" == "1" ]]; then
    printf '[%s] stop_before_stage1\n' "$(timestamp)"
    exit 0
fi

run_monitored stage1_jinan_real_only "$RUN_ROOT/stage1.wrapper.log" "$STAGE1_SCRIPT" "$STAGE1_DIR" same_flow 64 8
[[ -s "$STAGE1_DIR/best.pt" ]] || { echo "Stage 1 finished without best.pt" >&2; exit 1; }
printf 'stage1_checkpoint_sha256=%s\n' "$(sha256sum "$STAGE1_DIR/best.pt" | awk '{print $1}')" >> "$PROVENANCE"

for arm in "${ARMS[@]}"; do
    for seed in "${TRAINING_SEEDS[@]}"; do
        run_monitored "stage2_${arm}_seed_${seed}" "$RUN_ROOT/${arm}_seed_${seed}.wrapper.log" \
            env WAIT25_IMPROVEMENT_GPUS=0,1,2 WAIT25_IMPROVEMENT_EVAL_GPUS=0,1,2 \
                WAIT25_IMPROVEMENT_DATA_PATH="$STAGE2_DATA" \
                WAIT25_IMPROVEMENT_ALIGN_CKPT="$STAGE1_DIR/best.pt" \
                WAIT25_IMPROVEMENT_HIDDEN_DIM=64 WAIT25_IMPROVEMENT_PREFIX_TOKENS=8 \
                "$ARM_LAUNCHER" "$RUN_ROOT" "$arm" "$seed"
    done
done

timeout --foreground 120s "$PYTHON_BIN" "$AGGREGATOR" --run-root "$RUN_ROOT"
