#!/usr/bin/env bash
# Run only the missing cells for a Stage-1 x Stage-2 capacity factorial.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ROOT="${1:?Usage: $0 RUN_ROOT}"
RUN_ROOT="$(cd "$(dirname "$RUN_ROOT")" && pwd)/$(basename "$RUN_ROOT")"
SOURCE_ROOT="${CAPACITY_FACTORIAL_SOURCE_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_direct_urgency_five_tracks_20260822_recovered_cohort_r3}"
LAUNCH_ARM="$REPO_ROOT/scripts/launch_jinan_wait25_improvement_arm.sh"
AGGREGATE="$REPO_ROOT/scripts/aggregate_jinan_capacity_stage1_factorial.py"
BASE_DATA="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl"
BASE_STAGE1="$SOURCE_ROOT/baseline_threeflow/stage1/best.pt"
CAP_STAGE1="$SOURCE_ROOT/capacity_pt16_hd128/stage1/best.pt"
SEEDS=(314 9001 2718)
LOG="$RUN_ROOT/capacity_factorial_campaign.log"
STATUS="$RUN_ROOT/capacity_factorial_campaign.status"
FEISHU="$REPO_ROOT/scripts/feishu_push.py"

timestamp() { date -Is; }
notify() {
    if [[ -f "$FEISHU" ]]; then
        set +e
        timeout --kill-after=5s 15s "$PYTHON_BIN" "$FEISHU" "$1"
        set -e
    fi
    return 0
}

require_idle() {
    local gpu used
    for gpu in 0 1 2; do
        used="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
        [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 ]] || {
            echo "GPU $gpu is busy ($used MiB)" >&2
            return 1
        }
    done
}

arm_ready() {
    local root="$1" seed result
    [[ -s "$root/stage2/best.pt" && -s "$root/status.txt" ]] || return 1
    if ! rg -q '^stage2_status=success$' "$root/status.txt"; then
        [[ -s "$root/online.status" ]] && rg -q '^status=success$' "$root/online.status" || return 1
    fi
    rg -q '^online_status=success$' "$root/status.txt" || {
        [[ -s "$root/online.status" ]] && rg -q '^status=success$' "$root/online.status" || return 1
    }
    for seed in 2024 42 123; do
        result="$root/evaluation/anon_3_4_jinan_real/seed_${seed}/result.json"
        [[ -s "$result" ]] || return 1
    done
}

run_arm() {
    local track="$1" align="$2" hd="$3" pt="$4" seed="$5" root pid code
    root="$RUN_ROOT/$track/arms/wait225_mixed_rank/seed_$seed"
    if arm_ready "$root"; then
        printf '[%s] SKIP validated track=%s seed=%s\n' "$(timestamp)" "$track" "$seed"
        return 0
    fi
    [[ ! -e "$root" ]] || {
        echo "Existing incomplete run cannot be safely resumed: $root" >&2
        return 1
    }
    require_idle
    printf '[%s] START track=%s arm=wait225_mixed_rank seed=%s stage2=pt%s_hd%s align=%s\n' \
        "$(timestamp)" "$track" "$seed" "$pt" "$hd" "$align"
    WAIT25_IMPROVEMENT_GPUS=0,1,2 WAIT25_IMPROVEMENT_EVAL_GPUS=0,1,2 \
    WAIT25_IMPROVEMENT_ONLINE_STARTUP_MONITOR=initial \
    WAIT25_IMPROVEMENT_DATA_PATH="$BASE_DATA" \
    WAIT25_IMPROVEMENT_ALIGN_CKPT="$align" \
    WAIT25_IMPROVEMENT_HIDDEN_DIM="$hd" \
    WAIT25_IMPROVEMENT_PREFIX_TOKENS="$pt" \
    WAIT25_IMPROVEMENT_EXTRA_TRAIN_ARGS="--allow_mismatched_align_projectors" \
        bash "$LAUNCH_ARM" "$RUN_ROOT/$track" "wait225_mixed_rank" "$seed" &
    pid=$!
    set +e
    wait "$pid"
    code=$?
    set -e
    if [[ "$code" -ne 0 ]]; then
        printf '[%s] FAILED track=%s arm=wait225_mixed_rank seed=%s exit_code=%s\n' "$(timestamp)" "$track" "$seed" "$code" >&2
        return "$code"
    fi
    printf '[%s] DONE track=%s arm=wait225_mixed_rank seed=%s\n' "$(timestamp)" "$track" "$seed"
}

mkdir -p "$RUN_ROOT"
exec >> "$LOG" 2>&1

on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\nsource_root=%s\n' \
        "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" "$SOURCE_ROOT" > "$STATUS"
    notify "Jinan Stage1 x Stage2 capacity factorial ${outcome} (exit=${code}): $RUN_ROOT"
}
trap on_exit EXIT

printf 'started_at=%s\nstatus=running\nrun_root=%s\nsource_root=%s\n' \
    "$(timestamp)" "$RUN_ROOT" "$SOURCE_ROOT" > "$STATUS"
notify "Launching Jinan Stage1 x Stage2 capacity factorial missing cells with 3-GPU execution: $RUN_ROOT"

[[ -s "$BASE_DATA" && -s "$BASE_STAGE1" && -s "$CAP_STAGE1" && -x "$LAUNCH_ARM" ]] || {
    echo "Missing factorial prerequisite" >&2
    exit 1
}

printf 'schema=jinan_capacity_stage1_factorial_run_v1\nsource_root=%s\nbase_data=%s\nbase_data_sha256=%s\nbase_stage1=%s\nbase_stage1_sha256=%s\ncapacity_stage1=%s\ncapacity_stage1_sha256=%s\ntraining_flow=anon_3_4_jinan_real.json\narm=wait225_mixed_rank\nseeds=%s\ncross_arch_transfer=overlap_projector_copy\n' \
    "$SOURCE_ROOT" \
    "$BASE_DATA" "$(sha256sum "$BASE_DATA" | awk '{print $1}')" \
    "$BASE_STAGE1" "$(sha256sum "$BASE_STAGE1" | awk '{print $1}')" \
    "$CAP_STAGE1" "$(sha256sum "$CAP_STAGE1" | awk '{print $1}')" \
    "${SEEDS[*]}" > "$RUN_ROOT/provenance.env"

for seed in "${SEEDS[@]}"; do
    run_arm stage1_pt8_stage2_pt16 "$BASE_STAGE1" 128 16 "$seed"
done
for seed in "${SEEDS[@]}"; do
    run_arm stage1_pt16_stage2_pt8 "$CAP_STAGE1" 64 8 "$seed"
done

printf '[%s] START aggregate_results\n' "$(timestamp)"
"$PYTHON_BIN" "$AGGREGATE" --run-root "$RUN_ROOT" --source-root "$SOURCE_ROOT"
printf '[%s] DONE aggregate_results\n' "$(timestamp)"
