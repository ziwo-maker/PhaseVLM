#!/usr/bin/env bash
# Full Jinan direct-urgency campaign: data versioning, Stage-1 baseline,
# 12 Stage-2 arms, and complete Jinan online generalization for every model.
set -Eeuo pipefail

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/jinan_real_direct_urgency_20260821}"
REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${DIRECT_URGENCY_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /home/sysu/anaconda3/envs/hnr5090/bin/python ]]; then
        PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
if [[ -n "${DIRECT_URGENCY_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$DIRECT_URGENCY_LD_PRELOAD"
fi
BUILDER="$REPO_ROOT/scripts/build_direct_urgency_stage2_dataset.py"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_direct_urgency_online_suite.sh"
ARM_SCRIPT="$REPO_ROOT/scripts/launch_jinan_direct_urgency_arm.sh"
AGGREGATE_SCRIPT="$REPO_ROOT/scripts/aggregate_jinan_direct_urgency.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
SOURCE_DATA="$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DIRECT_DATA="$RUN_ROOT/data/train_1440_direct_phase_urgency_v1.jsonl"
DIRECT_SUMMARY="$RUN_ROOT/data/train_1440_direct_phase_urgency_v1.summary.json"
STAGE1_CKPT="$REPO_ROOT/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
PROVENANCE_PATH="$RUN_ROOT/provenance.env"
ARMS=(hard_bc equal_mixed_rank wait2_mixed_rank wait25_mixed_rank)
TRAINING_SEEDS=(2024 42 123)
LAST_NOTIFICATION_EPOCH=0
AVAILABLE_GPUS=()

discover_idle_gpus() {
    local gpu used
    AVAILABLE_GPUS=()
    while IFS=, read -r gpu used; do
        gpu="${gpu//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
        if (( used < 500 )); then
            AVAILABLE_GPUS+=("$gpu")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    ((${#AVAILABLE_GPUS[@]} > 0)) || {
        echo "No idle GPU is visible to this runtime" >&2
        return 1
    }
    DIRECT_URGENCY_GPUS="$(IFS=,; printf '%s' "${AVAILABLE_GPUS[*]}")"
    export DIRECT_URGENCY_GPUS
    if [[ -z "${DIRECT_URGENCY_EVAL_GPU:-}" ]]; then
        DIRECT_URGENCY_EVAL_GPU="${AVAILABLE_GPUS[0]}"
    fi
    [[ " ${AVAILABLE_GPUS[*]} " =~ " ${DIRECT_URGENCY_EVAL_GPU} " ]] || {
        echo "DIRECT_URGENCY_EVAL_GPU=${DIRECT_URGENCY_EVAL_GPU} is not idle" >&2
        return 1
    }
    export DIRECT_URGENCY_EVAL_GPU
}

mkdir -p "$RUN_ROOT"
exec >> "$CAMPAIGN_LOG" 2>&1

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

completed_count() {
    local count=0 status
    for status in "$RUN_ROOT"/arms/*/seed_*/status.txt; do
        [[ -f "$status" ]] || continue
        if awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$status"; then
            count=$((count + 1))
        fi
    done
    printf '%s' "$count"
}

maybe_progress_notification() {
    local phase="$1"
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "jinan_real direct-urgency campaign progress: $(completed_count)/12 Stage-2 arms complete; current=${phase}."
}

preflight() {
    [[ -s "$SOURCE_DATA" ]] || { echo "Missing source data: $SOURCE_DATA" >&2; return 1; }
    [[ -s "$STAGE1_CKPT" ]] || { echo "Missing Jinan Stage 1 checkpoint: $STAGE1_CKPT" >&2; return 1; }
    [[ -d /home/data/model/qwen-vl-8B ]] || { echo "Missing model: /home/data/model/qwen-vl-8B" >&2; return 1; }
    for path in \
        "$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json" \
        "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real.json" \
        "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real_2000.json" \
        "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real_2500.json"; do
        [[ -s "$path" ]] || { echo "Missing online prerequisite: $path" >&2; return 1; }
    done
    discover_idle_gpus
    printf 'Detected idle GPUs: %s; fixed online evaluation GPU: %s\n' \
        "$DIRECT_URGENCY_GPUS" "$DIRECT_URGENCY_EVAL_GPU"
}

build_data() {
    if [[ -s "$DIRECT_DATA" && -s "$DIRECT_SUMMARY" ]]; then
        printf '[%s] reuse versioned direct-urgency data: %s\n' "$(timestamp)" "$DIRECT_DATA"
        return 0
    fi
    mkdir -p "$(dirname "$DIRECT_DATA")"
    printf '[%s] building direct-urgency dataset\n' "$(timestamp)"
    timeout --foreground 600s "$PYTHON_BIN" "$BUILDER" --input "$SOURCE_DATA" --output "$DIRECT_DATA" --summary "$DIRECT_SUMMARY"
    "$PYTHON_BIN" - "$DIRECT_SUMMARY" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["rows"] == 22860, summary
assert summary["terminal_labels"] == 1440, summary
assert summary["full_candidate_metric_rows"] == 1440, summary
assert summary["label_semantics"] == "existing_rule_labels; observed_phase_queue_wait_no_rollout", summary
print("direct dataset verified", json.dumps(summary, sort_keys=True))
PY
}

write_provenance() {
    local path relative git_head git_diff_hash
    git_head="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
    git_diff_hash="$({ git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    {
        printf 'created_at=%s\n' "$(timestamp)"
        printf 'git_head=%s\n' "$git_head"
        printf 'git_diff_sha256=%s\n' "$git_diff_hash"
        printf 'source_data=%s\nsource_data_sha256=%s\n' "$SOURCE_DATA" "$(sha256sum "$SOURCE_DATA" | awk '{print $1}')"
        printf 'direct_data=%s\ndirect_data_sha256=%s\n' "$DIRECT_DATA" "$(sha256sum "$DIRECT_DATA" | awk '{print $1}')"
        printf 'direct_summary=%s\ndirect_summary_sha256=%s\n' "$DIRECT_SUMMARY" "$(sha256sum "$DIRECT_SUMMARY" | awk '{print $1}')"
        printf 'stage1_checkpoint=%s\nstage1_checkpoint_sha256=%s\n' "$STAGE1_CKPT" "$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')"
        printf 'online_eval_gpu=%s\nonline_eval_seed_execution=serial_single_gpu\n' "$DIRECT_URGENCY_EVAL_GPU"
        for path in \
            "$BUILDER" \
            "$ARM_SCRIPT" \
            "$ONLINE_SUITE" \
            "$AGGREGATE_SCRIPT" \
            "$REPO_ROOT/projection_train/train_projection_bc.py" \
            "$REPO_ROOT/projection_train/run_projection_online.py" \
            "$REPO_ROOT/projection_train/cost_ranking.py" \
            "$REPO_ROOT/projection_train/grouped_dataset.py" \
            "$REPO_ROOT/data_split.py"; do
            relative="${path#"$REPO_ROOT"/}"
            relative="${relative//\//_}"
            printf 'file_%s_sha256=%s\n' "$relative" "$(sha256sum "$path" | awk '{print $1}')"
        done
    } > "$PROVENANCE_PATH"
    printf '[%s] wrote provenance manifest: %s\n' "$(timestamp)" "$PROVENANCE_PATH"
}

run_baseline() {
    local baseline_root="$RUN_ROOT/baseline"
    if [[ -f "$baseline_root/online.status" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$baseline_root/online.status"; then
        printf '[%s] baseline online suite already complete\n' "$(timestamp)"
        return 0
    fi
    mkdir -p "$baseline_root"
    printf '[%s] Stage 1 baseline online suite\n' "$(timestamp)"
    "$ONLINE_SUITE" "$RUN_ROOT" "$baseline_root" "$STAGE1_CKPT" jinan_direct_urgency_stage1_baseline
}

run_arm() {
    local arm="$1" seed="$2"
    local arm_root="$RUN_ROOT/arms/$arm/seed_$seed"
    if [[ -f "$arm_root/status.txt" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$arm_root/status.txt"; then
        printf '[%s] arm already complete: %s seed=%s\n' "$(timestamp)" "$arm" "$seed"
        return 0
    fi
    [[ ! -e "$arm_root" ]] || { echo "Refusing to overwrite incomplete arm: $arm_root" >&2; return 1; }
    printf '[%s] launching Stage 2 arm=%s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    setsid bash "$ARM_SCRIPT" "$RUN_ROOT" "$arm" "$seed" &
    local child_pid=$!
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep 60
        maybe_progress_notification "${arm}/seed_${seed}"
    done
    wait "$child_pid"
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\nrun_root=%s\n' "$(timestamp)" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
        notify "jinan_real direct-urgency campaign completed: 12 Stage-2 models and 108 online matrix results. Aggregate: $RUN_ROOT/AGGREGATE.md"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\nrun_root=%s\n' "$(timestamp)" "$code" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
        notify "jinan_real direct-urgency campaign stopped with exit_code=${code}; inspect $CAMPAIGN_LOG"
    fi
    trap - EXIT
    exit "$code"
}
trap 'exit 143' HUP INT TERM
trap on_exit EXIT

printf 'started_at=%s\nstatus=running\nrun_root=%s\n' "$(timestamp)" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
preflight
build_data
write_provenance
notify "jinan_real direct-urgency campaign launched: four Stage-2 arms x seeds 2024/42/123, Jinan-real training data, all Jinan real/2000/2500 online demand tests, training_gpus=${DIRECT_URGENCY_GPUS}, online_gpu=${DIRECT_URGENCY_EVAL_GPU}."
printf '[%s] Stage 1 is an alignment initializer only; hard_bc is the matched online Stage 2 baseline.\n' "$(timestamp)"
for arm in "${ARMS[@]}"; do
    for seed in "${TRAINING_SEEDS[@]}"; do
        run_arm "$arm" "$seed"
    done
done
timeout --foreground 120s "$PYTHON_BIN" "$AGGREGATE_SCRIPT" --run-root "$RUN_ROOT"
