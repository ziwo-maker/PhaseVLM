#!/usr/bin/env bash
# Resume the five-track campaign without repeating validated artifacts.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ROOT="${1:?Usage: $0 RUN_ROOT}"
RUN_ROOT="$(cd "$(dirname "$RUN_ROOT")" && pwd)/$(basename "$RUN_ROOT")"
LAUNCH_ARM="$REPO_ROOT/scripts/launch_jinan_wait25_improvement_arm.sh"
STAGE1="$REPO_ROOT/scripts/run_stage1_direct_urgency_ablation.sh"
ONLINE="$REPO_ROOT/projection_train/run_projection_online.py"
TAIL_BUILDER="$REPO_ROOT/scripts/build_queue_wait_distill_dataset.py"
TAIL_AUGMENT="$REPO_ROOT/scripts/add_direct_phase_tail_labels.py"
AGGREGATE="$REPO_ROOT/scripts/aggregate_jinan_direct_urgency_five_track_campaign.py"
MODEL=/home/data/model/qwen-vl-8B
BASE_DATA="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl"
BASE_STAGE1="$RUN_ROOT/baseline_threeflow/stage1/best.pt"
SEEDS=(314 9001 2718)
LOG="$RUN_ROOT/resume_campaign.log"
STATUS="$RUN_ROOT/resume_campaign.status"
FEISHU="$REPO_ROOT/scripts/feishu_push.py"

timestamp() { date -Is; }
notify() { [[ -f "$FEISHU" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU" "$1" || true; }
require_idle() {
    local gpu used
    for gpu in 0 1 2; do
        used="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
        [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 ]] || { echo "GPU $gpu is busy ($used MiB)" >&2; return 1; }
    done
}
arm_ready() {
    local root="$1" seed result
    [[ -s "$root/stage2/best.pt" && -s "$root/status.txt" ]] || return 1
    # seed_9001's wrapper status predates the recovered online run; its
    # checkpoint and successful protocol-validated online.status are enough
    # to prove the Stage-2 training completed without repeating it.
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
    local track="$1" arm="$2" seed="$3" data="$4" align="$5" hd="$6" pt="$7"
    local root="$RUN_ROOT/$track/arms/$arm/seed_$seed" pid
    if arm_ready "$root"; then
        printf '[%s] SKIP validated track=%s arm=%s seed=%s\n' "$(timestamp)" "$track" "$arm" "$seed"
        return 0
    fi
    [[ ! -e "$root" ]] || { echo "Existing incomplete run cannot be safely resumed: $root" >&2; return 1; }
    require_idle
    printf '[%s] START track=%s arm=%s seed=%s\n' "$(timestamp)" "$track" "$arm" "$seed"
    WAIT25_IMPROVEMENT_GPUS=0,1,2 WAIT25_IMPROVEMENT_EVAL_GPUS=0,1,2 \
    WAIT25_IMPROVEMENT_DATA_PATH="$data" WAIT25_IMPROVEMENT_ALIGN_CKPT="$align" \
    WAIT25_IMPROVEMENT_HIDDEN_DIM="$hd" WAIT25_IMPROVEMENT_PREFIX_TOKENS="$pt" \
        setsid bash "$LAUNCH_ARM" "$RUN_ROOT/$track" "$arm" "$seed" &
    pid=$!
    wait "$pid"
    printf '[%s] DONE track=%s arm=%s seed=%s\n' "$(timestamp)" "$track" "$arm" "$seed"
}
run_sync() { local label="$1"; shift; printf '[%s] START %s\n' "$(timestamp)" "$label"; "$@"; printf '[%s] DONE %s\n' "$(timestamp)" "$label"; }

build_tail_data() {
    local root="$RUN_ROOT/tail_direct_labels" raw="$RUN_ROOT/tail_direct_labels/collected/state_action.json" base="$RUN_ROOT/tail_direct_labels/data/rule_tail_base.jsonl" data="$RUN_ROOT/tail_direct_labels/data/train_tail_p95_direct.jsonl" work_dir
    [[ -s "$data" ]] && { printf '[%s] SKIP validated tail data\n' "$(timestamp)"; return 0; }
    mkdir -p "$root/collected" "$root/data"; require_idle
    run_sync tail_state_collection env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$ONLINE" \
      --memo five_track_tail_collect --projection_ckpt "$RUN_ROOT/seed_expansion/arms/wait2_mixed_rank/seed_314/stage2/best.pt" \
      --llm_path "$MODEL" --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --num_rounds 1 --seed 2024 \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --inference_batch_size 64 --bf16 --deterministic --collect_states_output "$root/collected/decision_states.jsonl" --result_json "$root/collected/result.json"
    work_dir="$(jq -r '.work_dir' "$root/collected/result.json")"; [[ -s "$work_dir/state_action.json" ]] || return 1; cp "$work_dir/state_action.json" "$raw"
    run_sync tail_rule_anchor "$PYTHON_BIN" "$TAIL_BUILDER" --state_action "$raw" --output "$base" --label_source rule --rule_gap 2 --left_wait_protect 60 --summary "$root/data/rule_tail_base.summary.json"
    run_sync tail_dataset_augment "$PYTHON_BIN" "$TAIL_AUGMENT" --input "$base" --output "$data" --summary "$root/data/train_tail_p95_direct.summary.json"
}

mkdir -p "$RUN_ROOT"; exec >> "$LOG" 2>&1
on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\n' "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" > "$STATUS"
    notify "Five-track Jinan recovery cohort ${outcome} (exit=${code}): $RUN_ROOT"
}
trap on_exit EXIT
printf 'started_at=%s\nstatus=running\nrun_root=%s\n' "$(timestamp)" "$RUN_ROOT" > "$STATUS"
notify "Resuming five-track Jinan campaign with validated-artifact skipping and 3-GPU execution: $RUN_ROOT"
[[ -s "$BASE_DATA" && -d "$MODEL" ]] || { echo 'Missing campaign prerequisite' >&2; exit 1; }

# The original shared initializer was lost.  A recovery cohort must use one
# freshly trained three-flow initializer for every controlled comparison.
if [[ ! -s "$BASE_STAGE1" ]]; then
    run_sync baseline_threeflow_stage1 bash "$STAGE1" "$RUN_ROOT/baseline_threeflow/stage1" three_flow 64 8
fi
[[ -s "$BASE_STAGE1" ]] || { echo "Recovered baseline Stage-1 checkpoint is missing: $BASE_STAGE1" >&2; exit 1; }
printf 'cohort_type=recovered_after_missing_original_stage1\nhistorical_incomplete_cohort=%s\nbase_stage1=%s\nbase_stage1_sha256=%s\nbase_stage1_data_mode=three_flow\n' \
    "$REPO_ROOT/records/pipeline_runs/jinan_direct_urgency_five_tracks_20260822_rerun1" "$BASE_STAGE1" "$(sha256sum "$BASE_STAGE1" | awk '{print $1}')" > "$RUN_ROOT/provenance.env"

for seed in "${SEEDS[@]}"; do run_arm seed_expansion wait2_mixed_rank "$seed" "$BASE_DATA" "$BASE_STAGE1" 64 8; done
for seed in "${SEEDS[@]}"; do run_arm seed_expansion wait225_mixed_rank "$seed" "$BASE_DATA" "$BASE_STAGE1" 64 8; done
build_tail_data
TAIL_DATA="$RUN_ROOT/tail_direct_labels/data/train_tail_p95_direct.jsonl"
for seed in "${SEEDS[@]}"; do run_arm tail_direct_labels tail_p95_mixed_rank "$seed" "$TAIL_DATA" "$BASE_STAGE1" 64 8; done

STAGE1_SAME="$RUN_ROOT/stage1_sameflow/stage1"
if [[ ! -s "$STAGE1_SAME/best.pt" ]]; then run_sync stage1_sameflow bash "$STAGE1" "$STAGE1_SAME" same_flow 64 8; fi
for seed in "${SEEDS[@]}"; do run_arm stage1_sameflow wait225_mixed_rank "$seed" "$BASE_DATA" "$STAGE1_SAME/best.pt" 64 8; done

STAGE1_CAP="$RUN_ROOT/capacity_pt16_hd128/stage1"
if [[ ! -s "$STAGE1_CAP/best.pt" ]]; then run_sync stage1_capacity bash "$STAGE1" "$STAGE1_CAP" three_flow 128 16; fi
for seed in "${SEEDS[@]}"; do run_arm capacity_pt16_hd128 wait225_mixed_rank "$seed" "$BASE_DATA" "$STAGE1_CAP/best.pt" 128 16; done

run_sync aggregate_results "$PYTHON_BIN" "$AGGREGATE" --run-root "$RUN_ROOT"
