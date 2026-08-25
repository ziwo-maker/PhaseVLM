#!/usr/bin/env bash
# Execute the five evidence-driven Stage 2 improvement tracks serially on a shared GPU set.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ROOT="${1:-$REPO_ROOT/records/pipeline_runs/jinan_direct_urgency_five_tracks_20260822}"
LAUNCH_ARM="$REPO_ROOT/scripts/launch_jinan_wait25_improvement_arm.sh"
STAGE1="$REPO_ROOT/scripts/run_stage1_direct_urgency_ablation.sh"
TAIL_BUILDER="$REPO_ROOT/scripts/build_queue_wait_distill_dataset.py"
TAIL_AUGMENT="$REPO_ROOT/scripts/add_direct_phase_tail_labels.py"
ONLINE="$REPO_ROOT/projection_train/run_projection_online.py"
AGGREGATE="$REPO_ROOT/scripts/aggregate_jinan_direct_urgency_five_track_campaign.py"
MODEL=/home/data/model/qwen-vl-8B
BASE_DATA="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl"
BASE_STAGE1="$REPO_ROOT/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
EPOCH8="$REPO_ROOT/records/pipeline_runs/jinan_real_wait25_improvement_20260822/arms/wait25_strong_rank/seed_42/stage2/epoch_8.pt"
LOG="$RUN_ROOT/campaign.log"
STATUS="$RUN_ROOT/campaign.status"
FEISHU="$REPO_ROOT/scripts/feishu_push.py"
SEEDS=(314 9001 2718)

notify() { [[ -f "$FEISHU" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU" "$1" || true; }
timestamp() { date -Is; }

require_idle() {
    local gpu used
    for gpu in 0 1 2; do
        used="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
        [[ -n "$used" && "$used" -lt 500 ]] || { echo "GPU $gpu is busy ($used MiB)" >&2; return 1; }
    done
}

snapshot() {
    local label="$1" pid="$2"
    {
        printf '\n[%s] %s pid=%s\n' "$(timestamp)" "$label" "$pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 120 || true
        timeout --foreground 10s tail -n 100 "$LOG" || true
        timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|killed' "$LOG" | tail -n 40 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$RUN_ROOT/startup_monitor.log" 2>&1
}

run_sync() {
    local label="$1"; shift
    printf '[%s] START %s\n' "$(timestamp)" "$label"
    "$@"
    printf '[%s] DONE %s\n' "$(timestamp)" "$label"
}

run_arm() {
    local track="$1" arm="$2" seed="$3" data="$4" align="$5" hd="$6" pt="$7"
    local arm_root="$RUN_ROOT/$track/arms/$arm/seed_$seed"
    [[ ! -e "$arm_root" ]] || { echo "Existing run target: $arm_root" >&2; return 1; }
    require_idle
    printf '[%s] START stage2 track=%s arm=%s seed=%s\n' "$(timestamp)" "$track" "$arm" "$seed"
    WAIT25_IMPROVEMENT_GPUS=0,1,2 WAIT25_IMPROVEMENT_EVAL_GPUS=0,1,2 \
    WAIT25_IMPROVEMENT_DATA_PATH="$data" WAIT25_IMPROVEMENT_ALIGN_CKPT="$align" \
    WAIT25_IMPROVEMENT_HIDDEN_DIM="$hd" WAIT25_IMPROVEMENT_PREFIX_TOKENS="$pt" \
        setsid bash "$LAUNCH_ARM" "$RUN_ROOT/$track" "$arm" "$seed" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    wait "$pid"
    printf '[%s] DONE stage2 track=%s arm=%s seed=%s\n' "$(timestamp)" "$track" "$arm" "$seed"
}

build_tail_data() {
    local root="$RUN_ROOT/tail_direct_labels" raw="$RUN_ROOT/tail_direct_labels/collected/state_action.json"
    local base="$root/data/rule_tail_base.jsonl" data="$root/data/train_tail_p95_direct.jsonl"
    mkdir -p "$root/collected" "$root/data"
    require_idle
    run_sync tail_state_collection env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$ONLINE" \
        --memo five_track_tail_collect --projection_ckpt "$RUN_ROOT/seed_expansion/arms/wait2_mixed_rank/seed_314/stage2/best.pt" \
        --llm_path "$MODEL" --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --num_rounds 1 --seed 2024 \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 --deterministic --collect_states_output "$root/collected/decision_states.jsonl" \
        --result_json "$root/collected/result.json"
    local work_dir
    work_dir="$(jq -r '.work_dir' "$root/collected/result.json")"
    [[ -s "$work_dir/state_action.json" ]] || { echo "Missing collected state_action: $work_dir/state_action.json" >&2; return 1; }
    cp "$work_dir/state_action.json" "$raw"
    run_sync tail_rule_anchor "$PYTHON_BIN" "$TAIL_BUILDER" --state_action "$raw" --output "$base" --label_source rule --rule_gap 2 --left_wait_protect 60 --summary "$root/data/rule_tail_base.summary.json"
    run_sync tail_dataset_augment "$PYTHON_BIN" "$TAIL_AUGMENT" --input "$base" --output "$data" --summary "$root/data/train_tail_p95_direct.summary.json"
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\nrun_root=%s\n' "$(timestamp)" "$RUN_ROOT" > "$STATUS"
        notify "Five-track Jinan direct-urgency campaign completed: $RUN_ROOT"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\nrun_root=%s\n' "$(timestamp)" "$code" "$RUN_ROOT" > "$STATUS"
        notify "Five-track Jinan direct-urgency campaign stopped (exit=${code}); inspect $LOG"
    fi
}
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" && -s "$BASE_DATA" && -s "$BASE_STAGE1" && -s "$EPOCH8" && -d "$MODEL" ]] || { echo "Missing campaign prerequisite" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
exec >> "$LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\ntracks=checkpoint_tiebreak,seed_expansion,tail_direct_labels,stage1_sameflow,capacity_pt16_hd128\n' "$(timestamp)" "$RUN_ROOT" > "$STATUS"
printf 'git_head=%s\nbase_data_sha256=%s\nbase_stage1_sha256=%s\n' "$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD)" "$(sha256sum "$BASE_DATA" | awk '{print $1}')" "$(sha256sum "$BASE_STAGE1" | awk '{print $1}')" > "$RUN_ROOT/provenance.env"
notify "Five-track Jinan direct-urgency campaign launched: checkpoint tie-break, seed expansion, direct tail labels, same-flow Stage 1, and pt16/hd128 capacity."

# Track 2: existing checkpoint, no training and no online selection.
mkdir -p "$RUN_ROOT/checkpoint_tiebreak"
require_idle
run_sync checkpoint_tiebreak_epoch8 env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$ONLINE" \
    --memo five_track_strong_rank_epoch8 --projection_ckpt "$EPOCH8" --llm_path "$MODEL" --dataset jinan --traffic_file anon_3_4_jinan_real.json \
    --run_counts 3600 --num_rounds 1 --seed 2024 --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --inference_batch_size 64 --bf16 --deterministic --disable_snapshots --result_json "$RUN_ROOT/checkpoint_tiebreak/epoch8_result.json"

# Track 3: uncertainty reduction with new independent training seeds.
for seed in "${SEEDS[@]}"; do run_arm seed_expansion wait2_mixed_rank "$seed" "$BASE_DATA" "$BASE_STAGE1" 64 8; done
for seed in "${SEEDS[@]}"; do run_arm seed_expansion wait225_mixed_rank "$seed" "$BASE_DATA" "$BASE_STAGE1" 64 8; done

# Track 1: collect direct per-phase P95 values, then train the new tail-aware objective.
build_tail_data
TAIL_DATA="$RUN_ROOT/tail_direct_labels/data/train_tail_p95_direct.jsonl"
for seed in "${SEEDS[@]}"; do run_arm tail_direct_labels tail_p95_mixed_rank "$seed" "$TAIL_DATA" "$BASE_STAGE1" 64 8; done

# Track 4: change only Stage-1 data scope, then apply the best average-objective Stage-2 arm.
run_sync stage1_sameflow "$STAGE1" "$RUN_ROOT/stage1_sameflow/stage1" same_flow 64 8
for seed in "${SEEDS[@]}"; do run_arm stage1_sameflow wait225_mixed_rank "$seed" "$BASE_DATA" "$RUN_ROOT/stage1_sameflow/stage1/best.pt" 64 8; done

# Track 5: controlled architecture capacity, using the multi-flow Stage-1 protocol.
run_sync stage1_capacity "$STAGE1" "$RUN_ROOT/capacity_pt16_hd128/stage1" three_flow 128 16
for seed in "${SEEDS[@]}"; do run_arm capacity_pt16_hd128 wait225_mixed_rank "$seed" "$BASE_DATA" "$RUN_ROOT/capacity_pt16_hd128/stage1/best.pt" 128 16; done

run_sync aggregate_results "$PYTHON_BIN" "$AGGREGATE" --run-root "$RUN_ROOT"
