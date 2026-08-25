#!/usr/bin/env bash
# JSCD P2 pilot: P1 mechanism gate, then three-card DDP F/P/J second-pass runs.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT=${OUTPUT_DIR:-$REPO/records/pipeline_runs/jscd_p2_pilot_$(date +%Y%m%d_%H%M%S)}
INPUT=$REPO/records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl
ROADNET=$REPO/data/Hangzhou/4_4/roadnet_4_4.json
TRAFFIC=$REPO/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json
PI0=$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt
FIRST_RUN=$REPO/records/pipeline_runs/prct_hangzhou_matched_h120_20260818_212203
TEST_FLOW=${TEST_FLOW:-anon_4_4_hangzhou_real.json}
SEEDS=(2024 42 123)
P1_POLICY_SEED=2024
P1_HOLDOUT_SNAPSHOTS=32
P2_SNAPSHOTS=86

if [ "${1:-}" = "--help" ]; then
    echo "Usage: OUTPUT_DIR=<fresh-output-dir> bash scripts/launch_jscd_p2_pilot.sh"
    exit 0
fi

test ! -e "$OUT"
mkdir -p "$OUT"
exec > "$OUT/launch.log" 2>&1
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for path in "$PY" "$INPUT" "$ROADNET" "$TRAFFIC" "$PI0"; do
    test -e "$path"
done
for seed in "${SEEDS[@]}"; do
    test -s "$FIRST_RUN/train_prct_h120/c30/k118/tr$seed/checkpoints/best.pt"
    test -s "$FIRST_RUN/train_frozen_neighbor_h120/c30/k118/tr$seed/checkpoints/best.pt"
done

relabel() {
    local gpu=$1 input=$2 output=$3 runtime=$4 policy=$5 neighbor_mode=$6
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/relabel_policy_responsive.py \
        --input "$input" --output "$output" --roadnet "$ROADNET" \
        --traffic_file "$TRAFFIC" --dataset hangzhou --num_rows 4 --num_cols 4 \
        --source_policy_ckpt "$policy" --work_dir "$runtime" --sim_horizon 120 \
        --neighbor-mode "$neighbor_mode" --min_action_time 30 \
        --traffic_series_interval 2 --cost_scope global --inference_batch_size 16 --bf16
}

echo "[$(date -Is)] partition source states for the P1 mechanism gate and P2 labels"
"$PY" -u scripts/split_jscd_snapshot_pool.py \
    --input "$INPUT" --p1-output "$OUT/p1/holdout_states.jsonl" \
    --p2-output "$OUT/p2/training_states.jsonl" --manifest "$OUT/snapshot_partition.json" \
    --holdout-snapshots "$P1_HOLDOUT_SNAPSHOTS" --seed 20260819

echo "[$(date -Is)] P1: pi0 and pi1 continuation labels run in parallel on GPUs 0 and 1"
relabel 0 "$OUT/p1/holdout_states.jsonl" "$OUT/p1/pi0_continuation.jsonl" \
    "$OUT/p1/runtime_pi0" "$PI0" responsive &
P1_PI0_PID=$!
relabel 1 "$OUT/p1/holdout_states.jsonl" "$OUT/p1/pi1_continuation.jsonl" \
    "$OUT/p1/runtime_pi1" "$FIRST_RUN/train_prct_h120/c30/k118/tr$P1_POLICY_SEED/checkpoints/best.pt" responsive &
P1_PI1_PID=$!
wait "$P1_PI0_PID"
wait "$P1_PI1_PID"

echo "[$(date -Is)] P1: enforce pre-registered self-consistency gate"
"$PY" -u scripts/analyze_jscd_self_consistency.py \
    --pi0 "$OUT/p1/pi0_continuation.jsonl" --pi1 "$OUT/p1/pi1_continuation.jsonl" \
    --output "$OUT/p1/self_consistency.json"

echo "[$(date -Is)] P2 labels: F/P/J(2024) in parallel on all three GPUs"
relabel 0 "$OUT/p2/training_states.jsonl" "$OUT/p2/frozen_second_pass.jsonl" \
    "$OUT/p2/runtime_frozen" "$PI0" frozen &
FROZEN_PID=$!
relabel 1 "$OUT/p2/training_states.jsonl" "$OUT/p2/static_prct_second_pass.jsonl" \
    "$OUT/p2/runtime_static_prct" "$PI0" responsive &
STATIC_PID=$!
relabel 2 "$OUT/p2/training_states.jsonl" "$OUT/p2/jscd_second_pass_tr2024.jsonl" \
    "$OUT/p2/runtime_jscd_tr2024" "$FIRST_RUN/train_prct_h120/c30/k118/tr2024/checkpoints/best.pt" responsive &
JSCD_2024_PID=$!
wait "$FROZEN_PID"
wait "$STATIC_PID"
wait "$JSCD_2024_PID"

echo "[$(date -Is)] P2 labels: JSCD continuation pi1 for seeds 42 and 123"
relabel 0 "$OUT/p2/training_states.jsonl" "$OUT/p2/jscd_second_pass_tr42.jsonl" \
    "$OUT/p2/runtime_jscd_tr42" "$FIRST_RUN/train_prct_h120/c30/k118/tr42/checkpoints/best.pt" responsive &
JSCD_42_PID=$!
relabel 1 "$OUT/p2/training_states.jsonl" "$OUT/p2/jscd_second_pass_tr123.jsonl" \
    "$OUT/p2/runtime_jscd_tr123" "$FIRST_RUN/train_prct_h120/c30/k118/tr123/checkpoints/best.pt" responsive &
JSCD_123_PID=$!
wait "$JSCD_42_PID"
wait "$JSCD_123_PID"

run_second_pass() {
    local arm=$1 seed=$2 labels=$3 init_ckpt=$4
    echo "[$(date -Is)] P2 train: arm=$arm seed=$seed (DDP GPUs 0,1,2)"
    "$PY" -u scripts/run_phase7_crcd.py \
        --f30 "$labels" --f120 "$labels" --f300 "$labels" \
        --output_dir "$OUT/train_$arm" --teachers c30 --label_budgets "$P2_SNAPSHOTS" \
        --selection_seed 20260819 --training_seeds "$seed" --eval_seeds 2024 \
        --train_gpus 0,1,2 --eval_gpus 0 --loss mixed \
        --dataset hangzhou --test_flow "$TEST_FLOW" \
        --source_stage2_ckpt "$init_ckpt"
}

# F controls a second frozen-teacher pass; P and J start from the same pi1.
for seed in "${SEEDS[@]}"; do
    run_second_pass frozen "$seed" "$OUT/p2/frozen_second_pass.jsonl" \
        "$FIRST_RUN/train_frozen_neighbor_h120/c30/k118/tr$seed/checkpoints/best.pt"
    run_second_pass static_prct "$seed" "$OUT/p2/static_prct_second_pass.jsonl" \
        "$FIRST_RUN/train_prct_h120/c30/k118/tr$seed/checkpoints/best.pt"
    run_second_pass jscd "$seed" "$OUT/p2/jscd_second_pass_tr$seed.jsonl" \
        "$FIRST_RUN/train_prct_h120/c30/k118/tr$seed/checkpoints/best.pt"
done

echo "[$(date -Is)] JSCD P2 pilot primary-flow training/evaluation complete"
