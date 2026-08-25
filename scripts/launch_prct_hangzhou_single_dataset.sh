#!/usr/bin/env bash
# Single-dataset PRCT causal package with a matched frozen-neighbor control.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
if [ "${1:-}" = "--help" ]; then
    echo "Usage: OUTPUT_DIR=<fresh-output-dir> bash scripts/launch_prct_hangzhou_single_dataset.sh"
    exit 0
fi
OUT=${OUTPUT_DIR:-$REPO/records/pipeline_runs/prct_hangzhou_matched_h120_$(date +%Y%m%d_%H%M%S)}
INPUT=$REPO/records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl
RESPONSIVE=$OUT/relabeled_prct_h120.jsonl
FROZEN=$OUT/relabeled_frozen_neighbor_h120.jsonl
ROADNET=$REPO/data/Hangzhou/4_4/roadnet_4_4.json
TRAFFIC=$REPO/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json
TEST_FLOW=${TEST_FLOW:-anon_4_4_hangzhou_real.json}
SOURCE_CKPT=$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt
AGGREGATE=$REPO/records/pipeline_runs/phase7_policy_responsive_h120_followup_20260814/h120_k118_n9_aggregate.json
SEEDS=(2024 42 123 7 11 99 314 2718 9001)

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for path in "$PY" "$INPUT" "$ROADNET" "$TRAFFIC" "$SOURCE_CKPT" "$AGGREGATE"; do
    test -e "$path"
done

echo "[$(date -Is)] B0 determinism smoke: two identical PRCT teachers on GPUs 0 and 1"
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/relabel_policy_responsive.py \
    --input "$INPUT" --output "$OUT/responsive_h120_smoke_a.jsonl" \
    --roadnet "$ROADNET" --traffic_file "$TRAFFIC" --dataset hangzhou \
    --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" \
    --work_dir "$OUT/runtime_responsive_smoke_a" --sim_horizon 120 --max_snapshots 16 \
    --min_action_time 30 --traffic_series_interval 2 --cost_scope global \
    --inference_batch_size 16 --bf16 &
SMOKE_A_PID=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/relabel_policy_responsive.py \
    --input "$INPUT" --output "$OUT/responsive_h120_smoke_b.jsonl" \
    --roadnet "$ROADNET" --traffic_file "$TRAFFIC" --dataset hangzhou \
    --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" \
    --work_dir "$OUT/runtime_responsive_smoke_b" --sim_horizon 120 --max_snapshots 16 \
    --min_action_time 30 --traffic_series_interval 2 --cost_scope global \
    --inference_batch_size 16 --bf16 &
SMOKE_B_PID=$!

wait "$SMOKE_A_PID"
wait "$SMOKE_B_PID"

"$PY" -u projection_train/teacher_validity_analysis.py \
    --inputs "$OUT/responsive_h120_smoke_a.jsonl=smoke_a" "$OUT/responsive_h120_smoke_b.jsonl=smoke_b" \
    --output-dir "$OUT/determinism_smoke"

echo "[$(date -Is)] B1 matched teacher generation: frozen neighbors on GPU 0, responsive neighbors on GPU 1"
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/relabel_policy_responsive.py \
    --input "$INPUT" --output "$FROZEN" \
    --roadnet "$ROADNET" --traffic_file "$TRAFFIC" --dataset hangzhou \
    --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" \
    --work_dir "$OUT/runtime_frozen_h120" --sim_horizon 120 --neighbor-mode frozen \
    --min_action_time 30 --traffic_series_interval 2 --cost_scope global \
    --inference_batch_size 16 --bf16 &
FROZEN_PID=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/relabel_policy_responsive.py \
    --input "$INPUT" --output "$RESPONSIVE" \
    --roadnet "$ROADNET" --traffic_file "$TRAFFIC" --dataset hangzhou \
    --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" \
    --work_dir "$OUT/runtime_responsive_h120" --sim_horizon 120 --neighbor-mode responsive \
    --min_action_time 30 --traffic_series_interval 2 --cost_scope global \
    --inference_batch_size 16 --bf16 &
RESPONSIVE_PID=$!

wait "$FROZEN_PID"
wait "$RESPONSIVE_PID"

"$PY" -u projection_train/teacher_validity_analysis.py \
    --inputs "$FROZEN=frozen_h120" "$RESPONSIVE=responsive_h120" \
    --output-dir "$OUT/teacher_comparison"

"$PY" -u scripts/audit_prct_hangzhou.py \
    --input "$INPUT" --responsive "$RESPONSIVE" --frozen "$FROZEN" \
    --artifact "$AGGREGATE" --output-dir "$OUT/audit"

echo "[$(date -Is)] B2 frozen-neighbor H120: K=32,118; n=9 training seeds"
"$PY" -u scripts/run_phase7_policy_responsive.py \
    --labels "$FROZEN" --output_dir "$OUT/train_frozen_neighbor_h120" \
    --label_budgets 32 118 --selection_seed 2024 \
    --training_seeds "${SEEDS[@]}" --eval_seeds 2024 42 123 \
    --train_gpus 0,1,2 --eval_gpus 0 1 2 --loss mixed \
    --dataset hangzhou --test_flow "$TEST_FLOW"

echo "[$(date -Is)] B2 PRCT H120: K=32,118; n=9 training seeds"
"$PY" -u scripts/run_phase7_policy_responsive.py \
    --labels "$RESPONSIVE" --output_dir "$OUT/train_prct_h120" \
    --label_budgets 32 118 --selection_seed 2024 \
    --training_seeds "${SEEDS[@]}" --eval_seeds 2024 42 123 \
    --train_gpus 0,1,2 --eval_gpus 0 1 2 --loss mixed \
    --dataset hangzhou --test_flow "$TEST_FLOW"

echo "[$(date -Is)] matched PRCT single-dataset run completed"
"$PY" scripts/feishu_push.py "Matched PRCT Hangzhou run completed: frozen-neighbor and responsive H120 label audits plus both K=32/118 x 9 training-seed matrices are available at $OUT"
