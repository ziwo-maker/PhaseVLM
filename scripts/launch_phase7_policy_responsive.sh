#!/usr/bin/env bash
# Phase 7 responsive-teacher workflow. Run from a monitored shell/session.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
cd "$REPO"

INPUT="${INPUT:-$REPO/records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl}"
ROADNET="${ROADNET:-$REPO/data/Hangzhou/4_4/roadnet_4_4.json}"
TRAFFIC="${TRAFFIC:-$REPO/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt}"
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/phase7_policy_responsive_$(date +%Y%m%d)}"
HORIZONS="${HORIZONS:-30 120}"
RUN_TRAINING="${RUN_TRAINING:-0}"

mkdir -p "$OUT"

for horizon in $HORIZONS; do
    label="$OUT/relabeled_policy_responsive_h${horizon}.jsonl"
    runtime="$OUT/runtime_h${horizon}"
    "$PY" -u scripts/relabel_policy_responsive.py \
        --input "$INPUT" \
        --output "$label" \
        --roadnet "$ROADNET" \
        --traffic_file "$TRAFFIC" \
        --dataset hangzhou \
        --num_rows 4 --num_cols 4 \
        --source_policy_ckpt "$SOURCE_CKPT" \
        --work_dir "$runtime" \
        --sim_horizon "$horizon" \
        --min_action_time 30 \
        --traffic_series_interval 2 \
        --cost_scope global \
        --inference_batch_size 16 \
        --bf16
done

"$PY" -u projection_train/teacher_validity_analysis.py \
    --inputs \
        "$INPUT=frozen_30" \
        "$OUT/relabeled_policy_responsive_h30.jsonl=responsive_30" \
        "$OUT/relabeled_policy_responsive_h120.jsonl=responsive_120" \
    --output-dir "$OUT/teacher_validity"

if [ "$RUN_TRAINING" = "1" ]; then
    "$PY" -u scripts/run_phase7_policy_responsive.py \
        --labels "$OUT/relabeled_policy_responsive_h30.jsonl" \
        --output_dir "$OUT/train_responsive_h30"
fi
