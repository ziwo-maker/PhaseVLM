#!/usr/bin/env bash
# Experiment #1: multi-horizon SimExpert teacher validity (Phase 7 premise test).
#
# Reuses the existing 30s Stage-3 teacher label file and re-runs the SAME 118
# snapshots at 120s and 300s horizons, then analyzes cross-horizon agreement.
#
# Input : records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl  (30s teacher, 118 snapshots x 16 intersections)
# Output: records/pipeline_runs/teacher_validity_20260813/
set -euo pipefail

PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
REPO=/home/code/LLMSTSC
cd "$REPO"

OUT="$REPO/records/pipeline_runs/teacher_validity_20260813"
INPUT="$REPO/records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl"
ROADNET="$REPO/data/Hangzhou/4_4/roadnet_4_4.json"
TRAFFIC="$REPO/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json"
DATASET=hangzhou

mkdir -p "$OUT"

relabel() {
    local H=$1
    echo "=== relabel sim_horizon=$H ==="
    "$PY" scripts/relabel_simexpert.py \
        --input "$INPUT" \
        --output "$OUT/relabeled_h${H}.jsonl" \
        --roadnet "$ROADNET" \
        --traffic_file "$TRAFFIC" \
        --dataset "$DATASET" \
        --work_dir "$OUT/work_h${H}" \
        --sim_horizon "$H" \
        --cost_att_weight 1.0 --cost_aql_weight 1.0 --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
        > "$OUT/relabel_h${H}.log" 2>&1
    echo "done H=$H"
}

relabel 120 &
P1=$!
relabel 300 &
P2=$!

wait "$P1" "$P2"

echo "=== both relabels done, running analysis ==="
"$PY" projection_train/teacher_validity_analysis.py \
    --inputs \
        "$INPUT=30" \
        "$OUT/relabeled_h120.jsonl=120" \
        "$OUT/relabeled_h300.jsonl=300" \
    --output-dir "$OUT"

echo "=== teacher_validity done ==="
