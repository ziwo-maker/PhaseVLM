#!/usr/bin/env bash
# Frozen-neighbor SimExpert c30 relabel for jinan (policy-free; no LLM, CPU-only).
# Relabels the collected snapshots with the 30s frozen counterfactual teacher
# (relabel_simexpert.py). Output feeds run_phase7_crcd.py --teachers c30.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814}"
INPUT="$OUT/collect/collected_states.jsonl"
ROADNET="$REPO/data/Jinan/3_4/roadnet_3_4.json"
TRAFFIC="${TRAFFIC_FILE:-$REPO/data/Jinan/3_4/anon_3_4_jinan_real_2000.json}"
HORIZON="${HORIZON:-30}"

cd "$REPO"
mkdir -p "$OUT"

label="$OUT/relabeled_simexpert_c${HORIZON}.jsonl"
runtime="$OUT/runtime_simexpert_c${HORIZON}"

"$PY" -u scripts/relabel_simexpert.py \
    --input "$INPUT" \
    --output "$label" \
    --roadnet "$ROADNET" \
    --traffic_file "$TRAFFIC" \
    --dataset jinan \
    --work_dir "$runtime" \
    --sim_horizon "$HORIZON" \
    --cost_att_weight 1.0 \
    --cost_aql_weight 1.0 \
    --cost_awt_weight 1.0 \
    --cost_spillback_weight 0.2 \
    --python_path "$PY" \
    > "$OUT/relabel_simexpert_c${HORIZON}.log" 2>&1

echo "frozen c${HORIZON} relabel done -> $label"
cat "${label%.jsonl}_coverage.json" 2>/dev/null || true
