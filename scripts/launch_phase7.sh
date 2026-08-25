#!/bin/bash
# Launch Phase 7 (CRCD) teacher study: build -> train -> online eval.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
cd "$REPO"

F30="$REPO/records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl"
F120="$REPO/records/pipeline_runs/teacher_validity_20260813/relabeled_h120.jsonl"
F300="$REPO/records/pipeline_runs/teacher_validity_20260813/relabeled_h300.jsonl"

OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/phase7_crcd_20260813}"
TEACHERS="${TEACHERS:-c30 c300 crcd_nogate crcd}"
BUDGETS="${BUDGETS:-8 32 118}"
TRAIN_SEEDS="${TRAIN_SEEDS:-2024 42 123}"
EVAL_SEEDS="${EVAL_SEEDS:-2024 42 123}"
ONLY="${ONLY:-}"

ARGS=(
    --f30 "$F30" --f120 "$F120" --f300 "$F300"
    --output_dir "$OUT"
    --teachers $TEACHERS
    --label_budgets $BUDGETS
    --training_seeds $TRAIN_SEEDS
    --eval_seeds $EVAL_SEEDS
    --selection_seed 2024
    --train_gpus "0,1"
    --eval_gpus 0 1 2
)

if [ -n "$ONLY" ]; then
    ARGS+=(--only "$ONLY")
fi

exec "$PY" -u scripts/run_phase7_crcd.py "${ARGS[@]}"
