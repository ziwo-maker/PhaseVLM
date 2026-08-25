#!/usr/bin/env bash
# AWT-reweighted policy-responsive H120 reproduction (in-network jinan).
# Reuses existing collected snapshots (source-policy collection is independent
# of cost weights), re-relabels with AWT=3.0 cost weight, trains K=32 & K=118
# (9 seeds), and aggregates. One-shot detached pipeline.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
BASE="$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814"
OUT="$REPO/records/pipeline_runs/jinan_real_responsive_h120_awtw3_20260815"

mkdir -p "$OUT"
if [ ! -e "$OUT/collect" ]; then
    ln -s "$BASE/collect" "$OUT/collect"
fi

log() { echo "[$(date '+%F %T')] $*"; }

{
log "=== PHASE 1/3: AWT-reweight relabel (att=0.5 aql=0.5 awt=3.0 spillback=0.2) ==="
OUTPUT_DIR="$OUT" COST_ATT_WEIGHT=0.5 COST_AQL_WEIGHT=0.5 COST_AWT_WEIGHT=3.0 COST_SPILLBACK_WEIGHT=0.2 \
    bash scripts/run_jinan_real_relabel_parallel.sh

log "=== PHASE 2/3: train K=32 & K=118 (9 seeds) ==="
OUTPUT_DIR="$OUT" bash scripts/run_jinan_real_train.sh

log "=== PHASE 3/3: aggregate ==="
"$PY" scripts/aggregate_jinan_real_h120.py --output_dir "$OUT" --zero_shot_dir "$BASE/zero_shot"

log "PIPELINE DONE"
} 2>&1 | tee "$OUT/pipeline.log"
