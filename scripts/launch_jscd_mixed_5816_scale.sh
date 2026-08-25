#!/usr/bin/env bash
# Detached launcher for balanced JSCD P2 + Hangzhou_5816 data-scale training.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT=${OUTPUT_DIR:-$REPO/records/pipeline_runs/jscd_mix5816_scale_$(date +%Y%m%d_%H%M%S)}

if [ "${1:-}" = "--help" ]; then
    echo "Usage: OUTPUT_DIR=<new-dir> bash scripts/launch_jscd_mixed_5816_scale.sh"
    exit 0
fi

test ! -e "$OUT"
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
"$PY" -u scripts/run_jscd_mixed_5816_scale.py \
    --output_dir "$OUT" --python "$PY" --train_gpus 0,1,2 --eval_gpu 0 \
    > "$OUT/launch.log" 2>&1
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
    MESSAGE="JSCD mixed-5816 scale experiment completed: $OUT. See aggregate.json and status.json."
else
    MESSAGE="JSCD mixed-5816 scale experiment exited rc=$RC: $OUT. See launch.log and status.json."
fi
"$PY" scripts/feishu_push.py "$MESSAGE" || true
exit "$RC"
