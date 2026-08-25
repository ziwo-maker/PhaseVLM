#!/usr/bin/env bash
# jinan_real_2500 held-out — full pipeline orchestrator.
#   Stage 1+2 (source, exclude 2500) -> setup (zero-shot/collect/relabel) -> full matrix.
# Each stage script is idempotent; this just chains them and fails fast with a marker.
set -uo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
cd "$REPO"

log() { echo "[$(date '+%F %T')] $*"; }

"$PY" scripts/feishu_push.py "jinan2500 held-out pipeline START (stage1+2 -> setup -> full matrix)"

log "=== STAGE 1+2 (source policy, held-out) ==="
bash scripts/run_jinan2500_stage1_stage2.sh || { log "STAGE12 FAILED"; "$PY" scripts/feishu_push.py "jinan2500 STAGE12 FAILED"; exit 1; }
"$PY" scripts/feishu_push.py "jinan2500 stage1+2 DONE"

log "=== SETUP (zero-shot / collect / relabel) ==="
bash scripts/run_jinan2500_setup.sh || { log "SETUP FAILED"; "$PY" scripts/feishu_push.py "jinan2500 SETUP FAILED"; exit 1; }
"$PY" scripts/feishu_push.py "jinan2500 setup DONE"

log "=== FULL MATRIX (3 arms x K={32,118} x 9 seeds) ==="
bash scripts/run_jinan2500_train.sh || { log "TRAIN FAILED"; "$PY" scripts/feishu_push.py "jinan2500 TRAIN FAILED"; exit 1; }

log "=== ALL DONE ==="
