#!/usr/bin/env bash
# jinan_real_2500 held-out — full pipeline using the pt8/hd64/max_length=416 source
# policy. Chains setup (zero-shot/collect/relabel) -> full matrix train/eval.
# Each stage script is idempotent; fails fast with a marker.
set -uo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
cd "$REPO"

log() { echo "[$(date '+%F %T')] $*"; }

"$PY" scripts/feishu_push.py "jinan2500 pt8(ml416) pipeline START (setup -> full matrix)"

log "=== SETUP (zero-shot / collect / relabel) ==="
bash scripts/run_jinan2500_pt8_setup.sh || { log "SETUP FAILED"; "$PY" scripts/feishu_push.py "jinan2500 pt8 SETUP FAILED"; exit 1; }
"$PY" scripts/feishu_push.py "jinan2500 pt8 setup DONE"

log "=== FULL MATRIX (3 arms x K={32,118} x 9 seeds) ==="
bash scripts/run_jinan2500_pt8_train.sh || { log "TRAIN FAILED"; "$PY" scripts/feishu_push.py "jinan2500 pt8 TRAIN FAILED"; exit 1; }

log "=== ALL DONE ==="
