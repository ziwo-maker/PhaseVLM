#!/usr/bin/env bash
# pt8/hd64 jinan_real — setup phases (zero-shot -> collect -> relabel).
#
# Produces the 3 teacher label files for the full-matrix ablation:
#   relabeled_simexpert_c30.jsonl        (frozen-neighbor c30, policy-free)
#   relabeled_policy_responsive_h30.jsonl (responsive H30, pt8 source policy)
#   relabeled_policy_responsive_h120.jsonl(responsive H120, pt8 source policy)
#
# Idempotent: skips any phase whose output already exists. Run before
# run_jinan_real_pt8_train.sh so relabel outputs can be validated first.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="$REPO/records/pipeline_runs/jinan_real_pt8_full_matrix_20260815"
SRC_STAGE1="$REPO/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
SRC_STAGE2="$REPO/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt"
HD=64
PT=8

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$OUT"
cd "$REPO"

export SOURCE_CKPT="$SRC_STAGE2"
export HIDDEN_DIM="$HD"
export PREFIX_TOKENS="$PT"

# ---- Phase 1: zero-shot (pt8) on jinan_real test flow ----
if [ ! -f "$OUT/zero_shot/eval_summary.json" ]; then
    log "Phase 1/3: zero-shot eval (pt8) on anon_3_4_jinan_real.json"
    OUTPUT_DIR="$OUT" bash scripts/run_jinan_real_zero_shot.sh
else
    log "Phase 1/3: zero-shot exists, skip"
fi

# ---- Phase 2: collect decision states (pt8) ----
if [ ! -f "$OUT/collect/collected_states.jsonl" ]; then
    log "Phase 2/3: collect (pt8) on anon_3_4_jinan_real_2000.json"
    OUTPUT_DIR="$OUT" bash scripts/run_jinan_real_collect.sh
else
    log "Phase 2/3: collect exists, skip"
fi

# ---- Phase 3: relabel (3 teachers in parallel) ----
log "Phase 3/3: relabel (frozen c30 + responsive H30 + H120) in parallel"

if [ ! -f "$OUT/relabeled_simexpert_c30.jsonl" ]; then
    OUTPUT_DIR="$OUT" bash scripts/run_jinan_real_relabel_frozen_c30.sh \
        > "$OUT/relabel_frozen_c30.log" 2>&1 &
    P_C30=$!
else
    log "  frozen c30 relabel exists, skip"; P_C30=""
fi

if [ ! -f "$OUT/relabeled_policy_responsive_h30.jsonl" ]; then
    OUTPUT_DIR="$OUT" SOURCE_CKPT="$SRC_STAGE2" HORIZON=30 GPU=1 \
        bash scripts/run_jinan_real_relabel.sh \
        > "$OUT/relabel_h30.log" 2>&1 &
    P_H30=$!
else
    log "  responsive H30 relabel exists, skip"; P_H30=""
fi

if [ ! -f "$OUT/relabeled_policy_responsive_h120.jsonl" ]; then
    OUTPUT_DIR="$OUT" SOURCE_CKPT="$SRC_STAGE2" HORIZON=120 GPU=2 \
        bash scripts/run_jinan_real_relabel.sh \
        > "$OUT/relabel_h120.log" 2>&1 &
    P_H120=$!
else
    log "  responsive H120 relabel exists, skip"; P_H120=""
fi

rc=0
for pid in $P_C30 $P_H30 $P_H120; do
    if [ -n "$pid" ]; then
        wait "$pid" || rc=1
    fi
done
if [ "$rc" -ne 0 ]; then
    log "ERROR: one or more relabel shards failed — see relabel_*.log"
    exit 1
fi
log "Phase 3/3: relabel done"

# ---- Validate label outputs ----
log "Validating label files..."
"$PY" - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
for name in ("relabeled_simexpert_c30.jsonl",
             "relabeled_policy_responsive_h30.jsonl",
             "relabeled_policy_responsive_h120.jsonl"):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        print(f"  MISSING {name}")
        sys.exit(1)
    n = 0
    acts = {}
    snaps = set()
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line); n += 1
        snaps.add(r.get("snapshot_path", ""))
        acts[r.get("expert_action")] = acts.get(r.get("expert_action"), 0) + 1
    print(f"  {name}: records={n} snapshots={len(snaps)} action_dist={acts}")
    if n == 0:
        print(f"  EMPTY {name}")
        sys.exit(1)
print("validation OK")
EOF

log "SETUP DONE -> $OUT"
