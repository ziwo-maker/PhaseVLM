#!/usr/bin/env bash
# pt8/hd64 jinan_real — training + eval phase (full matrix).
#
# 3 teacher arms (frozen c30 / responsive H30 / responsive H120), each
# K={32,118} x 9 training seeds = 18 configs, run serially per GPU on one of
# GPUs 0/1/2 in parallel. Uses run_phase7_crcd.py --teachers c30 for all three
# (the "argmin over candidate_action_waits" teacher), distinguished by label
# file and output subdirectory.
#
# Prereq: run_jinan_real_pt8_setup.sh must have completed (labels present).
# Idempotent: run_phase7_crcd.py skips already-complete configs.
set -uo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="$REPO/records/pipeline_runs/jinan_real_pt8_full_matrix_20260815"
SRC_STAGE1="$REPO/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
SRC_STAGE2="$REPO/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt"
TRAIN_SEEDS="2024 42 123 7 11 99 314 2718 9001"
EVAL_SEEDS="2024 42 123"

log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p "$OUT"
cd "$REPO"

run_arm() {
    local gpu="$1" label="$2" subdir="$3"
    log "[gpu=$gpu] launching arm $subdir (label=$(basename "$label"))"
    "$PY" -u scripts/run_phase7_crcd.py \
        --f30 "$label" --f120 "$label" --f300 "$label" \
        --output_dir "$OUT/$subdir" \
        --teachers c30 \
        --label_budgets 32 118 \
        --selection_seed 2024 \
        --training_seeds $TRAIN_SEEDS \
        --eval_seeds $EVAL_SEEDS \
        --loss mixed \
        --train_gpus "$gpu" \
        --eval_gpus "$gpu" \
        --dataset jinan \
        --test_flow anon_3_4_jinan_real.json \
        --source_stage1_ckpt "$SRC_STAGE1" \
        --source_stage2_ckpt "$SRC_STAGE2" \
        --hidden_dim 64 --prefix_tokens 8 \
        > "$OUT/train_$(basename "$subdir").log" 2>&1
}

# Verify labels exist before launching.
for label in relabeled_simexpert_c30.jsonl \
            relabeled_policy_responsive_h30.jsonl \
            relabeled_policy_responsive_h120.jsonl; do
    if [ ! -f "$OUT/$label" ]; then
        log "ERROR: missing label $OUT/$label — run setup first"
        exit 1
    fi
done

log "=== launching 3 training arms (frozen c30 / H30 / H120) on GPUs 0/1/2 ==="
run_arm 0 "$OUT/relabeled_simexpert_c30.jsonl" "train_frozen_c30" &
p0=$!
run_arm 1 "$OUT/relabeled_policy_responsive_h30.jsonl" "train_h30" &
p1=$!
run_arm 2 "$OUT/relabeled_policy_responsive_h120.jsonl" "train_h120" &
p2=$!

log "training pids: $p0 $p1 $p2"
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
wait "$p2" || rc=1

log "=== training arms finished (rc=$rc) — aggregating ==="
"$PY" scripts/aggregate_jinan_pt8_full_matrix.py --output_dir "$OUT"

# ---- best-effort Feishu notification ----
"$PY" - "$OUT" "$rc" <<'EOF'
import json, sys, os
out, rc = sys.argv[1], int(sys.argv[2])
try:
    cfg = json.load(open("/root/.claude/feishu.json"))
    if cfg.get("mode") != "push" or not cfg.get("webhook_url"):
        sys.exit(0)
    agg_path = os.path.join(out, "full_matrix_aggregate.json")
    if os.path.exists(agg_path):
        agg = json.load(open(agg_path))
        zs = agg.get("zero_shot") or {}
        lines = [f"pt8 jinan_real full matrix {'DONE' if rc==0 else 'PARTIAL'}"]
        if zs:
            lines.append("zero-shot AWT=%.2f" % zs.get("AWT", float('nan')))
        for k, a in agg.get("arms", {}).items():
            g = a["aggregate"]
            lines.append("%s AWT=%.2f±%.2f (n=%d)%s" % (
                k, g["AWT"]["mean"], g["AWT"]["std"], g["AWT"]["n"],
                (" Δ=%.2f" % g["delta_vs_zero_shot"]["AWT"]) if "delta_vs_zero_shot" in g else ""))
        text = "\n".join(lines)
    else:
        text = f"pt8 jinan_real full matrix finished rc={rc} (aggregate not yet available)"
    import urllib.request
    payload = {"msg_type": "text", "content": {"text": text}}
    req = urllib.request.Request(cfg["webhook_url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    print("feishu sent")
except Exception as e:
    print("feishu skipped:", type(e).__name__)
EOF

log "TRAIN DONE (rc=$rc) -> $OUT/full_matrix_aggregate.json"
exit "$rc"
