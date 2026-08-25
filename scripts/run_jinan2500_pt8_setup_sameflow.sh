#!/usr/bin/env bash
# jinan_real_2500 held-out — setup phases using the pt8/hd64/max_length=416 source
# policy (All_best/jinan_2500_best_*.pt): zero-shot -> collect -> relabel (3 teachers).
#
# Test flow   = anon_3_4_jinan_real_2500.json   (held-out target)
# Collect flow= anon_3_4_jinan_real_2500.json   (SAME as test flow — user requirement)
# Source      = pt8/hd64 max_length=416 policy trained on jinan_real ONLY (exp_jinan).
#
# Produces the 3 teacher label files for the full matrix:
#   relabeled_simexpert_c30.jsonl         (frozen-neighbor c30, policy-free, CPU)
#   relabeled_policy_responsive_h30.jsonl  (responsive H30)
#   relabeled_policy_responsive_h120.jsonl (responsive H120)
#
# Idempotent: skips any phase whose output already exists.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="$REPO/records/pipeline_runs/jinan_2500_pt8_full_matrix_sameflow_20260817"
SRC_STAGE2="/home/code/LLMSTSC/All_best/jinan_2500_best_AQL=126.6167，ATT=256.2499，AWT=42.1846.pt"
HD=64
PT=8
MAX_LEN=416
LLM=/home/data/model/qwen-vl-8B
DATASET=jinan
TEST_FLOW=anon_3_4_jinan_real_2500.json
COLLECT_FLOW=anon_3_4_jinan_real_2500.json
# Same-flow: relabel SimExpert replays use the same traffic flow as collect/test.
# Relabel scripts expect a FULL path (their default is $REPO/data/Jinan/3_4/...).
TRAFFIC_FILE="$REPO/data/Jinan/3_4/$COLLECT_FLOW"
SEEDS=(2024 42 123)

log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p "$OUT"
cd "$REPO"

if [ ! -s "$SRC_STAGE2" ]; then
    log "ERROR: source Stage-2 checkpoint missing: $SRC_STAGE2"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: zero-shot (pt8/hd64/ml416 source) on the 2500 test flow
# ---------------------------------------------------------------------------
if [ -f "$OUT/zero_shot/eval_summary.json" ]; then
    log "Phase 1/3: zero-shot exists, skip"
else
    log "Phase 1/3: zero-shot eval on $TEST_FLOW (3 seeds, GPUs 0/1/2, ml416)"
    mkdir -p "$OUT/zero_shot"
    pids=()
    for i in 0 1 2; do
        seed=${SEEDS[$i]}
        CUDA_VISIBLE_DEVICES="$i" "$PY" -u projection_train/run_projection_online.py \
            --memo "jinan2500_pt8_zero_shot_seed${seed}" \
            --projection_ckpt "$SRC_STAGE2" \
            --llm_path "$LLM" \
            --dataset "$DATASET" \
            --traffic_file "$TEST_FLOW" \
            --run_counts 3600 --num_rounds 1 --seed "$seed" \
            --prompt_mode queue_wait_short \
            --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
            --hidden_dim "$HD" --prefix_tokens "$PT" --prefix_mode prefix --max_length "$MAX_LEN" \
            --inference_batch_size 64 --bf16 \
            --result_json "$OUT/zero_shot/result_seed${seed}.json" \
            > "$OUT/zero_shot/eval_seed${seed}.log" 2>&1 &
        pids+=("$!")
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    if [ "$rc" -ne 0 ]; then log "ERROR: zero-shot eval failed"; exit 1; fi

    ZERO_SHOT_DIR="$OUT/zero_shot" "$PY" - <<'EOF'
import json, os, statistics
out = os.environ["ZERO_SHOT_DIR"]
keymap = {"ATT": "test_avg_travel_time_over", "AQL": "test_avg_queue_len_over", "AWT": "test_avg_waiting_time_over"}
per, vals = {}, {m: [] for m in keymap}
for seed in (2024, 42, 123):
    p = os.path.join(out, f"result_seed{seed}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    res = d.get("results", d)
    row = {m: float(res[k]) for m, k in keymap.items()}
    per[f"seed_{seed}"] = row
    for m in keymap:
        vals[m].append(row[m])
summary = {
    "per_seed": per,
    **{m: {"n": len(vals[m]), "mean": statistics.fmean(vals[m]),
           "std": statistics.pstdev(vals[m]) if len(vals[m]) > 1 else 0.0} for m in keymap},
}
json.dump(summary, open(os.path.join(out, "eval_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
EOF
    log "zero-shot done"
fi

# ---------------------------------------------------------------------------
# Phase 2: collect decision states on the collection flow (pt8/hd64/ml416)
# ---------------------------------------------------------------------------
if [ -f "$OUT/collect/collected_states.jsonl" ]; then
    log "Phase 2/3: collect exists, skip"
else
    log "Phase 2/3: collect on $COLLECT_FLOW (GPU 0, ml416)"
    mkdir -p "$OUT/collect"
    CUDA_VISIBLE_DEVICES=0 "$PY" -u projection_train/run_projection_online.py \
        --memo "jinan2500_pt8_collect" \
        --projection_ckpt "$SRC_STAGE2" \
        --llm_path "$LLM" \
        --dataset "$DATASET" \
        --traffic_file "$COLLECT_FLOW" \
        --run_counts 3600 --num_rounds 1 --seed 2024 \
        --prompt_mode queue_wait_short \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim "$HD" --prefix_tokens "$PT" --prefix_mode prefix --max_length "$MAX_LEN" \
        --inference_batch_size 64 --bf16 \
        --collect_states_output "$OUT/collect/collected_states.jsonl" \
        > "$OUT/collect/collect.log" 2>&1
    log "collect done"
fi

# Validate every snapshot parses as JSON before relabel. The SimExpert relabeler
# cannot replay a truncated snapshot, so FILTER OUT any corrupt snapshots and
# continue if enough remain (K=118 needs ~118; the selector uses min(K, n)).
"$PY" - "$OUT/collect/collected_states.jsonl" <<'EOF'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("MISSING collected_states.jsonl"); sys.exit(1)
records = []
for line in open(p):
    line = line.strip()
    if not line:
        continue
    records.append(json.loads(line))

good = set()
bad = {}
for r in records:
    sp = str(r.get("snapshot_path", ""))
    if not sp or sp in good or sp in bad:
        continue
    try:
        with open(sp) as f:
            json.load(f)
        good.add(sp)
    except Exception as e:
        bad[sp] = str(e)

if bad:
    print(f"WARNING: {len(bad)} corrupt snapshot(s) — filtering out (dump is flaky)")
    for sp, e in bad.items():
        print(f"  drop {os.path.basename(sp)}: {e[:60]}")
    if os.path.exists(p + ".orig"):
        os.remove(p + ".orig")
    os.rename(p, p + ".orig")
    clean = [r for r in records if str(r.get("snapshot_path", "")) in good]
    with open(p, "w") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    records = clean

n_snaps = len(good)
print(f"records={len(records)} distinct_snapshots={n_snaps}")
if n_snaps < 100:
    print(f"ERROR: only {n_snaps} clean snapshots (< 100) — insufficient")
    sys.exit(1)
print(f"OK: {n_snaps} clean snapshots ready for relabel")
EOF

# ---------------------------------------------------------------------------
# Phase 3: relabel (frozen c30 + responsive H30 + H120) in parallel
#   c30 : CPU (policy-free)
#   H30 : GPU 0
#   H120: GPU 1 + GPU 2 (sharded)
# ---------------------------------------------------------------------------
log "Phase 3/3: relabel (3 teachers)"

if [ -f "$OUT/relabeled_simexpert_c30.jsonl" ]; then
    log "  frozen c30 exists, skip"; P_C30=""
else
    OUTPUT_DIR="$OUT" TRAFFIC_FILE="$TRAFFIC_FILE" HORIZON=30 bash scripts/run_jinan_real_relabel_frozen_c30.sh \
        > "$OUT/relabel_frozen_c30.log" 2>&1 &
    P_C30=$!
fi

if [ -f "$OUT/relabeled_policy_responsive_h30.jsonl" ]; then
    log "  responsive H30 exists, skip"; P_H30=""
else
    OUTPUT_DIR="$OUT" SOURCE_CKPT="$SRC_STAGE2" TRAFFIC_FILE="$TRAFFIC_FILE" HORIZON=30 GPU=0 \
        bash scripts/run_jinan_real_relabel.sh \
        > "$OUT/relabel_h30.log" 2>&1 &
    P_H30=$!
fi

if [ -f "$OUT/relabeled_policy_responsive_h120.jsonl" ]; then
    log "  responsive H120 exists, skip"; P_H120=""
else
    OUTPUT_DIR="$OUT" SOURCE_CKPT="$SRC_STAGE2" TRAFFIC_FILE="$TRAFFIC_FILE" HORIZON=120 NUM_SHARDS=2 GPUS="1 2" \
        bash scripts/run_jinan_real_relabel_parallel.sh \
        > "$OUT/relabel_h120.log" 2>&1 &
    P_H120=$!
fi

rc=0
for pid in $P_C30 $P_H30 $P_H120; do
    if [ -n "$pid" ]; then wait "$pid" || rc=1; fi
done
if [ "$rc" -ne 0 ]; then
    log "ERROR: one or more relabel jobs failed — see relabel_*.log"
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate label outputs
# ---------------------------------------------------------------------------
log "Validating label files..."
"$PY" - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
for name in ("relabeled_simexpert_c30.jsonl",
             "relabeled_policy_responsive_h30.jsonl",
             "relabeled_policy_responsive_h120.jsonl"):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        print(f"  MISSING {name}"); sys.exit(1)
    n = 0; acts = {}; snaps = set()
    for line in open(p):
        line = line.strip()
        if not line: continue
        r = json.loads(line); n += 1
        snaps.add(r.get("snapshot_path", ""))
        acts[r.get("expert_action")] = acts.get(r.get("expert_action"), 0) + 1
    print(f"  {name}: records={n} snapshots={len(snaps)} action_dist={acts}")
    if n == 0:
        print(f"  EMPTY {name}"); sys.exit(1)
print("validation OK")
EOF

log "SETUP DONE -> $OUT"
