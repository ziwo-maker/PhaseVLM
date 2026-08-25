#!/usr/bin/env bash
# Sharded policy-responsive H120 relabel across N GPUs for jinan.
# Splits the collected snapshot JSONL into contiguous snapshot shards, runs one
# relabeler per GPU in parallel, then concatenates outputs + merges coverage.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814}"
INPUT="$OUT/collect/collected_states.jsonl"
ROADNET="$REPO/data/Jinan/3_4/roadnet_3_4.json"
TRAFFIC="${TRAFFIC_FILE:-$REPO/data/Jinan/3_4/anon_3_4_jinan_real_2000.json}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt}"
HORIZON="${HORIZON:-120}"
NUM_SHARDS="${NUM_SHARDS:-3}"
# space-separated GPU indices; length must equal NUM_SHARDS
GPUS="${GPUS:-0 1 2}"
# composite cost weights (min-max normalized metric mix); override via env
COST_ATT_WEIGHT="${COST_ATT_WEIGHT:-1.0}"
COST_AQL_WEIGHT="${COST_AQL_WEIGHT:-1.0}"
COST_AWT_WEIGHT="${COST_AWT_WEIGHT:-1.0}"
COST_SPILLBACK_WEIGHT="${COST_SPILLBACK_WEIGHT:-0.2}"

cd "$REPO"

# --- split input into contiguous snapshot shards (sorted to mirror relabeler) ---
"$PY" - "$INPUT" "$OUT" "$NUM_SHARDS" <<'EOF'
import json, sys, os
input_path, out_dir, num_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
records = []
snapshot_order = []
snapshot_indices = {}
with open(input_path) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        idx = len(records)
        records.append(r)
        sp = str(r.get("snapshot_path", ""))
        if sp and os.path.exists(sp):
            if sp not in snapshot_indices:
                snapshot_indices[sp] = []
                snapshot_order.append(sp)
            snapshot_indices[sp].append(idx)
snapshot_order.sort()
total = len(snapshot_order)
per = (total + num_shards - 1) // num_shards
for s in range(num_shards):
    start = s * per
    end = min(total, start + per)
    if start >= end:
        continue
    snaps = snapshot_order[start:end]
    idxs = set()
    for sp in snaps:
        idxs.update(snapshot_indices[sp])
    out = os.path.join(out_dir, f"collected_states.shard{s}.jsonl")
    with open(out, "w") as f:
        for i, r in enumerate(records):
            if i in idxs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"shard{s}: snapshots {start}..{end-1} ({len(snaps)} snapshots, {len(idxs)} records) -> {out}", flush=True)
print(f"total_snapshots={total}", flush=True)
EOF

read -ra GPU_ARR <<< "$GPUS"

run_shard() {
    local shard="$1"
    local gpu="$2"
    local shard_input="$OUT/collected_states.shard${shard}.jsonl"
    local shard_output="$OUT/relabeled_policy_responsive_h${HORIZON}.shard${shard}.jsonl"
    local runtime="$OUT/runtime_h${HORIZON}_shard${shard}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/relabel_policy_responsive.py \
        --input "$shard_input" \
        --output "$shard_output" \
        --roadnet "$ROADNET" \
        --traffic_file "$TRAFFIC" \
        --dataset jinan \
        --num_rows 3 --num_cols 4 \
        --source_policy_ckpt "$SOURCE_CKPT" \
        --llm_path /home/data/model/qwen-vl-8B \
        --work_dir "$runtime" \
        --sim_horizon "$HORIZON" \
        --min_action_time 30 \
        --traffic_series_interval 2 \
        --cost_scope global \
        --cost_att_weight "$COST_ATT_WEIGHT" \
        --cost_aql_weight "$COST_AQL_WEIGHT" \
        --cost_awt_weight "$COST_AWT_WEIGHT" \
        --cost_spillback_weight "$COST_SPILLBACK_WEIGHT" \
        --inference_batch_size 16 \
        --bf16 \
        > "$OUT/relabel_h${HORIZON}.shard${shard}.log" 2>&1
}

pids=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_ARR[$s]:-0}"
    run_shard "$s" "$gpu" &
    pids+=("$!")
done
echo "launched $NUM_SHARDS relabel shards: pids ${pids[*]}" | tee -a "$OUT/relabel_h${HORIZON}.shards.log"

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "one or more shards failed; see $OUT/relabel_h${HORIZON}.shard*.log" | tee -a "$OUT/relabel_h${HORIZON}.shards.log"
    exit "$rc"
fi

# --- concatenate outputs (preserves sorted-snapshot order) ---
final="$OUT/relabeled_policy_responsive_h${HORIZON}.jsonl"
: > "$final"
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    cat "$OUT/relabeled_policy_responsive_h${HORIZON}.shard${s}.jsonl" >> "$final"
done

# --- merge coverage ---
"$PY" - "$OUT" "$HORIZON" "$NUM_SHARDS" <<'EOF'
import json, sys, os
out_dir, horizon, num_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
merged = None
for s in range(num_shards):
    cov_path = os.path.join(out_dir, f"relabeled_policy_responsive_h{horizon}.shard{s}_coverage.json")
    with open(cov_path) as f:
        cov = json.load(f)
    if merged is None:
        merged = dict(cov)
        merged["shards"] = []
    else:
        for k in ("total_records", "total_snapshots", "selected_snapshots",
                  "processed_snapshots", "skipped_snapshots", "skipped_records",
                  "candidate_rollouts"):
            merged[k] = merged.get(k, 0) + cov.get(k, 0)
    merged["shards"].append(os.path.basename(cov_path))
out = os.path.join(out_dir, f"relabeled_policy_responsive_h{horizon}_coverage.json")
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print("merged coverage ->", out, "processed_snapshots=", merged["processed_snapshots"],
      "candidate_rollouts=", merged["candidate_rollouts"], flush=True)
EOF

echo "relabel done -> $final" | tee -a "$OUT/relabel_h${HORIZON}.shards.log"
