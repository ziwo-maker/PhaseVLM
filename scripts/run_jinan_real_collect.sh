#!/usr/bin/env bash
# Collect decision-point states (with snapshots) on the jinan collection flow
# using the source policy. Output feeds relabel_policy_responsive.py.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814}/collect"
CKPT="${SOURCE_CKPT:-$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
PREFIX_TOKENS="${PREFIX_TOKENS:-32}"
LLM=/home/data/model/qwen-vl-8B
DATASET=jinan
COLLECT_FLOW=anon_3_4_jinan_real_2000.json
GPU="${GPU:-0}"

mkdir -p "$OUT"
cd "$REPO"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u projection_train/run_projection_online.py \
    --memo "jinan_real_collect" \
    --projection_ckpt "$CKPT" \
    --llm_path "$LLM" \
    --dataset "$DATASET" \
    --traffic_file "$COLLECT_FLOW" \
    --run_counts 3600 \
    --num_rounds 1 \
    --seed 2024 \
    --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim "$HIDDEN_DIM" --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix --max_length 64 \
    --inference_batch_size 64 \
    --bf16 \
    --collect_states_output "$OUT/collected_states.jsonl" \
    > "$OUT/collect.log" 2>&1

echo "collection done -> $OUT/collected_states.jsonl"
"$PY" - "$OUT/collected_states.jsonl" <<'EOF'
import json, sys, os
p = sys.argv[1]
snaps = {}
n = 0
for line in open(p):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line); n += 1
    s = r.get("snapshot_path", "")
    if s:
        snaps[s] = snaps.get(s, 0) + 1
print(f"records={n} distinct_snapshots={len(snaps)} intersections_per_snapshot={max(snaps.values()) if snaps else 0}")
EOF
