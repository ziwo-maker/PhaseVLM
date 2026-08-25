#!/usr/bin/env bash
# Policy-responsive H120 relabeling of the collected jinan snapshots.
# Mirrors launch_phase7_policy_responsive.sh (Hangzhou) with jinan grid/dataset.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814}"
INPUT="$OUT/collect/collected_states.jsonl"
ROADNET="$REPO/data/Jinan/3_4/roadnet_3_4.json"
TRAFFIC="${TRAFFIC_FILE:-$REPO/data/Jinan/3_4/anon_3_4_jinan_real_2000.json}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt}"
GPU="${GPU:-0}"
HORIZON="${HORIZON:-120}"

cd "$REPO"
mkdir -p "$OUT"

label="$OUT/relabeled_policy_responsive_h${HORIZON}.jsonl"
runtime="$OUT/runtime_h${HORIZON}"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/relabel_policy_responsive.py \
    --input "$INPUT" \
    --output "$label" \
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
    --inference_batch_size 16 \
    --bf16 \
    > "$OUT/relabel_h${HORIZON}.log" 2>&1

echo "relabel done -> $label"
cat "${label%.jsonl}_coverage.json" 2>/dev/null || true
