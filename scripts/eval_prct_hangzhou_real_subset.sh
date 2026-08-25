#!/usr/bin/env bash
# Re-evaluate representative completed PRCT checkpoints on Hangzhou-real.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
SOURCE_RUN=${SOURCE_RUN:-$REPO/records/pipeline_runs/prct_hangzhou_matched_h120_20260818_212203}
OUT=${OUTPUT_DIR:-$SOURCE_RUN/hangzhou_real_subset_$(date +%Y%m%d_%H%M%S)}
FLOW=anon_4_4_hangzhou_real.json
TRAIN_SEEDS=(2024 11 123)
EVAL_SEEDS=(2024 42 123)

if [ "${1:-}" = "--help" ]; then
    echo "Usage: OUTPUT_DIR=<fresh-output-dir> bash scripts/eval_prct_hangzhou_real_subset.sh"
    exit 0
fi

if [ -e "$OUT" ]; then
    echo "Refusing to reuse output directory: $OUT" >&2
    exit 2
fi
test -f "$REPO/data/Hangzhou/4_4/$FLOW"
test -d "$SOURCE_RUN/train_frozen_neighbor_h120"
test -d "$SOURCE_RUN/train_prct_h120"
mkdir -p "$OUT/logs"
exec > "$OUT/launch.log" 2>&1
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" - "$OUT/config.json" "$SOURCE_RUN" "$FLOW" <<'PY'
import json
import sys

output, source_run, flow = sys.argv[1:]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "study": "PRCT matched frozen-vs-responsive Hangzhou-real subset re-evaluation",
            "source_run": source_run,
            "traffic_file": flow,
            "K": 118,
            "training_seeds": [2024, 11, 123],
            "eval_seeds": [2024, 42, 123],
            "selection_rationale": {
                "2024": "PRCT was worse on all three prior 5816 metrics",
                "11": "PRCT was better on all three prior 5816 metrics",
                "123": "PRCT improved ATT/AQL but worsened AWT on prior 5816",
            },
        },
        handle,
        indent=2,
    )
PY

run_model() {
    local arm=$1 train_seed=$2 checkpoint arm_out
    checkpoint="$SOURCE_RUN/$arm/c30/k118/tr$train_seed/checkpoints/best.pt"
    arm_out="$OUT/$arm/k118/tr$train_seed/evaluation"
    test -s "$checkpoint"
    mkdir -p "$arm_out"

    echo "[$(date -Is)] evaluate arm=$arm K=118 training_seed=$train_seed flow=$FLOW"
    local pids=()
    local gpu=0
    for eval_seed in "${EVAL_SEEDS[@]}"; do
        local result_json="$arm_out/result_seed${eval_seed}.json"
        local log_file="$OUT/logs/${arm}_k118_tr${train_seed}_eval${eval_seed}.log"
        CUDA_VISIBLE_DEVICES=$gpu "$PY" -u projection_train/run_projection_online.py \
            --memo "prct_hzreal_${arm}_k118_tr${train_seed}_s${eval_seed}" \
            --projection_ckpt "$checkpoint" --llm_path /home/data/model/qwen-vl-8B \
            --dataset hangzhou --traffic_file "$FLOW" --run_counts 3600 --num_rounds 1 \
            --seed "$eval_seed" --prompt_mode queue_wait_short --inference_batch_size 64 --bf16 \
            --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
            --prefix_tokens 32 --prefix_mode prefix --max_length 64 --result_json "$result_json" \
            > "$log_file" 2>&1 &
        pids+=("$!")
        gpu=$((gpu + 1))
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    for eval_seed in "${EVAL_SEEDS[@]}"; do
        test -s "$arm_out/result_seed${eval_seed}.json"
    done
}

for train_seed in "${TRAIN_SEEDS[@]}"; do
    run_model train_frozen_neighbor_h120 "$train_seed"
    run_model train_prct_h120 "$train_seed"
done

"$PY" scripts/feishu_push.py \
    "PRCT Hangzhou-real subset re-evaluation completed: paired K=118 checkpoints for training seeds 2024, 11, 123; all used $FLOW. Output: $OUT"
echo "[$(date -Is)] Hangzhou-real subset re-evaluation completed: $OUT"
