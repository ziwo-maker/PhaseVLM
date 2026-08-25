#!/usr/bin/env bash
# Zero-shot online eval of the Jinan source policy on the jinan_real test flow.
# 3 eval seeds in parallel on GPUs 0/1/2, then aggregate ATT/AQL/AWT.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814}/zero_shot"
CKPT="${SOURCE_CKPT:-$REPO/records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
PREFIX_TOKENS="${PREFIX_TOKENS:-32}"
LLM=/home/data/model/qwen-vl-8B
DATASET=jinan
TEST_FLOW=anon_3_4_jinan_real.json
SEEDS=(2024 42 123)
GPUS=(0 1 2)

mkdir -p "$OUT"
cd "$REPO"

pids=()
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    gpu=${GPUS[$i]}
    result_json="$OUT/result_seed${seed}.json"
    log="$OUT/eval_seed${seed}.log"
    if [ -f "$result_json" ]; then
        echo "skip seed $seed (result exists)"
        continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u projection_train/run_projection_online.py \
        --memo "jinan_real_zero_shot_seed${seed}" \
        --projection_ckpt "$CKPT" \
        --llm_path "$LLM" \
        --dataset "$DATASET" \
        --traffic_file "$TEST_FLOW" \
        --run_counts 3600 \
        --num_rounds 1 \
        --seed "$seed" \
        --prompt_mode queue_wait_short \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim "$HIDDEN_DIM" --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 \
        --bf16 \
        --result_json "$result_json" \
        > "$log" 2>&1 &
    pids+=("$!")
    echo "launched seed=$seed on GPU $gpu (pid ${pids[-1]})"
done

rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then rc=1; fi
done
if [ "$rc" -ne 0 ]; then
    echo "some zero-shot eval failed; see logs"
    exit 1
fi

ZERO_SHOT_DIR="$OUT" "$PY" - <<'EOF'
import json, os, statistics
out = os.environ.get("ZERO_SHOT_DIR") or "records/pipeline_runs/jinan_real_responsive_h120_20260814/zero_shot"
keymap = {"ATT": "test_avg_travel_time_over", "AQL": "test_avg_queue_len_over", "AWT": "test_avg_waiting_time_over"}
per = {}
vals = {m: [] for m in keymap}
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
    **{m: {"n": len(vals[m]), "mean": statistics.fmean(vals[m]), "std": statistics.pstdev(vals[m]) if len(vals[m]) > 1 else 0.0} for m in keymap},
}
json.dump(summary, open(os.path.join(out, "eval_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
EOF
echo "zero-shot done: $OUT/eval_summary.json"
