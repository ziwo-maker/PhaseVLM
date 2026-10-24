#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="/home/code/LLMSTSC"
PYTHON_BIN="/home/sysu/anaconda3/envs/hnr5090/bin/python"
CHECKPOINT="$PROJECT_ROOT/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642/stage2/checkpoints/epoch_15.pt"
MODEL_PATH="/home/data/model/qwen-vl-8B"
RUN_ID="${RUN_ID:-jinan3_8b_visual_signal_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/records/pipeline_runs/$RUN_ID}"
ARMS=(full zero_prefix shuffled_image constant_image)
SEEDS=(2024 42 123 7)
GPUS=(0 1 2 3)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results" "$RUN_ROOT/status"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$RUN_ID" "$CHECKPOINT" "$MODEL_PATH" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

output, run_id, checkpoint, model_path = sys.argv[1:]
payload = {
    "run_id": run_id,
    "created_at": datetime.now().astimezone().isoformat(),
    "checkpoint": checkpoint,
    "model_path": model_path,
    "dataset": "jinan",
    "traffic_file": "anon_3_4_jinan_real.json",
    "prompt_mode": "queue_wait_short",
    "visual_arms": ["full", "zero_prefix", "shuffled_image", "constant_image"],
    "seeds": [2024, 42, 123, 7],
    "gpu_assignment": {"2024": 0, "42": 1, "123": 2, "7": 3},
    "same_flow_evaluation": True,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

run_one() {
  local gpu="$1"
  local seed="$2"
  local arm="$3"
  local output_dir="$RUN_ROOT/results/$arm/seed_$seed"
  local result_json="$output_dir/result.json"
  local log_path="$RUN_ROOT/logs/${arm}_seed_${seed}_gpu_${gpu}.log"
  local status_path="$RUN_ROOT/status/${arm}_seed_${seed}.status"

  mkdir -p "$output_dir"
  printf 'state=running\ngpu=%s\nseed=%s\narm=%s\nstarted_at=%s\n' \
    "$gpu" "$seed" "$arm" "$(date --iso-8601=seconds)" > "$status_path"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" projection_train/run_projection_online.py \
    --memo "online_tests/$RUN_ID/$arm/seed_$seed" \
    --projection_ckpt "$CHECKPOINT" \
    --llm_path "$MODEL_PATH" \
    --num_rounds 1 \
    --proj_name "Jinan3-8B-visual-ablation-$arm" \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real.json \
    --run_counts 3600 \
    --device cuda:0 \
    --seed "$seed" \
    --prompt_mode queue_wait_short \
    --seq_len 16 \
    --image_size 224 \
    --periodicity 16 \
    --input_dim 8 \
    --hidden_dim 64 \
    --prefix_tokens 8 \
    --prefix_mode prefix \
    --max_length 64 \
    --min_action_time 30 \
    --traffic_series_interval 2 \
    --inference_batch_size 0 \
    --visual_ablation "$arm" \
    --result_json "$result_json" \
    --bf16 \
    --deterministic \
    --disable_snapshots \
    > "$log_path" 2>&1
  local exit_code=$?

  printf 'state=%s\nexit_status=%s\ngpu=%s\nseed=%s\narm=%s\ncompleted_at=%s\nresult_json=%s\nlog=%s\n' \
    "$([[ $exit_code -eq 0 ]] && printf completed || printf failed)" \
    "$exit_code" "$gpu" "$seed" "$arm" "$(date --iso-8601=seconds)" \
    "$result_json" "$log_path" > "$status_path"
  return "$exit_code"
}

worker() {
  local gpu="$1"
  local seed="$2"
  local failures=0
  local arm
  for arm in "${ARMS[@]}"; do
    run_one "$gpu" "$seed" "$arm" || failures=$((failures + 1))
  done
  return "$failures"
}

cd "$PROJECT_ROOT" || exit 2
pids=()
for index in "${!GPUS[@]}"; do
  worker "${GPUS[$index]}" "${SEEDS[$index]}" &
  pids+=("$!")
  printf 'gpu=%s seed=%s worker_pid=%s\n' "${GPUS[$index]}" "${SEEDS[$index]}" "$!"
done
printf '%s\n' "${pids[@]}" > "$RUN_ROOT/worker_pids.txt"

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

"$PYTHON_BIN" scripts/aggregate_visual_signal_ablation.py \
  --run_root "$RUN_ROOT" \
  --expected_seeds "${SEEDS[@]}"
aggregate_status=$?

printf 'workers_failed=%s\naggregate_status=%s\ncompleted_at=%s\n' \
  "$failures" "$aggregate_status" "$(date --iso-8601=seconds)" > "$RUN_ROOT/completion.status"
if [[ $failures -ne 0 || $aggregate_status -ne 0 ]]; then
  exit 1
fi
