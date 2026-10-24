#!/usr/bin/env bash
# Full same-target online sweep for the max_length=512 Jinan source training run.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/hangzhou5816_ref_stage1_stage2_4gpu_ml512_20260905_102007}"
STAGE2_DIR="${STAGE2_DIR:-$SOURCE_RUN/stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_RUN/online_hangzhou5816_epoch_sweep_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
CHECKPOINTS_RAW="${CHECKPOINTS:-epoch_3 epoch_6 epoch_9 epoch_12 epoch_15 best}"

DATASET=hangzhou
TRAFFIC_FILE=anon_4_4_hangzhou_real_5816.json
RUN_COUNTS=3600

require_file() { [[ -s "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }; }

run_task() {
  local checkpoint="$1" seed="$2" gpu="$3"
  local ckpt="$STAGE2_DIR/${checkpoint}.pt"
  local task_dir="$OUTPUT_DIR/$checkpoint/seed_$seed"
  local result="$task_dir/result.json"
  local log="$task_dir/online.log"
  local status="$task_dir/status.txt"
  mkdir -p "$task_dir"
  if timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "hangzhou5816_ml512_${checkpoint}_seed${seed}" \
    --projection_ckpt "$ckpt" --llm_path "$MODEL_PATH" \
    --dataset "$DATASET" --traffic_file "$TRAFFIC_FILE" \
    --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 \
    --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --result_json "$result" > "$log" 2>&1; then
    printf 'ok\n' > "$status"
  else
    rc=$?
    printf 'failed exit_code=%s\n' "$rc" > "$status"
  fi
}

write_summary() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" "${CHECKPOINTS[@]}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoints = sys.argv[2:]
keys = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
    "queued_vehicles": "test_queuing_vehicle_num_over",
}
summary = {"scope": "Hangzhou anon_4_4_hangzhou_real_5816.json, run_counts=3600", "checkpoints": {}}
for checkpoint in checkpoints:
    rows = []
    for path in sorted((root / checkpoint).glob("seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("results", payload)
        rows.append({"seed": int(path.parent.name.removeprefix("seed_")), **{name: result.get(key) for name, key in keys.items()}})
    aggregate = {}
    for name in keys:
        values = [float(row[name]) for row in rows if row[name] is not None]
        if values:
            aggregate[name] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "n": len(values)}
    summary["checkpoints"][checkpoint] = {"per_seed": rows, "aggregate": aggregate}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
for checkpoint, item in summary["checkpoints"].items():
    awt = item["aggregate"].get("AWT", {})
    print(f"{checkpoint}: AWT={awt.get('mean')} n={awt.get('n', 0)}")
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_dir "$MODEL_PATH"; require_dir "$STAGE2_DIR"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  read -r -a SEEDS <<< "$SEEDS_RAW"
  read -r -a CHECKPOINTS <<< "$CHECKPOINTS_RAW"
  (( ${#GPUS[@]} == 4 )) || { echo "Exactly four GPUs are required; got: $GPU_LIST" >&2; exit 1; }
  for checkpoint in "${CHECKPOINTS[@]}"; do require_file "$STAGE2_DIR/${checkpoint}.pt"; done
  mkdir -p "$OUTPUT_DIR"
  {
    printf 'source_run=%s\n' "$SOURCE_RUN"
    printf 'dataset=%s\ntraffic_file=%s\nrun_counts=%s\n' "$DATASET" "$TRAFFIC_FILE" "$RUN_COUNTS"
    printf 'checkpoints=%s\nseeds=%s\ngpus=%s\n' "$CHECKPOINTS_RAW" "$SEEDS_RAW" "$GPU_LIST"
    printf 'architecture=hidden_dim=256,prefix_tokens=32,max_length=512,prompt_mode=queue_wait_short\n'
  } > "$OUTPUT_DIR/manifest.txt"

  TASK_CHECKPOINTS=(); TASK_SEEDS=()
  for checkpoint in "${CHECKPOINTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TASK_CHECKPOINTS+=("$checkpoint"); TASK_SEEDS+=("$seed")
    done
  done
  printf 'checkpoint\tseed\tstatus\n' > "$OUTPUT_DIR/status.tsv"
  for ((start=0; start<${#TASK_CHECKPOINTS[@]}; start+=${#GPUS[@]})); do
    pids=(); labels=()
    for ((offset=0; offset<${#GPUS[@]} && start+offset<${#TASK_CHECKPOINTS[@]}; offset++)); do
      index=$((start + offset)); checkpoint="${TASK_CHECKPOINTS[$index]}"; seed="${TASK_SEEDS[$index]}"; gpu="${GPUS[$offset]}"
      run_task "$checkpoint" "$seed" "$gpu" &
      pids+=("$!"); labels+=("$checkpoint:$seed")
    done
    for ((offset=0; offset<${#pids[@]}; offset++)); do
      wait "${pids[$offset]}"
      checkpoint="${labels[$offset]%%:*}"; seed="${labels[$offset]##*:}"
      task_status=$(cat "$OUTPUT_DIR/$checkpoint/seed_$seed/status.txt")
      printf '%s\t%s\t%s\n' "$checkpoint" "$seed" "$task_status" >> "$OUTPUT_DIR/status.tsv"
    done
  done
  write_summary
}

main "$@"
