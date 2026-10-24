#!/usr/bin/env bash
# Same-flow Hangzhou-2 online evaluation for selected Stage-2 checkpoints.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
MODEL_PATH=/mnt/data/Qwen3-VL-8B-Instruct
SOURCE_RUN=${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/hangzhou2_paper_3gpu_20260925_144836}
STAGE2_DIR=${STAGE2_DIR:-$SOURCE_RUN/stage2}
OUTPUT_DIR=${OUTPUT_DIR:-$SOURCE_RUN/online_hangzhou2_epochs_$(date +%Y%m%d_%H%M%S)}
GPU_LIST=${GPU_LIST:-0,2,3}
CHECKPOINTS_RAW=${CHECKPOINTS:-epoch_3 epoch_4 epoch_6 epoch_9 epoch_12 epoch_15}
SEED=${SEED:-2024}
DATASET=hangzhou
TRAFFIC_FILE=anon_4_4_hangzhou_real_5816.json
RUN_COUNTS=3600

require_file() { [[ -s "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { printf 'Missing directory: %s\n' "$1" >&2; exit 1; }; }

run_task() {
  local checkpoint="$1" gpu="$2"
  local task_dir="$OUTPUT_DIR/$checkpoint/seed_$SEED"
  mkdir -p "$task_dir"
  if timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
    "$REPO_ROOT/projection_train/run_projection_online.py" \
    --memo "hangzhou2_epoch_sweep_${checkpoint}_seed${SEED}" \
    --projection_ckpt "$STAGE2_DIR/${checkpoint}.pt" --llm_path "$MODEL_PATH" \
    --dataset "$DATASET" --traffic_file "$TRAFFIC_FILE" \
    --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$SEED" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 256 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 \
    --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --result_json "$task_dir/result.json" > "$task_dir/online.log" 2>&1; then
    printf 'ok\n' > "$task_dir/status.txt"
  else
    rc=$?
    printf 'failed exit_code=%s\n' "$rc" > "$task_dir/status.txt"
  fi
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_dir "$MODEL_PATH"
  require_dir "$STAGE2_DIR"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  read -r -a CHECKPOINTS <<< "$CHECKPOINTS_RAW"
  (( ${#GPUS[@]} == 3 )) || { printf 'Exactly 3 GPUs are required; got %s\n' "$GPU_LIST" >&2; exit 1; }
  for checkpoint in "${CHECKPOINTS[@]}"; do require_file "$STAGE2_DIR/${checkpoint}.pt"; done

  mkdir -p "$OUTPUT_DIR"
  {
    printf 'source_run=%s\nsource_scenario=Hangzhou-2\n' "$SOURCE_RUN"
    printf 'dataset=%s\ntraffic_file=%s\nrun_counts=%s\n' "$DATASET" "$TRAFFIC_FILE" "$RUN_COUNTS"
    printf 'checkpoints=%s\nseed=%s\ngpus=%s\n' "$CHECKPOINTS_RAW" "$SEED" "$GPU_LIST"
    printf 'architecture=hidden_dim=256,prefix_tokens=8,max_length=512,prompt_mode=queue_wait_short\n'
    printf 'evaluation_scope=same-flow Hangzhou-2; one CityFlow round per checkpoint\n'
  } > "$OUTPUT_DIR/manifest.txt"
  printf 'checkpoint\tseed\tgpu\tstatus\n' > "$OUTPUT_DIR/status.tsv"
  printf 'running\n' > "$OUTPUT_DIR/status.txt"

  local failed=0
  for ((start=0; start<${#CHECKPOINTS[@]}; start+=${#GPUS[@]})); do
    pids=(); labels=(); assigned_gpus=()
    for ((offset=0; offset<${#GPUS[@]} && start+offset<${#CHECKPOINTS[@]}; offset++)); do
      checkpoint=${CHECKPOINTS[$((start + offset))]}
      gpu=${GPUS[$offset]}
      run_task "$checkpoint" "$gpu" &
      pids+=("$!"); labels+=("$checkpoint"); assigned_gpus+=("$gpu")
    done
    for ((offset=0; offset<${#pids[@]}; offset++)); do
      wait "${pids[$offset]}" || true
      task_status=$(<"$OUTPUT_DIR/${labels[$offset]}/seed_$SEED/status.txt")
      printf '%s\t%s\t%s\t%s\n' "${labels[$offset]}" "$SEED" "${assigned_gpus[$offset]}" "$task_status" >> "$OUTPUT_DIR/status.tsv"
      [[ "$task_status" == ok ]] || failed=1
    done
  done

  "$PYTHON_BIN" - "$OUTPUT_DIR" "$SEED" "${CHECKPOINTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = sys.argv[2]
summary = {"scope": f"Hangzhou-2 same-flow, anon_4_4_hangzhou_real_5816.json, run_counts=3600, seed={seed}", "checkpoints": {}}
keys = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
    "queued_vehicles": "test_queuing_vehicle_num_over",
}
for checkpoint in sys.argv[3:]:
    result_path = root / checkpoint / f"seed_{seed}" / "result.json"
    item = {"result_path": str(result_path), "status": "missing"}
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload.get("results", payload)
        item["status"] = "ok"
        item.update({name: result.get(key) for name, key in keys.items()})
    summary["checkpoints"][checkpoint] = item
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

  if (( failed )); then
    printf 'completed_with_failures\n' > "$OUTPUT_DIR/status.txt"
    return 1
  fi
  printf 'completed\n' > "$OUTPUT_DIR/status.txt"
}

main "$@"
