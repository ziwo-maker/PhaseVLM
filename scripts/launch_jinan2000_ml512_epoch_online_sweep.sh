#!/usr/bin/env bash
# Four-way same-flow online sweep for selected Stage-2 epochs.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan2000_simexpert_best_ml512_bs1ga16_20260906_110909}"
RUN_ID="${RUN_ID:-jinan2000_ml512_epoch_online_sweep_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_RUN/online_epoch_sweep/$RUN_ID}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
SEED="${SEED:-2024}"
ONLINE_TIMEOUT="${ONLINE_TIMEOUT:-14400}"

EPOCHS=(4 9 13 15)
GPUS=(0 1 2 3)
TRAFFIC_FILE="anon_3_4_jinan_real_2000.json"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_FILE="$RUN_ROOT/status.txt"
MANIFEST="$RUN_ROOT/manifest.json"
SUMMARY_JSON="$RUN_ROOT/summary.json"
SUMMARY_CSV="$RUN_ROOT/summary.csv"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_file() { [[ -s "$1" ]] || { log "Missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "Missing directory: $1" >&2; exit 1; }; }

write_manifest() {
  "$PYTHON_BIN" - "$MANIFEST" "$SOURCE_RUN" "$MODEL_PATH" "$RUN_COUNTS" "$SEED" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, source_run, model_path, run_counts, seed = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_run": source_run,
    "model_path": model_path,
    "dataset": "jinan",
    "traffic_file": "anon_3_4_jinan_real_2000.json",
    "evaluation_scope": "same-flow checkpoint selection",
    "run_counts": int(run_counts),
    "seed": int(seed),
    "checkpoints": [
        {"epoch": 4, "gpu": 0, "reason": "early offline improvement"},
        {"epoch": 9, "gpu": 1, "reason": "mid-course offline cost convergence"},
        {"epoch": 13, "gpu": 2, "reason": "lowest offline val_top1_cost_regret"},
        {"epoch": 15, "gpu": 3, "reason": "final highest offline validation accuracy and macro F1"},
    ],
    "architecture": {
        "seq_len": 16, "image_size": 224, "periodicity": 16,
        "input_dim": 8, "hidden_dim": 64, "prefix_tokens": 8,
        "prefix_mode": "prefix", "max_length": 512,
        "prompt_mode": "queue_wait_short",
    },
    "online": {"min_action_time": 30, "traffic_series_interval": 2, "inference_batch_size": 64, "snapshots": False},
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
PY
}

run_one() {
  local epoch="$1"
  local gpu="$2"
  local ckpt="$SOURCE_RUN/stage2/epoch_${epoch}.pt"
  local worker_dir="$RESULT_DIR/epoch_${epoch}/seed_${SEED}"
  local result_json="$worker_dir/result.json"
  local worker_status="$worker_dir/status.txt"
  local worker_log="$LOG_DIR/epoch_${epoch}_seed_${SEED}_gpu${gpu}.log"

  mkdir -p "$worker_dir"
  printf 'running epoch=%s gpu=%s started_at=%s\n' "$epoch" "$gpu" "$(date --iso-8601=seconds)" > "$worker_status"
  if timeout --foreground "$ONLINE_TIMEOUT" env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/run_projection_online.py \
      --memo "jinan2000_ml512_epoch_sweep/epoch_${epoch}/seed_${SEED}" \
      --projection_ckpt "$ckpt" --llm_path "$MODEL_PATH" \
      --dataset jinan --traffic_file "$TRAFFIC_FILE" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$SEED" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
      --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
      --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --disable_snapshots \
      --result_json "$result_json" > "$worker_log" 2>&1; then
    printf 'completed epoch=%s gpu=%s completed_at=%s result_json=%s\n' "$epoch" "$gpu" "$(date --iso-8601=seconds)" "$result_json" > "$worker_status"
  else
    local code=$?
    printf 'failed epoch=%s gpu=%s exit_code=%s failed_at=%s\n' "$epoch" "$gpu" "$code" "$(date --iso-8601=seconds)" > "$worker_status"
    return "$code"
  fi
}

write_summary() {
  "$PYTHON_BIN" - "$RESULT_DIR" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_dir, summary_path, csv_path = map(Path, sys.argv[1:])
rows = []
for result_path in sorted(result_dir.glob("epoch_*/seed_*/result.json")):
    epoch = int(result_path.parts[-3].split("_")[-1])
    seed = int(result_path.parts[-2].split("_")[-1])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("results") or {}
    if isinstance(result, list):
        result = result[-1] if result else {}
    rows.append({
        "epoch": epoch,
        "seed": seed,
        "result_path": str(result_path),
        "test_avg_queue_len_over": result.get("test_avg_queue_len_over"),
        "test_avg_travel_time_over": result.get("test_avg_travel_time_over"),
        "test_avg_waiting_time_over": result.get("test_avg_waiting_time_over"),
        "test_queuing_vehicle_num_over": result.get("test_queuing_vehicle_num_over"),
        "test_reward_over": result.get("test_reward_over"),
        "test_avg_queue_len_std": result.get("test_avg_queue_len_std"),
        "test_avg_waiting_time_std": result.get("test_avg_waiting_time_std"),
    })
summary_path.write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["epoch", "seed"])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"completed_results": len(rows), "summary": str(summary_path)}, ensure_ascii=False))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_dir "$MODEL_PATH"
  require_file "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"
  for epoch in "${EPOCHS[@]}"; do require_file "$SOURCE_RUN/stage2/epoch_${epoch}.pt"; done
  [[ ! -e "$RUN_ROOT" ]] || { log "Run root already exists: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_DIR" "$RESULT_DIR"
  write_manifest
  printf 'running started_at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"

  local index epoch gpu pid rc=0
  declare -a pids=()
  for index in "${!EPOCHS[@]}"; do
    epoch="${EPOCHS[$index]}"
    gpu="${GPUS[$index]}"
    log "Launching epoch ${epoch} on GPU ${gpu}"
    run_one "$epoch" "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  write_summary
  if [[ "$rc" -eq 0 ]]; then
    printf 'completed completed_at=%s summary_json=%s summary_csv=%s\n' "$(date --iso-8601=seconds)" "$SUMMARY_JSON" "$SUMMARY_CSV" > "$STATUS_FILE"
  else
    printf 'failed completed_at=%s summary_json=%s summary_csv=%s\n' "$(date --iso-8601=seconds)" "$SUMMARY_JSON" "$SUMMARY_CSV" > "$STATUS_FILE"
  fi
  return "$rc"
}

main "$@"
