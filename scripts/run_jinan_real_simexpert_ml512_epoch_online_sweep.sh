#!/usr/bin/env bash
# Same-flow online comparison of selected Stage-2 checkpoints from Jinan-real.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan_real_simexpert_h60_att1_aql1_awt1_spill0_stage12_ml512_ga16_20260908_001305}"
RUN_ID="${RUN_ID:-jinan_real_simexpert_h60_ml512_epoch_online_sweep_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_RUN/online_epoch_sweep/$RUN_ID}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
SEED="${SEED:-2024}"
ONLINE_TIMEOUT="${ONLINE_TIMEOUT:-14400}"

TRAFFIC_FILE="anon_3_4_jinan_real.json"
STAGE2_DIR="$SOURCE_RUN/stage2"
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
    "traffic_file": "anon_3_4_jinan_real.json",
    "evaluation_scope": "same-flow multi-checkpoint online selection",
    "run_counts": int(run_counts),
    "seed": int(seed),
    "selection_metric": "test_avg_waiting_time_over (lower is better; run-local single-seed comparison)",
    "checkpoints": [
        {"checkpoint": "epoch_3", "gpu": 1, "reason": "early converged checkpoint"},
        {"checkpoint": "epoch_6", "gpu": 2, "reason": "mid-course checkpoint"},
        {"checkpoint": "epoch_9", "gpu": 3, "reason": "late-course checkpoint"},
        {"checkpoint": "epoch_12", "gpu": 2, "reason": "lowest offline validation cost regret"},
        {"checkpoint": "epoch_15", "gpu": 1, "reason": "final checkpoint"},
    ],
    "architecture": {
        "seq_len": 16, "image_size": 224, "periodicity": 16,
        "input_dim": 8, "hidden_dim": 256, "prefix_tokens": 32,
        "prefix_mode": "prefix", "max_length": 512,
        "prompt_mode": "queue_wait_short",
    },
    "online": {
        "min_action_time": 30, "traffic_series_interval": 2,
        "inference_batch_size": 64, "snapshots": False,
        "phase_override": "disabled",
    },
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

run_one() {
  local checkpoint="$1"
  local gpu="$2"
  local ckpt="$STAGE2_DIR/${checkpoint}.pt"
  local worker_dir="$RESULT_DIR/$checkpoint/seed_${SEED}"
  local result_json="$worker_dir/result.json"
  local worker_status="$worker_dir/status.txt"
  local worker_log="$LOG_DIR/${checkpoint}_seed_${SEED}_gpu${gpu}.log"

  mkdir -p "$worker_dir"
  printf 'running checkpoint=%s seed=%s gpu=%s started_at=%s\n' "$checkpoint" "$SEED" "$gpu" "$(date --iso-8601=seconds)" > "$worker_status"
  if timeout --foreground "$ONLINE_TIMEOUT" env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/run_projection_online.py \
      --memo "jinan_real_simexpert_h60_ml512_epoch_sweep/${checkpoint}/seed_${SEED}" \
      --projection_ckpt "$ckpt" --llm_path "$MODEL_PATH" \
      --dataset jinan --traffic_file "$TRAFFIC_FILE" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$SEED" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
      --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
      --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --disable_snapshots \
      --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
      --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
      --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
      --result_json "$result_json" > "$worker_log" 2>&1; then
    printf 'completed checkpoint=%s seed=%s gpu=%s completed_at=%s result_json=%s\n' "$checkpoint" "$SEED" "$gpu" "$(date --iso-8601=seconds)" "$result_json" > "$worker_status"
  else
    local code=$?
    printf 'failed checkpoint=%s seed=%s gpu=%s exit_code=%s failed_at=%s\n' "$checkpoint" "$SEED" "$gpu" "$code" "$(date --iso-8601=seconds)" > "$worker_status"
    return "$code"
  fi
}

run_worker() {
  local gpu="$1"
  shift
  local checkpoint
  local rc=0
  for checkpoint in "$@"; do
    log "Launching ${checkpoint} on GPU ${gpu}"
    run_one "$checkpoint" "$gpu" || rc=1
  done
  return "$rc"
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
    checkpoint = result_path.parts[-3]
    seed = int(result_path.parts[-2].removeprefix("seed_"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("results", payload)
    if isinstance(result, list):
        result = result[-1] if result else {}
    rows.append({
        "checkpoint": checkpoint,
        "seed": seed,
        "result_path": str(result_path),
        "ATT": result.get("test_avg_travel_time_over"),
        "AQL": result.get("test_avg_queue_len_over"),
        "AWT": result.get("test_avg_waiting_time_over"),
        "queued_vehicles": result.get("test_queuing_vehicle_num_over"),
        "reward": result.get("test_reward_over"),
    })
rows.sort(key=lambda row: int(row["checkpoint"].removeprefix("epoch_")))
ranked = sorted(
    (row for row in rows if row["AWT"] is not None),
    key=lambda row: (float(row["AWT"]), float(row["AQL"] or float("inf")), float(row["ATT"] or float("inf"))),
)
summary = {
    "selection_metric": "AWT ascending, then AQL and ATT as tie-breakers",
    "results": rows,
    "run_local_winner": ranked[0] if ranked else None,
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
fields = ["checkpoint", "seed", "ATT", "AQL", "AWT", "queued_vehicles", "reward", "result_path"]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"completed_results": len(rows), "run_local_winner": summary["run_local_winner"]}, ensure_ascii=False))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_dir "$MODEL_PATH"
  require_file "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"
  for checkpoint in epoch_3 epoch_6 epoch_9 epoch_12 epoch_15; do
    require_file "$STAGE2_DIR/${checkpoint}.pt"
  done
  [[ ! -e "$RUN_ROOT" ]] || { log "Run root already exists: $RUN_ROOT" >&2; exit 1; }

  mkdir -p "$LOG_DIR" "$RESULT_DIR"
  write_manifest
  printf 'running started_at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"

  run_worker 1 epoch_3 epoch_15 & p1=$!
  run_worker 2 epoch_6 epoch_12 & p2=$!
  run_worker 3 epoch_9 & p3=$!
  rc=0
  wait "$p1" || rc=1
  wait "$p2" || rc=1
  wait "$p3" || rc=1
  write_summary
  if [[ "$rc" -eq 0 ]]; then
    printf 'completed completed_at=%s summary_json=%s summary_csv=%s\n' "$(date --iso-8601=seconds)" "$SUMMARY_JSON" "$SUMMARY_CSV" > "$STATUS_FILE"
  else
    printf 'failed completed_at=%s summary_json=%s summary_csv=%s\n' "$(date --iso-8601=seconds)" "$SUMMARY_JSON" "$SUMMARY_CSV" > "$STATUS_FILE"
  fi
  return "$rc"
}

main "$@"
