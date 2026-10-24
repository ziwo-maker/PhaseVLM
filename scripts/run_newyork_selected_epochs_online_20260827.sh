#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/restart_newyork28x7_qwshort_targeted_stage2_4gpu_20260827_104004}"
OUT_ROOT="${OUT_ROOT:-$SOURCE_RUN/online_selected_epochs_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-10 11 12 14}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
EVAL_SEED="${EVAL_SEED:-2024}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_28_7_newyork_real_double.json}"
RUN_COUNTS="${RUN_COUNTS:-3600}"

LOG_DIR="$OUT_ROOT/logs"
RESULT_DIR="$OUT_ROOT/results"
STATUS_TSV="$OUT_ROOT/status.tsv"
COMPLETION_STATUS="$OUT_ROOT/completion.status"

summarize_results() {
  "$PYTHON" - "$OUT_ROOT" "$SOURCE_RUN" <<'PY'
import csv
import json
import pathlib
import sys

out_root = pathlib.Path(sys.argv[1])
source_run = pathlib.Path(sys.argv[2])
rows = []
for result_path in sorted((out_root / "results").glob("epoch_*/result.json")):
    payload = json.load(open(result_path, encoding="utf-8"))
    metrics = dict(payload.get("results") or {})
    metrics["epoch"] = int(result_path.parent.name.split("_")[-1])
    metrics["path"] = str(result_path)
    metrics["projection_ckpt"] = str((payload.get("args") or {}).get("projection_ckpt", ""))
    metrics["sum_queue_wait_travel"] = sum(
        float(metrics[key])
        for key in (
            "test_avg_queue_len_over",
            "test_avg_waiting_time_over",
            "test_avg_travel_time_over",
        )
    )
    rows.append(metrics)

epoch15_path = source_run / "online_eval" / "seed_2024" / "anon_28_7_newyork_real_double" / "result.json"
reference = None
if epoch15_path.is_file():
    payload = json.load(open(epoch15_path, encoding="utf-8"))
    reference = dict(payload.get("results") or {})
    reference["epoch"] = 15
    reference["path"] = str(epoch15_path)
    reference["projection_ckpt"] = str((payload.get("args") or {}).get("projection_ckpt", ""))
    reference["sum_queue_wait_travel"] = sum(
        float(reference[key])
        for key in (
            "test_avg_queue_len_over",
            "test_avg_waiting_time_over",
            "test_avg_travel_time_over",
        )
    )

summary = {
    "protocol": {
        "dataset": "newyork_28x7",
        "traffic_file": "anon_28_7_newyork_real_double.json",
        "run_counts": 3600,
        "seed": 2024,
        "deterministic": False,
    },
    "results": sorted(rows, key=lambda row: row["epoch"]),
    "existing_epoch15_reference": reference,
}
(out_root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

fieldnames = [
    "epoch",
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "sum_queue_wait_travel",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
    "projection_ckpt",
    "path",
]
with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(rows + ([reference] if reference else []), key=lambda item: item["epoch"]):
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY
}

finish() {
  local status="$?"
  trap - EXIT
  summarize_results > "$LOG_DIR/summary.log" 2>&1 || true
  {
    printf 'exit_status=%s\n' "$status"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'out_root=%s\n' "$OUT_ROOT"
    printf 'summary_json=%s\n' "$OUT_ROOT/summary.json"
    printf 'summary_csv=%s\n' "$OUT_ROOT/summary.csv"
  } > "$COMPLETION_STATUS"
  exit "$status"
}

trap finish EXIT

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$REPO_ROOT"

read -r -a epochs <<< "$EPOCHS"
read -r -a gpus <<< "$GPU_LIST"
if [[ "${#epochs[@]}" -ne "${#gpus[@]}" ]]; then
  echo "EPOCHS and GPU_LIST must contain the same number of entries." >&2
  exit 1
fi

for required in "$PYTHON" "$MODEL_PATH" "$REPO_ROOT/data/NewYork/28_7/$TRAFFIC_FILE"; do
  [[ -e "$required" ]] || { echo "Required path is missing: $required" >&2; exit 1; }
done
for epoch in "${epochs[@]}"; do
  checkpoint="$SOURCE_RUN/stage2/checkpoints/epoch_${epoch}.pt"
  [[ -s "$checkpoint" ]] || { echo "Checkpoint is missing: $checkpoint" >&2; exit 1; }
done

printf 'epoch\tgpu\tpid\tstatus\tstarted_at\tfinished_at\tresult_json\tlog\n' > "$STATUS_TSV"
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_run=%s\n' "$SOURCE_RUN"
  printf 'out_root=%s\n' "$OUT_ROOT"
  printf 'epochs=%s\n' "$EPOCHS"
  printf 'gpus=%s\n' "$GPU_LIST"
  printf 'training_flow=%s\n' "$TRAFFIC_FILE"
  printf 'evaluation_flow=%s\n' "$TRAFFIC_FILE"
  printf 'flow_protocol=same_newyork_single_flow\n'
  printf 'run_counts=%s\n' "$RUN_COUNTS"
  printf 'eval_seed=%s\n' "$EVAL_SEED"
  printf 'deterministic=false\n'
} > "$OUT_ROOT/protocol.info"

pids=()
status=0
for index in "${!epochs[@]}"; do
  epoch="${epochs[$index]}"
  gpu="${gpus[$index]}"
  checkpoint="$SOURCE_RUN/stage2/checkpoints/epoch_${epoch}.pt"
  worker_dir="$RESULT_DIR/epoch_${epoch}"
  result_json="$worker_dir/result.json"
  worker_status="$worker_dir/status.txt"
  log_path="$LOG_DIR/epoch_${epoch}.log"
  mkdir -p "$worker_dir"

  (
    worker_exit=0
    {
      printf 'status=running\n'
      printf 'epoch=%s\n' "$epoch"
      printf 'gpu=%s\n' "$gpu"
      printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
      printf 'checkpoint=%s\n' "$checkpoint"
      printf 'checkpoint_sha256=%s\n' "$(sha256sum "$checkpoint" | awk '{print $1}')"
      printf 'result_json=%s\n' "$result_json"
    } > "$worker_status"

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON" projection_train/run_projection_online.py \
        --memo "online_tests/newyork28x7_selected_epochs_20260827/epoch_${epoch}/seed_${EVAL_SEED}" \
        --projection_ckpt "$checkpoint" \
        --llm_path "$MODEL_PATH" \
        --num_rounds 1 \
        --proj_name "NewYork28x7-qwshort-targeted-epoch${epoch}-online" \
        --dataset newyork_28x7 \
        --traffic_file "$TRAFFIC_FILE" \
        --run_counts "$RUN_COUNTS" \
        --seed "$EVAL_SEED" \
        --prompt_mode queue_wait_short \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --traffic_series_interval 2 \
        --inference_batch_size 64 \
        --bf16 \
        --result_json "$result_json" || worker_exit="$?"

    {
      printf 'status=%s\n' "$([[ "$worker_exit" -eq 0 ]] && echo success || echo failed)"
      printf 'exit_status=%s\n' "$worker_exit"
      printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    } >> "$worker_status"
    exit "$worker_exit"
  ) > "$log_path" 2>&1 &

  pid="$!"
  pids+=("$pid")
  printf '%s\t%s\t%s\trunning\t%s\t\t%s\t%s\n' \
    "$epoch" "$gpu" "$pid" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$STATUS_TSV"
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

summarize_results
exit "$status"
