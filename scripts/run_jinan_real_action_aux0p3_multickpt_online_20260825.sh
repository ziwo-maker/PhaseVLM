#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/data/model/qwen-vl-8B}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_real_action_aux_fine_sweep_queue_short_20260825_130457/compact_action_aux0p3/stage2/checkpoints}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_real_action_aux0p3_multickpt_online_20260825_$(date +%H%M%S)}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_3_4_jinan_real.json}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EVAL_SEED="${EVAL_SEED:-2024}"
GPU_LIST="${GPU_LIST:-0 1 2}"
CKPTS="${CKPTS:-epoch_6.pt epoch_8.pt epoch_10.pt epoch_13.pt epoch_14.pt epoch_15.pt}"

LOG_DIR="$OUT_ROOT/logs"
RESULT_DIR="$OUT_ROOT/results"
STATUS_TSV="$OUT_ROOT/status.tsv"

notify_feishu() {
  local title="$1"
  local body="$2"
  local template="${3:-blue}"
  local config
  local parsed
  local webhook

  for config in "${FEISHU_CONFIG:-}" "/root/.claude/feishu.json" "$HOME/.codex/feishu.json"; do
    [[ -n "$config" && -r "$config" ]] && break
  done
  [[ -r "$config" ]] || return 0
  parsed="$("$PYTHON" - "$config" <<'PY' 2>/dev/null || true
import json, sys
path = sys.argv[1]
try:
    cfg = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(0)
mode = str(cfg.get("mode", cfg.get("notify_mode", "on")) or "on").lower()
webhook = str(cfg.get("webhook_url", cfg.get("webhook", "")) or "")
if mode in {"off", "none", "disabled", "false", "0"} or not webhook:
    raise SystemExit(0)
print(webhook)
PY
)"
  webhook="$(printf '%s\n' "$parsed" | sed -n '1p')"
  [[ -n "$webhook" ]] || return 0

  "$PYTHON" - "$title" "$body" "$template" <<'PY' | \
    curl --silent --show-error --connect-timeout 5 --max-time 10 \
      --output /dev/null --request POST "$webhook" \
      --header 'Content-Type: application/json' --data @- || true
import json, sys
_, title, body, template = sys.argv
print(json.dumps({
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": [{"tag": "markdown", "content": body}],
    },
}, ensure_ascii=False))
PY
}

summarize_results() {
  "$PYTHON" - "$OUT_ROOT" "$OUT_ROOT/summary.json" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

out_root = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = []
for result_path in sorted((out_root / "results").glob("*/result.json")):
    ckpt = result_path.parent.name
    item = {"ckpt": ckpt, "path": str(result_path)}
    try:
        payload = json.load(open(result_path, encoding="utf-8"))
        results = payload.get("results") or {}
        metrics = results[-1] if isinstance(results, list) and results else results
        item.update(metrics if isinstance(metrics, dict) else {})
        item["work_dir"] = payload.get("work_dir", "")
    except Exception as exc:
        item["error"] = str(exc)
    rows.append(item)

summary = {"out_root": str(out_root), "results": rows}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

metric_keys = [
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_avg_queue_len_over",
    "test_queuing_vehicle_num_over",
    "test_reward_over",
]
csv_path = out_root / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["ckpt", *metric_keys, "path", "work_dir", "error"])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

on_exit() {
  local status="$?"
  local template="red"
  [[ "$status" -eq 0 ]] && template="green"
  summarize_results > "$LOG_DIR/summary.log" 2>&1 || true
  notify_feishu \
    "Jinan Real action_aux=0.3 multi-checkpoint online finished" \
    "Exit status: $status\nRun: $OUT_ROOT\nSummary: $OUT_ROOT/summary.csv" \
    "$template"
  trap - EXIT
  exit "$status"
}

trap on_exit EXIT

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$REPO_ROOT"

printf 'ckpt\tgpu\tpid\tstatus\tstarted_at\tfinished_at\tresult_json\tlog\n' > "$STATUS_TSV"
{
  printf 'out_root=%s\n' "$OUT_ROOT"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repo_root=%s\n' "$REPO_ROOT"
  printf 'python=%s\n' "$PYTHON"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'ckpt_root=%s\n' "$CKPT_ROOT"
  printf 'ckpts=%s\n' "$CKPTS"
  printf 'gpu_list=%s\n' "$GPU_LIST"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'run_counts=%s\n' "$RUN_COUNTS"
  printf 'eval_seed=%s\n' "$EVAL_SEED"
} | tee "$LOG_DIR/run_config.txt"

for required_path in "$PYTHON" "$MODEL_PATH" "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done

read -r -a gpus <<< "$GPU_LIST"
if [[ "${#gpus[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty." >&2
  exit 1
fi

pids=()
slots=()
status=0
slot=0
for ckpt_name in $CKPTS; do
  ckpt_path="$CKPT_ROOT/$ckpt_name"
  if [[ ! -s "$ckpt_path" ]]; then
    echo "Missing checkpoint: $ckpt_path" >&2
    status=1
    continue
  fi

  gpu="${gpus[$slot]}"
  ckpt_id="${ckpt_name%.pt}"
  ckpt_result_dir="$RESULT_DIR/$ckpt_id"
  result_json="$ckpt_result_dir/result.json"
  worker_status="$ckpt_result_dir/status.tsv"
  log_path="$LOG_DIR/${ckpt_id}.log"
  mkdir -p "$ckpt_result_dir"

  (
    trap - EXIT
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    printf 'ckpt\tgpu\tstatus\tstarted_at\tfinished_at\tresult_json\tlog\n' > "$worker_status"
    printf '%s\t%s\trunning\t%s\t\t%s\t%s\n' "$ckpt_name" "$gpu" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$worker_status"
    worker_exit=0
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "jinan_real_action_aux0p3_multickpt_${ckpt_id}_seed${EVAL_SEED}" \
      --projection_ckpt "$ckpt_path" \
      --llm_path "$MODEL_PATH" \
      --dataset jinan \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --seed "$EVAL_SEED" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --bf16 \
      --result_json "$result_json" || worker_exit="$?"
    printf '%s\t%s\t%s\t\t%s\t%s\t%s\n' "$ckpt_name" "$gpu" "$([[ "$worker_exit" -eq 0 ]] && echo done || echo failed)" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$worker_status"
    exit "$worker_exit"
  ) > "$log_path" 2>&1 &
  pid="$!"
  pids+=("$pid")
  slots+=("$slot")
  printf '%s\t%s\t%s\trunning\t%s\t\t%s\t%s\n' "$ckpt_name" "$gpu" "$pid" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$STATUS_TSV"

  slot=$((slot + 1))
  if [[ "$slot" -ge "${#gpus[@]}" ]]; then
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        status=1
      fi
    done
    pids=()
    slots=()
    slot=0
  fi
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    status=1
  fi
done

summarize_results | tee "$LOG_DIR/summary.log"
exit "$status"
