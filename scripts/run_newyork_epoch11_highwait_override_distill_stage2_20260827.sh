#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/restart_newyork28x7_qwshort_targeted_stage2_4gpu_20260827_104004}"
BASE_DATA="${BASE_DATA:-/mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857/raw/traffic_prompt_series_2s.jsonl}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_28_7_newyork_real_double.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/newyork28x7_epoch11_highwait_override_distill_stage2_$(date +%Y%m%d_%H%M%S)}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
COLLECT_GPUS="${COLLECT_GPUS:-0 1 2 3}"
COLLECT_SEEDS="${COLLECT_SEEDS:-2024 2025 2026 2027}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3}"
EVAL_SEED="${EVAL_SEED:-2024}"
RUN_COUNTS="${RUN_COUNTS:-3600}"

OVERRIDE_GAP="${OVERRIDE_GAP:-2}"
LEFT_WAIT_PROTECT="${LEFT_WAIT_PROTECT:-60}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"
STAGE2_LR="${STAGE2_LR:-3e-5}"
STAGE2_MIN_LR="${STAGE2_MIN_LR:-3e-6}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"

STAGE1_CKPT="${STAGE1_CKPT:-$SOURCE_RUN/stage1/best.pt}"
INIT_CKPT="${INIT_CKPT:-$SOURCE_RUN/stage2/checkpoints/epoch_11.pt}"
LOG_DIR="$RUN_ROOT/logs"
COLLECT_DIR="$RUN_ROOT/collect_override"
DATA_DIR="$RUN_ROOT/data"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"
ONLINE_ROOT="$RUN_ROOT/online_eval"
STATUS="$RUN_ROOT/status.env"
SUMMARY_JSON="$RUN_ROOT/summary.json"
SUMMARY_CSV="$RUN_ROOT/summary.csv"

timestamp() {
  date --iso-8601=seconds
}

notify_feishu() {
  local title="$1"
  local body="$2"
  if [[ ! -x "$PYTHON" || ! -s "$REPO_ROOT/scripts/feishu_push.py" ]]; then
    return 0
  fi
  {
    printf '%s\n\n%s\n' "$title" "$body"
  } | "$PYTHON" "$REPO_ROOT/scripts/feishu_push.py" >/dev/null 2>&1 || true
}

latest_stage2_metric() {
  local metrics="$STAGE2_DIR/metrics.jsonl"
  if [[ -s "$metrics" ]]; then
    tail -n 1 "$metrics"
  else
    printf 'metrics_pending'
  fi
}

set_stage() {
  local stage="$1"
  {
    printf 'current_stage=%s\n' "$stage"
    printf 'stage_updated_at=%s\n' "$(timestamp)"
  } >> "$STATUS"
}

monitor_progress() {
  while true; do
    sleep 2400
    if [[ -s "$STATUS" ]]; then
      notify_feishu \
        "NewYork epoch11 high-wait distill progress" \
        "Run: $RUN_ROOT
Status: $(tr '\n' ';' < "$STATUS")
Latest Stage2 metric: $(latest_stage2_metric)"
    fi
  done
}

summarize_results() {
  "$PYTHON" - "$RUN_ROOT" "$STAGE2_DIR" "$ONLINE_ROOT" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
stage2_dir = pathlib.Path(sys.argv[2])
online_root = pathlib.Path(sys.argv[3])
summary_json = pathlib.Path(sys.argv[4])
summary_csv = pathlib.Path(sys.argv[5])

rows = []
for result_path in sorted(online_root.glob("epoch_*/result.json")):
    payload = json.load(open(result_path, encoding="utf-8"))
    metrics = dict(payload.get("results") or {})
    epoch = int(result_path.parent.name.split("_")[-1])
    metrics["epoch"] = epoch
    metrics["projection_ckpt"] = str(stage2_dir / f"epoch_{epoch}.pt")
    metrics["path"] = str(result_path)
    metrics["sum_queue_wait_travel"] = sum(
        float(metrics[key])
        for key in (
            "test_avg_queue_len_over",
            "test_avg_waiting_time_over",
            "test_avg_travel_time_over",
        )
    )
    rows.append(metrics)

summary = {
    "run_root": str(run_root),
    "online_results": sorted(rows, key=lambda item: item["epoch"]),
}
summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(rows, key=lambda item: item["epoch"]):
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY
}

on_exit() {
  local code="$?"
  trap - EXIT
  mkdir -p "$LOG_DIR"
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  summarize_results > "$LOG_DIR/summarize.log" 2>&1 || true
  {
    if [[ "$code" -eq 0 ]]; then
      printf 'status=success\n'
    else
      printf 'status=failed\n'
    fi
    printf 'exit_status=%s\n' "$code"
    printf 'completed_at=%s\n' "$(timestamp)"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'summary_json=%s\n' "$SUMMARY_JSON"
    printf 'summary_csv=%s\n' "$SUMMARY_CSV"
  } > "$STATUS"
  if [[ "$code" -eq 0 ]]; then
    notify_feishu "NewYork epoch11 high-wait distill complete" "Run: $RUN_ROOT
Summary: $SUMMARY_CSV
Latest Stage2 metric: $(latest_stage2_metric)"
  else
    notify_feishu "NewYork epoch11 high-wait distill failed" "Run: $RUN_ROOT
Exit: $code
Inspect: $LOG_DIR"
  fi
  exit "$code"
}
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
for required in "$PYTHON" "$TORCHRUN" "$MODEL_PATH" "$BASE_DATA" "$STAGE1_CKPT" "$INIT_CKPT" "$REPO_ROOT/data/NewYork/28_7/$TRAFFIC_FILE"; do
  [[ -e "$required" ]] || { echo "Required path is missing: $required" >&2; exit 1; }
done

mkdir -p "$LOG_DIR" "$COLLECT_DIR" "$DATA_DIR" "$STAGE2_DIR" "$ONLINE_ROOT"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

{
  printf 'status=running\n'
  printf 'started_at=%s\n' "$(timestamp)"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'source_run=%s\n' "$SOURCE_RUN"
  printf 'base_data=%s\n' "$BASE_DATA"
  printf 'stage1_ckpt=%s\n' "$STAGE1_CKPT"
  printf 'init_ckpt=%s\n' "$INIT_CKPT"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'train_gpus=%s\n' "$TRAIN_GPUS"
  printf 'collect_gpus=%s\n' "$COLLECT_GPUS"
  printf 'collect_seeds=%s\n' "$COLLECT_SEEDS"
  printf 'eval_gpus=%s\n' "$EVAL_GPUS"
  printf 'override_gap=%s\n' "$OVERRIDE_GAP"
  printf 'left_wait_protect=%s\n' "$LEFT_WAIT_PROTECT"
  printf 'stage2_epochs=%s\n' "$STAGE2_EPOCHS"
  printf 'stage2_lr=%s\n' "$STAGE2_LR"
  printf 'stage2_min_lr=%s\n' "$STAGE2_MIN_LR"
  printf 'high_wait_resampling=enabled\n'
} > "$STATUS"

notify_feishu "NewYork epoch11 high-wait distill started" "Run: $RUN_ROOT
Flow: $TRAFFIC_FILE
Init: $INIT_CKPT
Override: gap=$OVERRIDE_GAP protect=$LEFT_WAIT_PROTECT
Collect GPUs: $COLLECT_GPUS
Collect seeds: $COLLECT_SEEDS
Stage2 epochs: $STAGE2_EPOCHS"
monitor_progress &
MONITOR_PID="$!"

set_stage "collect_override"
read -r -a collect_gpus <<< "$COLLECT_GPUS"
read -r -a collect_seeds <<< "$COLLECT_SEEDS"
if [[ "${#collect_gpus[@]}" -ne "${#collect_seeds[@]}" ]]; then
  echo "COLLECT_GPUS and COLLECT_SEEDS must contain the same number of entries." >&2
  exit 1
fi
collect_pids=()
collect_status=0
for index in "${!collect_seeds[@]}"; do
  seed="${collect_seeds[$index]}"
  gpu="${collect_gpus[$index]}"
  seed_dir="$COLLECT_DIR/seed_${seed}"
  mkdir -p "$seed_dir"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u projection_train/run_projection_online.py \
      --memo "online_tests/newyork28x7_epoch11_override_distill_collect/seed_${seed}" \
      --projection_ckpt "$INIT_CKPT" \
      --llm_path "$MODEL_PATH" \
      --num_rounds 1 \
      --proj_name "NewYork28x7-epoch11-override-distill-collect-seed${seed}" \
      --dataset newyork_28x7 \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --seed "$seed" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --traffic_series_interval 2 \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --bf16 \
      --max_phase_override_gap "$OVERRIDE_GAP" \
      --max_phase_override_left_wait_protect "$LEFT_WAIT_PROTECT" \
      --collect_states_output "$seed_dir/decision_states.jsonl" \
      --result_json "$seed_dir/result.json"
  ) > "$LOG_DIR/collect_override_seed_${seed}.log" 2>&1 &
  collect_pids+=("$!")
  printf 'seed=%s gpu=%s pid=%s result=%s log=%s\n' \
    "$seed" "$gpu" "${collect_pids[-1]}" "$seed_dir/result.json" "$LOG_DIR/collect_override_seed_${seed}.log" \
    >> "$LOG_DIR/collect_launch.env"
done
for pid in "${collect_pids[@]}"; do
  if ! wait "$pid"; then
    collect_status=1
  fi
done
if [[ "$collect_status" -ne 0 ]]; then
  echo "One or more override collection runs failed." >&2
  exit "$collect_status"
fi

set_stage "build_distill_dataset"
seed_datasets=()
for seed in "${collect_seeds[@]}"; do
  seed_dir="$COLLECT_DIR/seed_${seed}"
  work_dir="$("$PYTHON" - "$seed_dir/result.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("work_dir", ""))
PY
)"
  state_action_source="$work_dir/state_action.json"
  [[ -s "$state_action_source" ]] || { echo "Missing state_action.json: $state_action_source" >&2; exit 1; }
  cp -p "$state_action_source" "$seed_dir/state_action.json"
  seed_data="$DATA_DIR/seed_${seed}_rule_gap${OVERRIDE_GAP}_p${LEFT_WAIT_PROTECT}.jsonl"
  "$PYTHON" scripts/build_queue_wait_distill_dataset.py \
    --state_action "$seed_dir/state_action.json" \
    --prompt_series "$BASE_DATA" \
    --output "$seed_data" \
    --label_source rule \
    --rule_gap "$OVERRIDE_GAP" \
    --left_wait_protect "$LEFT_WAIT_PROTECT" \
    --summary "$DATA_DIR/seed_${seed}_distill.summary.json" \
    > "$LOG_DIR/build_distill_dataset_seed_${seed}.log" 2>&1
  seed_datasets+=("$seed_data")
done

DISTILL_DATA="$DATA_DIR/traffic_prompt_series_2s_epoch11_override_triggers_gap${OVERRIDE_GAP}_p${LEFT_WAIT_PROTECT}.jsonl"
DISTILL_SUMMARY="$DATA_DIR/distill_dataset.summary.json"
"$PYTHON" - "$BASE_DATA" "$DISTILL_DATA" "$DISTILL_SUMMARY" "${seed_datasets[@]}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

base = Path(sys.argv[1])
out = Path(sys.argv[2])
summary = Path(sys.argv[3])
seed_paths = [Path(value) for value in sys.argv[4:]]

base_rows = 0
trigger_rows = 0
trigger_actions = Counter()
trigger_sources = Counter()
out.parent.mkdir(parents=True, exist_ok=True)
with base.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
    for line in fin:
        if line.strip():
            base_rows += 1
        fout.write(line)
    for path in seed_paths:
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            diag = record.get("max_phase_override") or {}
            if diag.get("enabled") and diag.get("overridden"):
                record.setdefault("meta", {})["distill_append_source"] = str(path)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                trigger_rows += 1
                trigger_actions[str(record.get("expert_action", ""))] += 1
                trigger_sources[str(path)] += 1

payload = {
    "schema": "base_plus_override_triggers_v1",
    "base_path": str(base),
    "output_path": str(out),
    "seed_dataset_paths": [str(path) for path in seed_paths],
    "base_rows": base_rows,
    "appended_override_trigger_rows": trigger_rows,
    "total_rows": base_rows + trigger_rows,
    "trigger_action_counts": dict(trigger_actions),
    "trigger_source_counts": dict(trigger_sources),
}
summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

set_stage "stage2_train"
TRAIN_NPROC="$(printf '%s' "$TRAIN_GPUS" | awk -F, '{print NF}')"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
  --data_path "$DISTILL_DATA" \
  --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_CKPT" \
  --init_policy_ckpt "$INIT_CKPT" \
  --output_dir "$STAGE2_DIR" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode queue_wait_short \
  --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" \
  --epochs "$STAGE2_EPOCHS" --lr "$STAGE2_LR" --min_lr "$STAGE2_MIN_LR" --warmup_ratio 0.05 --lr_schedule cosine \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --val_ratio 0.1 --split_strategy stratified --num_workers "$STAGE2_NUM_WORKERS" \
  --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
  --best_metric val_macro_f1 --best_mode max --best_min_delta 0.0 \
  --resample_mode targeted \
  --resample_late_mode date_step --resample_late_interval_seconds 2 --resample_late_start_step 316 --resample_late_weight 1 \
  --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
  --resample_high_queue_threshold 14 --resample_high_queue_weight 1 \
  --resample_enable_high_wait --resample_high_wait_threshold 84 --resample_high_wait_weight 1 \
  --log_interval 20 --save_every 1 --seed 2024 --bf16 \
  --ddp_timeout_seconds 1800 \
  > "$LOG_DIR/stage2.log" 2>&1

set_stage "online_eval"
read -r -a eval_gpus <<< "$EVAL_GPUS"
eval_pids=()
eval_status=0
for epoch in $(seq 1 "$STAGE2_EPOCHS"); do
  checkpoint="$STAGE2_DIR/epoch_${epoch}.pt"
  [[ -s "$checkpoint" ]] || continue
  gpu="${eval_gpus[$(((epoch - 1) % ${#eval_gpus[@]}))]}"
  out_dir="$ONLINE_ROOT/epoch_${epoch}"
  mkdir -p "$out_dir"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u projection_train/run_projection_online.py \
      --memo "online_tests/newyork28x7_epoch11_highwait_override_distill_stage2/epoch_${epoch}/seed_${EVAL_SEED}" \
      --projection_ckpt "$checkpoint" \
      --llm_path "$MODEL_PATH" \
      --num_rounds 1 \
      --proj_name "NewYork28x7-epoch11-highwait-distill-stage2-epoch${epoch}" \
      --dataset newyork_28x7 \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --seed "$EVAL_SEED" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --traffic_series_interval 2 \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --bf16 \
      --result_json "$out_dir/result.json"
  ) > "$LOG_DIR/online_epoch_${epoch}.log" 2>&1 &
  eval_pids+=("$!")
  printf 'epoch=%s gpu=%s pid=%s result=%s log=%s\n' "$epoch" "$gpu" "${eval_pids[-1]}" "$out_dir/result.json" "$LOG_DIR/online_epoch_${epoch}.log" >> "$LOG_DIR/online_launch.env"
done

for pid in "${eval_pids[@]}"; do
  if ! wait "$pid"; then
    eval_status=1
  fi
done
if [[ "$eval_status" -ne 0 ]]; then
  echo "One or more online evaluations failed." >&2
  exit "$eval_status"
fi

set_stage "summarize"
summarize_results
