#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/restart_newyork28x7_qwshort_targeted_stage2_4gpu_20260827_104004}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_28_7_newyork_real_double.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/newyork28x7_epoch11_highwait_override_distill_fixed_4gpu_$(date +%Y%m%d_%H%M%S)}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
COLLECT_GPUS="${COLLECT_GPUS:-0 1 2 3}"
COLLECT_SEEDS="${COLLECT_SEEDS:-2024 2025 2026 2027}"
VAL_SEED="${VAL_SEED:-2027}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3}"
EVAL_SEEDS="${EVAL_SEEDS:-3031 3032 3033 3034}"
RUN_COUNTS="${RUN_COUNTS:-3600}"

OVERRIDE_GAP="${OVERRIDE_GAP:-2}"
LEFT_WAIT_PROTECT="${LEFT_WAIT_PROTECT:-60}"
MAX_WAIT_PROTECT="${MAX_WAIT_PROTECT:-90}"
TARGET_QUEUE_FLOOR="${TARGET_QUEUE_FLOOR:-4}"
TOTAL_QUEUE_FLOOR="${TOTAL_QUEUE_FLOOR:-14}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE2_MIN_LR="${STAGE2_MIN_LR:-1e-6}"
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
SELECTION_JSON="$RUN_ROOT/selection.json"

timestamp() {
  date --iso-8601=seconds
}

notify_feishu() {
  local title="$1"
  local body="$2"
  if [[ ! -x "$PYTHON" || ! -s "$REPO_ROOT/scripts/feishu_push.py" ]]; then
    return 0
  fi
  printf '%s\n\n%s\n' "$title" "$body" | "$PYTHON" "$REPO_ROOT/scripts/feishu_push.py" >/dev/null 2>&1 || true
}

set_stage() {
  local stage="$1"
  {
    printf 'current_stage=%s\n' "$stage"
    printf 'stage_updated_at=%s\n' "$(timestamp)"
  } >> "$STATUS"
}

latest_metric() {
  if [[ -s "$STAGE2_DIR/metrics.jsonl" ]]; then
    tail -n 1 "$STAGE2_DIR/metrics.jsonl"
  else
    printf 'metrics_pending'
  fi
}

monitor_progress() {
  while true; do
    sleep 2400
    notify_feishu \
      "NewYork fixed high-wait distill progress" \
      "Run: $RUN_ROOT
Status: $(tr '\n' ';' < "$STATUS")
Latest Stage2 metric: $(latest_metric)"
  done
}

summarize_results() {
  "$PYTHON" - "$RUN_ROOT" "$ONLINE_ROOT" "$STAGE2_DIR" "$INIT_CKPT" "$SUMMARY_JSON" "$SUMMARY_CSV" "$SELECTION_JSON" <<'PY'
import csv
import json
import pathlib
import shutil
import statistics
import sys

run_root = pathlib.Path(sys.argv[1])
online_root = pathlib.Path(sys.argv[2])
stage2_dir = pathlib.Path(sys.argv[3])
init_ckpt = pathlib.Path(sys.argv[4])
summary_json = pathlib.Path(sys.argv[5])
summary_csv = pathlib.Path(sys.argv[6])
selection_json = pathlib.Path(sys.argv[7])

metric_names = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
)
checkpoints = {"baseline": init_ckpt}
for path in sorted(stage2_dir.glob("epoch_*.pt")):
    checkpoints[path.stem] = path

raw_rows = []
for result_path in sorted(online_root.glob("*/seed_*/result.json")):
    payload = json.load(open(result_path, encoding="utf-8"))
    metrics = dict(payload.get("results") or {})
    label = result_path.parent.parent.name
    seed = int(result_path.parent.name.split("_")[-1])
    row = {"checkpoint": label, "seed": seed, "path": str(result_path)}
    row.update(metrics)
    row["sum_queue_wait_travel"] = sum(
        float(metrics[key])
        for key in (
            "test_avg_queue_len_over",
            "test_avg_waiting_time_over",
            "test_avg_travel_time_over",
        )
    )
    raw_rows.append(row)

aggregates = []
for label in checkpoints:
    rows = [row for row in raw_rows if row["checkpoint"] == label]
    if not rows:
        continue
    aggregate = {
        "checkpoint": label,
        "projection_ckpt": str(checkpoints[label]),
        "seed_count": len(rows),
    }
    for key in (*metric_names, "sum_queue_wait_travel"):
        values = [float(row[key]) for row in rows]
        aggregate[f"{key}_mean"] = statistics.mean(values)
        aggregate[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    aggregates.append(aggregate)

baseline = next((row for row in aggregates if row["checkpoint"] == "baseline"), None)
accepted = []
if baseline is not None:
    for row in aggregates:
        if row["checkpoint"] == "baseline":
            continue
        row["constraint_sum_within_3pct"] = (
            row["sum_queue_wait_travel_mean"] <= baseline["sum_queue_wait_travel_mean"] * 1.03
        )
        row["constraint_p95_improved"] = (
            row["test_vehicle_wait_p95_over_mean"] < baseline["test_vehicle_wait_p95_over_mean"]
        )
        row["constraint_max_not_worse"] = (
            row["test_vehicle_wait_max_over_mean"] <= baseline["test_vehicle_wait_max_over_mean"]
        )
        row["accepted"] = all(
            row[key]
            for key in (
                "constraint_sum_within_3pct",
                "constraint_p95_improved",
                "constraint_max_not_worse",
            )
        )
        if row["accepted"]:
            accepted.append(row)

selected = min(accepted, key=lambda row: row["sum_queue_wait_travel_mean"]) if accepted else baseline
selection = {
    "selected_checkpoint": selected["checkpoint"] if selected else "",
    "selected_projection_ckpt": selected["projection_ckpt"] if selected else "",
    "accepted_candidate_count": len(accepted),
    "fallback_to_baseline": bool(selected and selected["checkpoint"] == "baseline"),
    "constraints": {
        "sum_queue_wait_travel_relative_limit": 1.03,
        "require_p95_improvement": True,
        "require_max_wait_not_worse": True,
    },
}
selection_json.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if selected:
    shutil.copy2(selected["projection_ckpt"], stage2_dir / "online_best.pt")

summary = {
    "run_root": str(run_root),
    "raw_results": raw_rows,
    "aggregates": aggregates,
    "selection": selection,
}
summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

fieldnames = [
    "checkpoint",
    "seed_count",
    "test_reward_over_mean",
    "test_reward_over_std",
    "test_avg_queue_len_over_mean",
    "test_avg_queue_len_over_std",
    "test_avg_waiting_time_over_mean",
    "test_avg_waiting_time_over_std",
    "test_avg_travel_time_over_mean",
    "test_avg_travel_time_over_std",
    "sum_queue_wait_travel_mean",
    "sum_queue_wait_travel_std",
    "test_vehicle_wait_p95_over_mean",
    "test_vehicle_wait_p95_over_std",
    "test_vehicle_wait_max_over_mean",
    "test_vehicle_wait_max_over_std",
    "accepted",
    "projection_ckpt",
]
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in aggregates:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY
}

on_exit() {
  local code="$?"
  trap - EXIT
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
    printf 'selection_json=%s\n' "$SELECTION_JSON"
  } > "$STATUS"
  if [[ "$code" -eq 0 ]]; then
    notify_feishu "NewYork fixed high-wait distill complete" "Run: $RUN_ROOT
Selection: $(cat "$SELECTION_JSON" 2>/dev/null || true)"
  else
    notify_feishu "NewYork fixed high-wait distill failed" "Run: $RUN_ROOT
Exit status: $code
Status: $(tr '\n' ';' < "$STATUS")"
  fi
  exit "$code"
}

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" "$COLLECT_DIR" "$DATA_DIR" "$STAGE2_DIR" "$ONLINE_ROOT"
[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
[[ -x "$TORCHRUN" ]] || { echo "torchrun not executable: $TORCHRUN" >&2; exit 1; }
[[ -s "$STAGE1_CKPT" ]] || { echo "Missing Stage 1 checkpoint: $STAGE1_CKPT" >&2; exit 1; }
[[ -s "$INIT_CKPT" ]] || { echo "Missing init checkpoint: $INIT_CKPT" >&2; exit 1; }

trap on_exit EXIT
{
  printf 'status=running\n'
  printf 'started_at=%s\n' "$(timestamp)"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'source_run=%s\n' "$SOURCE_RUN"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'train_gpus=%s\n' "$TRAIN_GPUS"
  printf 'collect_gpus=%s\n' "$COLLECT_GPUS"
  printf 'collect_seeds=%s\n' "$COLLECT_SEEDS"
  printf 'validation_seed=%s\n' "$VAL_SEED"
  printf 'eval_seeds=%s\n' "$EVAL_SEEDS"
  printf 'override_gap=%s\n' "$OVERRIDE_GAP"
  printf 'left_wait_protect=%s\n' "$LEFT_WAIT_PROTECT"
  printf 'max_wait_protect=%s\n' "$MAX_WAIT_PROTECT"
  printf 'target_queue_floor=%s\n' "$TARGET_QUEUE_FLOOR"
  printf 'total_queue_floor=%s\n' "$TOTAL_QUEUE_FLOOR"
  printf 'stage2_epochs=%s\n' "$STAGE2_EPOCHS"
  printf 'stage2_lr=%s\n' "$STAGE2_LR"
  printf 'data_protocol=full_rollout_per_seed_explicit_holdout\n'
  printf 'resampling=disabled\n'
} > "$STATUS"

notify_feishu "NewYork fixed high-wait distill started" "Run: $RUN_ROOT
Flow: $TRAFFIC_FILE
Train GPUs: $TRAIN_GPUS
Collect seeds: $COLLECT_SEEDS
Validation seed: $VAL_SEED
Evaluation seeds: $EVAL_SEEDS"
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
for index in "${!collect_seeds[@]}"; do
  seed="${collect_seeds[$index]}"
  gpu="${collect_gpus[$index]}"
  seed_dir="$COLLECT_DIR/seed_${seed}"
  mkdir -p "$seed_dir"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u projection_train/run_projection_online.py \
      --memo "online_tests/newyork28x7_epoch11_override_fixed_collect/seed_${seed}" \
      --projection_ckpt "$INIT_CKPT" \
      --llm_path "$MODEL_PATH" \
      --num_rounds 1 \
      --proj_name "NewYork28x7-epoch11-override-fixed-collect-seed${seed}" \
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
      --max_phase_override_max_wait_protect "$MAX_WAIT_PROTECT" \
      --max_phase_override_target_queue_floor "$TARGET_QUEUE_FLOOR" \
      --max_phase_override_total_queue_floor "$TOTAL_QUEUE_FLOOR" \
      --collect_states_output "$seed_dir/decision_states.jsonl" \
      --result_json "$seed_dir/result.json"
  ) > "$LOG_DIR/collect_override_seed_${seed}.log" 2>&1 &
  collect_pids+=("$!")
  printf 'seed=%s gpu=%s pid=%s result=%s log=%s\n' \
    "$seed" "$gpu" "${collect_pids[-1]}" "$seed_dir/result.json" "$LOG_DIR/collect_override_seed_${seed}.log" \
    >> "$LOG_DIR/collect_launch.env"
done
collect_status=0
for pid in "${collect_pids[@]}"; do
  if ! wait "$pid"; then
    collect_status=1
  fi
done
[[ "$collect_status" -eq 0 ]] || { echo "One or more override collection runs failed." >&2; exit "$collect_status"; }

set_stage "build_rollout_datasets"
train_datasets=()
val_dataset=""
for seed in "${collect_seeds[@]}"; do
  seed_dir="$COLLECT_DIR/seed_${seed}"
  work_dir="$($PYTHON - "$seed_dir/result.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("work_dir", ""))
PY
)"
  state_action_source="$work_dir/state_action.json"
  [[ -s "$state_action_source" ]] || { echo "Missing state_action.json: $state_action_source" >&2; exit 1; }
  cp -p "$state_action_source" "$seed_dir/state_action.json"
  seed_data="$DATA_DIR/seed_${seed}_full_rollout_final.jsonl"
  "$PYTHON" scripts/build_queue_wait_distill_dataset.py \
    --state_action "$seed_dir/state_action.json" \
    --decision_states "$seed_dir/decision_states.jsonl" \
    --output "$seed_data" \
    --label_source final \
    --summary "$DATA_DIR/seed_${seed}_full_rollout_final.summary.json" \
    > "$LOG_DIR/build_rollout_dataset_seed_${seed}.log" 2>&1
  if [[ "$seed" == "$VAL_SEED" ]]; then
    val_dataset="$seed_data"
  else
    train_datasets+=("$seed_data")
  fi
done
[[ -n "$val_dataset" ]] || { echo "Validation seed $VAL_SEED is not in COLLECT_SEEDS." >&2; exit 1; }
printf '%s\n' "${train_datasets[@]}" > "$DATA_DIR/train_paths.txt"
printf '%s\n' "$val_dataset" > "$DATA_DIR/val_paths.txt"
"$PYTHON" - "$DATA_DIR" "$VAL_SEED" "${train_datasets[@]}" -- "$val_dataset" <<'PY'
import json
import pathlib
import sys
data_dir = pathlib.Path(sys.argv[1])
val_seed = int(sys.argv[2])
separator = sys.argv.index("--")
train_paths = sys.argv[3:separator]
val_path = sys.argv[separator + 1]
payload = {
    "schema": "full_rollout_explicit_holdout_v1",
    "train_paths": train_paths,
    "val_path": val_path,
    "validation_seed": val_seed,
    "resampling": "disabled",
    "sequence_source": "online_canonical_decision_states",
    "label_source": "logged_final_action",
}
(data_dir / "dataset_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

set_stage "stage2_train"
TRAIN_NPROC="$(printf '%s' "$TRAIN_GPUS" | awk -F, '{print NF}')"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
  --train_data_path "${train_datasets[@]}" \
  --val_data_path "$val_dataset" \
  --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_CKPT" \
  --init_policy_ckpt "$INIT_CKPT" \
  --output_dir "$STAGE2_DIR" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode queue_wait_short \
  --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" \
  --epochs "$STAGE2_EPOCHS" --lr "$STAGE2_LR" --min_lr "$STAGE2_MIN_LR" \
  --warmup_ratio 0.05 --lr_schedule cosine \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --num_workers "$STAGE2_NUM_WORKERS" \
  --target_mode hard --ranking_loss_weight 0 \
  --base_policy_kl_weight 0.02 --base_policy_kl_mode all \
  --best_metric val_loss --best_mode min --best_min_delta 0.0 \
  --log_interval 20 --save_every 1 --checkpoint_slim --seed 2024 --bf16 \
  --ddp_timeout_seconds 1800 \
  > "$LOG_DIR/stage2.log" 2>&1

set_stage "online_eval"
read -r -a eval_gpus <<< "$EVAL_GPUS"
read -r -a eval_seeds <<< "$EVAL_SEEDS"
eval_labels=("baseline")
eval_checkpoints=("$INIT_CKPT")
for epoch in $(seq 1 "$STAGE2_EPOCHS"); do
  checkpoint="$STAGE2_DIR/epoch_${epoch}.pt"
  [[ -s "$checkpoint" ]] || continue
  eval_labels+=("epoch_${epoch}")
  eval_checkpoints+=("$checkpoint")
done
eval_pids=()
eval_status=0
job_index=0
wait_eval_wave() {
  local pid
  for pid in "${eval_pids[@]}"; do
    if ! wait "$pid"; then
      eval_status=1
    fi
  done
  eval_pids=()
}
for checkpoint_index in "${!eval_labels[@]}"; do
  label="${eval_labels[$checkpoint_index]}"
  checkpoint="${eval_checkpoints[$checkpoint_index]}"
  for seed in "${eval_seeds[@]}"; do
    gpu="${eval_gpus[$((job_index % ${#eval_gpus[@]}))]}"
    out_dir="$ONLINE_ROOT/$label/seed_${seed}"
    mkdir -p "$out_dir"
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u projection_train/run_projection_online.py \
        --memo "online_tests/newyork28x7_epoch11_highwait_distill_fixed/${label}/seed_${seed}" \
        --projection_ckpt "$checkpoint" \
        --llm_path "$MODEL_PATH" \
        --num_rounds 1 \
        --proj_name "NewYork28x7-highwait-distill-fixed-${label}-seed${seed}" \
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
        --result_json "$out_dir/result.json"
    ) > "$LOG_DIR/online_${label}_seed_${seed}.log" 2>&1 &
    eval_pids+=("$!")
    printf 'checkpoint=%s seed=%s gpu=%s pid=%s result=%s log=%s\n' \
      "$label" "$seed" "$gpu" "${eval_pids[-1]}" "$out_dir/result.json" "$LOG_DIR/online_${label}_seed_${seed}.log" \
      >> "$LOG_DIR/online_launch.env"
    job_index=$((job_index + 1))
    if [[ "${#eval_pids[@]}" -eq "${#eval_gpus[@]}" ]]; then
      wait_eval_wave
    fi
  done
done
if [[ "${#eval_pids[@]}" -gt 0 ]]; then
  wait_eval_wave
fi
[[ "$eval_status" -eq 0 ]] || { echo "One or more online evaluations failed." >&2; exit "$eval_status"; }

set_stage "summarize"
summarize_results
