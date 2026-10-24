#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/restart_newyork28x7_qwshort_targeted_stage2_4gpu_20260827_104004}"
BASE_DATA="${BASE_DATA:-/mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857/raw/traffic_prompt_series_2s.jsonl}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_28_7_newyork_real_double.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/newyork28x7_waitopt_$(date +%Y%m%d_%H%M%S)}"

EVAL_GPUS="${EVAL_GPUS:-0 1 2 3}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_SEEDS="${EVAL_SEEDS:-3031 3032 3033 3034}"
COLLECT_SEEDS="${COLLECT_SEEDS:-2024 2025 2026 2027}"
EPOCHS_TO_EVAL="${EPOCHS_TO_EVAL:-10 11 12 14 15}"
RUN_COUNTS="${RUN_COUNTS:-3600}"

STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE2_LR="${STAGE2_LR:-5e-6}"
STAGE2_MIN_LR="${STAGE2_MIN_LR:-1e-6}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
TRIGGER_SAMPLE_WEIGHT="${TRIGGER_SAMPLE_WEIGHT:-3}"
RESAMPLE_HIGH_WAIT_THRESHOLD="${RESAMPLE_HIGH_WAIT_THRESHOLD:-84}"
RESAMPLE_HIGH_WAIT_WEIGHT="${RESAMPLE_HIGH_WAIT_WEIGHT:-1}"
RESAMPLE_HIGH_QUEUE_THRESHOLD="${RESAMPLE_HIGH_QUEUE_THRESHOLD:-14}"
RESAMPLE_HIGH_QUEUE_WEIGHT="${RESAMPLE_HIGH_QUEUE_WEIGHT:-1}"
PHASE2_CONFIGS="${PHASE2_CONFIGS:-$'label\tqueue_weight\twait_weight\tp95_weight\tgap\ttarget_queue_floor\ttotal_queue_floor\tleft_wait_protect\tmax_wait_protect\nw15_p00_g15\t1.0\t1.5\t0.0\t1.5\t4\t14\t60\t90\nw20_p05_g15\t1.0\t2.0\t0.5\t1.5\t4\t14\t60\t90\nw30_p10_g20\t1.0\t3.0\t1.0\t2.0\t4\t14\t60\t90'}"
PHASE3_VARIANTS="${PHASE3_VARIANTS:-$'label\tawt\taql\ttail\trank\tatt\tbase_kl\nawt2_aql1_tail05_r003\t2.0\t1.0\t0.5\t0.03\t1.0\t0.0\nawt2_aql075_tail1_r005\t2.0\t0.75\t1.0\t0.05\t1.0\t0.0\nawt3_aql1_tail1_r003\t3.0\t1.0\t1.0\t0.03\t1.0\t0.0'}"

mkdir -p "$RUN_ROOT"/{logs,phase1_checkpoint_eval,phase2_override_grid,phase3_collect,phase3_data,phase3_train,phase3_online}
STATUS="$RUN_ROOT/status.env"
BASELINE_CKPT="$SOURCE_RUN/stage2/checkpoints/epoch_11.pt"
STAGE1_CKPT="$SOURCE_RUN/stage1/best.pt"
COMBINED_DATA="$RUN_ROOT/phase3_data/base_plus_wait_triggers.jsonl"

timestamp() { date --iso-8601=seconds; }

write_status() {
  {
    printf 'status=%s\n' "$1"
    printf 'stage=%s\n' "$2"
    printf 'updated_at=%s\n' "$(timestamp)"
    printf 'run_root=%s\n' "$RUN_ROOT"
  } > "$STATUS"
}

require_file() {
  [[ -s "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

summarize_tree() {
  local input_root="$1"
  local output_json="$2"
  local output_csv="$3"
  "$PYTHON" - "$input_root" "$output_json" "$output_csv" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
out_csv = pathlib.Path(sys.argv[3])
metric_names = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
)
raw = []
for result_path in sorted(root.glob("*/*/result.json")):
    payload = json.load(open(result_path, encoding="utf-8"))
    metrics = payload.get("results") or payload
    if not all(key in metrics for key in metric_names):
        continue
    row = {
        "group": result_path.parent.parent.name,
        "seed": result_path.parent.name,
        "path": str(result_path),
    }
    for key in metric_names:
        row[key] = float(metrics[key])
    row["sum_queue_wait_travel"] = (
        row["test_avg_queue_len_over"]
        + row["test_avg_waiting_time_over"]
        + row["test_avg_travel_time_over"]
    )
    raw.append(row)

aggregates = []
for group in sorted({row["group"] for row in raw}):
    rows = [row for row in raw if row["group"] == group]
    item = {"group": group, "seed_count": len(rows)}
    for key in (*metric_names, "sum_queue_wait_travel"):
        values = [row[key] for row in rows]
        item[f"{key}_mean"] = statistics.mean(values)
        item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    item["wait_objective_mean"] = (
        item["test_avg_waiting_time_over_mean"]
        + 0.15 * item["test_vehicle_wait_p95_over_mean"]
    )
    aggregates.append(item)

out_json.write_text(json.dumps({"raw_results": raw, "aggregates": aggregates}, indent=2), encoding="utf-8")
fields = [
    "group", "seed_count",
    "test_reward_over_mean", "test_avg_queue_len_over_mean",
    "test_avg_waiting_time_over_mean", "test_avg_travel_time_over_mean",
    "sum_queue_wait_travel_mean", "test_vehicle_wait_p95_over_mean",
    "test_vehicle_wait_max_over_mean", "wait_objective_mean",
    "sum_queue_wait_travel_std", "test_avg_waiting_time_over_std",
    "test_vehicle_wait_p95_over_std",
]
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in aggregates:
        writer.writerow({key: row.get(key, "") for key in fields})
PY
}

launch_online() {
  local gpu="$1"; shift
  local label="$1"; shift
  local seed="$1"; shift
  local ckpt="$1"; shift
  local out_dir="$1"; shift
  local log_path="$1"; shift
  mkdir -p "$out_dir" "$(dirname "$log_path")"
  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u projection_train/run_projection_online.py \
      --memo "online_tests/newyork28x7_waitopt/${label}/seed_${seed}" \
      --projection_ckpt "$ckpt" \
      --llm_path "$MODEL_PATH" \
      --dataset newyork_28x7 \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --num_rounds 1 \
      --seed "$seed" \
      --proj_name "NewYork28x7-waitopt-${label}-seed${seed}" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --traffic_series_interval 2 \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --bf16 \
      --result_json "$out_dir/result.json" \
      "$@"
  ) > "$log_path" 2>&1 &
  LAST_PID="$!"
}

wait_wave() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

select_override() {
  "$PYTHON" - "$RUN_ROOT/phase1_checkpoint_eval/summary.json" "$RUN_ROOT/phase2_override_grid/summary.json" "$RUN_ROOT/phase2_override_grid/selection.env" <<'PY'
import json
import pathlib
import sys

phase1 = json.load(open(sys.argv[1], encoding="utf-8"))
phase2 = json.load(open(sys.argv[2], encoding="utf-8"))
out = pathlib.Path(sys.argv[3])
baseline = next(row for row in phase1["aggregates"] if row["group"] == "epoch_11")
accepted = []
for row in phase2["aggregates"]:
    row = dict(row)
    row["constraint_sum_within_3pct"] = row["sum_queue_wait_travel_mean"] <= baseline["sum_queue_wait_travel_mean"] * 1.03
    row["constraint_max_within_3pct"] = row["test_vehicle_wait_max_over_mean"] <= baseline["test_vehicle_wait_max_over_mean"] * 1.03
    row["constraint_queue_within_10pct"] = row["test_avg_queue_len_over_mean"] <= baseline["test_avg_queue_len_over_mean"] * 1.10
    row["accepted"] = all(row[key] for key in ("constraint_sum_within_3pct", "constraint_max_within_3pct", "constraint_queue_within_10pct"))
    if row["accepted"]:
        accepted.append(row)
pool = accepted or [dict(row, accepted=False) for row in phase2["aggregates"]]
selected = min(pool, key=lambda row: (row["wait_objective_mean"], row["sum_queue_wait_travel_mean"]))
out.write_text(
    "\n".join([
        f"selected_override_label={selected['group']}",
        f"selected_override_accepted={str(bool(selected.get('accepted'))).lower()}",
        f"baseline_sum={baseline['sum_queue_wait_travel_mean']}",
        f"baseline_wait={baseline['test_avg_waiting_time_over_mean']}",
        f"baseline_p95={baseline['test_vehicle_wait_p95_over_mean']}",
        f"selected_sum={selected['sum_queue_wait_travel_mean']}",
        f"selected_wait={selected['test_avg_waiting_time_over_mean']}",
        f"selected_p95={selected['test_vehicle_wait_p95_over_mean']}",
        "",
    ]),
    encoding="utf-8",
)
PY
}

trap 'code=$?; write_status failed "exit_$code"; exit "$code"' ERR

cd "$REPO_ROOT"
require_file "$BASELINE_CKPT"
require_file "$STAGE1_CKPT"
require_file "$BASE_DATA"
write_status running preflight

read -r -a EVAL_GPU_ARRAY <<< "$EVAL_GPUS"
if [[ "${#EVAL_GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "No EVAL_GPUS configured." >&2
  exit 1
fi

write_status running phase1_checkpoint_eval
pids=()
job_index=0
for epoch in $EPOCHS_TO_EVAL; do
  ckpt="$SOURCE_RUN/stage2/checkpoints/epoch_${epoch}.pt"
  require_file "$ckpt"
  for seed in $EVAL_SEEDS; do
    gpu="${EVAL_GPU_ARRAY[$((job_index % ${#EVAL_GPU_ARRAY[@]}))]}"
    label="epoch_${epoch}"
    out_dir="$RUN_ROOT/phase1_checkpoint_eval/$label/seed_${seed}"
    log_path="$RUN_ROOT/logs/phase1_${label}_seed_${seed}.log"
    launch_online "$gpu" "$label" "$seed" "$ckpt" "$out_dir" "$log_path" --disable_snapshots
    pids+=("$LAST_PID")
    job_index=$((job_index + 1))
    if [[ "${#pids[@]}" -eq "${#EVAL_GPU_ARRAY[@]}" ]]; then
      wait_wave "${pids[@]}"
      pids=()
    fi
  done
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_wave "${pids[@]}"
fi
summarize_tree "$RUN_ROOT/phase1_checkpoint_eval" "$RUN_ROOT/phase1_checkpoint_eval/summary.json" "$RUN_ROOT/phase1_checkpoint_eval/summary.csv"

write_status running phase2_override_grid
printf '%s\n' "$PHASE2_CONFIGS" > "$RUN_ROOT/phase2_override_grid/configs.tsv"
pids=()
job_index=0
while IFS=$'\t' read -r label queue_weight wait_weight p95_weight gap tq_floor total_floor left_protect max_protect; do
  [[ "$label" == "label" ]] && continue
  for seed in $EVAL_SEEDS; do
    gpu="${EVAL_GPU_ARRAY[$((job_index % ${#EVAL_GPU_ARRAY[@]}))]}"
    out_dir="$RUN_ROOT/phase2_override_grid/$label/seed_${seed}"
    log_path="$RUN_ROOT/logs/phase2_${label}_seed_${seed}.log"
    launch_online "$gpu" "$label" "$seed" "$BASELINE_CKPT" "$out_dir" "$log_path" \
      --max_phase_override_gap "$gap" \
      --max_phase_override_queue_weight "$queue_weight" \
      --max_phase_override_wait_weight "$wait_weight" \
      --max_phase_override_p95_wait_weight "$p95_weight" \
      --max_phase_override_target_queue_floor "$tq_floor" \
      --max_phase_override_total_queue_floor "$total_floor" \
      --max_phase_override_left_wait_protect "$left_protect" \
      --max_phase_override_max_wait_protect "$max_protect" \
      --disable_snapshots
    pids+=("$LAST_PID")
    job_index=$((job_index + 1))
    if [[ "${#pids[@]}" -eq "${#EVAL_GPU_ARRAY[@]}" ]]; then
      wait_wave "${pids[@]}"
      pids=()
    fi
  done
done < "$RUN_ROOT/phase2_override_grid/configs.tsv"
if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_wave "${pids[@]}"
fi
summarize_tree "$RUN_ROOT/phase2_override_grid" "$RUN_ROOT/phase2_override_grid/summary.json" "$RUN_ROOT/phase2_override_grid/summary.csv"
select_override

write_status running phase3_collect_wait_triggers
# shellcheck disable=SC1091
source "$RUN_ROOT/phase2_override_grid/selection.env"
selected_line="$(awk -F'\t' -v label="$selected_override_label" '$1==label {print}' "$RUN_ROOT/phase2_override_grid/configs.tsv")"
if [[ -z "$selected_line" ]]; then
  echo "Could not recover selected override config: $selected_override_label" >&2
  exit 1
fi
IFS=$'\t' read -r selected_label selected_qw selected_ww selected_p95 selected_gap selected_tq selected_total selected_left selected_max <<< "$selected_line"
pids=()
job_index=0
for seed in $COLLECT_SEEDS; do
  gpu="${EVAL_GPU_ARRAY[$((job_index % ${#EVAL_GPU_ARRAY[@]}))]}"
  out_dir="$RUN_ROOT/phase3_collect/seed_${seed}"
  log_path="$RUN_ROOT/logs/phase3_collect_seed_${seed}.log"
  mkdir -p "$out_dir"
  launch_online "$gpu" "phase3_collect_${selected_label}" "$seed" "$BASELINE_CKPT" "$out_dir" "$log_path" \
    --max_phase_override_gap "$selected_gap" \
    --max_phase_override_queue_weight "$selected_qw" \
    --max_phase_override_wait_weight "$selected_ww" \
    --max_phase_override_p95_wait_weight "$selected_p95" \
    --max_phase_override_target_queue_floor "$selected_tq" \
    --max_phase_override_total_queue_floor "$selected_total" \
    --max_phase_override_left_wait_protect "$selected_left" \
    --max_phase_override_max_wait_protect "$selected_max" \
    --collect_states_output "$out_dir/decision_states.jsonl" \
    --disable_snapshots
  pids+=("$LAST_PID")
  job_index=$((job_index + 1))
  if [[ "${#pids[@]}" -eq "${#EVAL_GPU_ARRAY[@]}" ]]; then
    wait_wave "${pids[@]}"
    pids=()
  fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_wave "${pids[@]}"
fi
summarize_tree "$RUN_ROOT/phase3_collect" "$RUN_ROOT/phase3_collect/summary.json" "$RUN_ROOT/phase3_collect/summary.csv"

trigger_paths=()
for seed in $COLLECT_SEEDS; do
  seed_dir="$RUN_ROOT/phase3_collect/seed_${seed}"
  work_dir="$("$PYTHON" - "$seed_dir/result.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("work_dir", ""))
PY
)"
  require_file "$work_dir/state_action.json"
  cp -p "$work_dir/state_action.json" "$seed_dir/state_action.json"
  full_jsonl="$RUN_ROOT/phase3_data/seed_${seed}_full_waitaware.jsonl"
  trigger_jsonl="$RUN_ROOT/phase3_data/seed_${seed}_triggers.jsonl"
  "$PYTHON" scripts/build_queue_wait_distill_dataset.py \
    --state_action "$seed_dir/state_action.json" \
    --decision_states "$seed_dir/decision_states.jsonl" \
    --label_source final \
    --output "$full_jsonl" \
    --summary "$RUN_ROOT/phase3_data/seed_${seed}_full_waitaware.summary.json" \
    > "$RUN_ROOT/logs/build_wait_trigger_seed_${seed}.log" 2>&1
  "$PYTHON" - "$full_jsonl" "$trigger_jsonl" "$RUN_ROOT/phase3_data/seed_${seed}_triggers.summary.json" "$TRIGGER_SAMPLE_WEIGHT" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
summary = pathlib.Path(sys.argv[3])
rows = 0
kept = 0
actions = {}
with src.open("r", encoding="utf-8") as source, dst.open("w", encoding="utf-8") as out:
    for line in source:
        if not line.strip():
            continue
        rows += 1
        record = json.loads(line)
        diag = record.get("max_phase_override") or {}
        if not bool(diag.get("overridden")):
            continue
        record["sample_weight"] = max(float(record.get("sample_weight", 1.0) or 1.0), float(sys.argv[4]))
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
        action = str(record.get("expert_action") or "")
        actions[action] = actions.get(action, 0) + 1
summary.write_text(json.dumps({"source_rows": rows, "trigger_rows": kept, "action_counts": actions}, indent=2), encoding="utf-8")
if kept <= 0:
    raise SystemExit("No trigger rows selected.")
PY
  trigger_paths+=("$trigger_jsonl")
done

"$PYTHON" - "$BASE_DATA" "$COMBINED_DATA" "${trigger_paths[@]}" <<'PY'
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
triggers = [pathlib.Path(path) for path in sys.argv[3:]]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    for path in [base, *triggers]:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    handle.write(line)
PY

write_status running phase3_train_and_eval
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
TRAIN_NPROC="${TRAIN_NPROC:-${#TRAIN_GPU_ARRAY[@]}}"
printf '%s\n' "$PHASE3_VARIANTS" > "$RUN_ROOT/phase3_train/variants.tsv"

while IFS=$'\t' read -r label awt aql tail rank att base_kl; do
  [[ "$label" == "label" ]] && continue
  out_dir="$RUN_ROOT/phase3_train/$label/checkpoints"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
    --data_path "$COMBINED_DATA" \
    --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_CKPT" \
    --init_policy_ckpt "$BASELINE_CKPT" \
    --output_dir "$out_dir" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --prompt_mode queue_wait_short \
    --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" \
    --epochs "$STAGE2_EPOCHS" --lr "$STAGE2_LR" --min_lr "$STAGE2_MIN_LR" \
    --warmup_ratio 0.05 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers 4 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --ranking_loss_weight "$rank" --ranking_min_cost_gap 0.02 \
    --cost_awt_weight "$awt" --cost_aql_weight "$aql" --cost_att_weight "$att" --cost_tail_wait_weight "$tail" \
    --base_policy_kl_weight "$base_kl" \
    --resample_mode targeted \
    --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
    --resample_high_queue_threshold "$RESAMPLE_HIGH_QUEUE_THRESHOLD" --resample_high_queue_weight "$RESAMPLE_HIGH_QUEUE_WEIGHT" \
    --resample_enable_high_wait --resample_high_wait_threshold "$RESAMPLE_HIGH_WAIT_THRESHOLD" --resample_high_wait_weight "$RESAMPLE_HIGH_WAIT_WEIGHT" \
    --best_metric val_expected_cost_regret --best_mode min --best_min_delta 0.0 \
    --log_interval 20 --save_every 1 --checkpoint_slim --seed 2024 --bf16 \
    --ddp_timeout_seconds 1800 \
    > "$RUN_ROOT/logs/phase3_train_${label}.log" 2>&1

  eval_ckpt="$out_dir/best.pt"
  require_file "$eval_ckpt"
  pids=()
  job_index=0
  for seed in $EVAL_SEEDS; do
    gpu="${EVAL_GPU_ARRAY[$((job_index % ${#EVAL_GPU_ARRAY[@]}))]}"
    eval_out="$RUN_ROOT/phase3_online/$label/seed_${seed}"
    eval_log="$RUN_ROOT/logs/phase3_online_${label}_seed_${seed}.log"
    launch_online "$gpu" "phase3_${label}" "$seed" "$eval_ckpt" "$eval_out" "$eval_log" --disable_snapshots
    pids+=("$LAST_PID")
    job_index=$((job_index + 1))
    if [[ "${#pids[@]}" -eq "${#EVAL_GPU_ARRAY[@]}" ]]; then
      wait_wave "${pids[@]}"
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_wave "${pids[@]}"
  fi
done < "$RUN_ROOT/phase3_train/variants.tsv"

summarize_tree "$RUN_ROOT/phase3_online" "$RUN_ROOT/phase3_online/summary.json" "$RUN_ROOT/phase3_online/summary.csv"
write_status success complete
