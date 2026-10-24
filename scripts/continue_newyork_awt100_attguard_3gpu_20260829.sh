#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/restart_newyork28x7_qwshort_targeted_stage2_4gpu_20260827_104004}"
FAILED_RUN="${FAILED_RUN:-$REPO_ROOT/records/pipeline_runs/newyork28x7_awt100_attguard_20260828_205628}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/newyork28x7_awt100_attguard_continue_$(date +%Y%m%d_%H%M%S)}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2}"
TRAIN_NPROC="${TRAIN_NPROC:-3}"
EVAL_SEEDS="${EVAL_SEEDS:-3031 3032 3033 3034}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_28_7_newyork_real_double.json}"

STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE2_LR="${STAGE2_LR:-5e-6}"
STAGE2_MIN_LR="${STAGE2_MIN_LR:-1e-6}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
RESAMPLE_HIGH_WAIT_THRESHOLD="${RESAMPLE_HIGH_WAIT_THRESHOLD:-60}"
RESAMPLE_HIGH_WAIT_WEIGHT="${RESAMPLE_HIGH_WAIT_WEIGHT:-3}"
RESAMPLE_HIGH_QUEUE_THRESHOLD="${RESAMPLE_HIGH_QUEUE_THRESHOLD:-99999}"
RESAMPLE_HIGH_QUEUE_WEIGHT="${RESAMPLE_HIGH_QUEUE_WEIGHT:-1}"
PHASE3_VARIANTS="${PHASE3_VARIANTS:-$'label\tawt\taql\ttail\trank\tatt\tbase_kl\nawt5_attguard\t5.0\t0.10\t2.0\t0.05\t1.0\t0.05\nawt6_attguard\t6.0\t0.10\t2.5\t0.05\t1.0\t0.08\nawt7_attguard\t7.0\t0.15\t3.0\t0.07\t1.0\t0.10'}"

BASELINE_CKPT="$SOURCE_RUN/stage2/checkpoints/epoch_11.pt"
STAGE1_CKPT="$SOURCE_RUN/stage1/best.pt"
COMBINED_DATA="$FAILED_RUN/phase3_data/base_plus_wait_triggers.jsonl"
STATUS="$RUN_ROOT/status.env"
CAMPAIGN_LOG="$RUN_ROOT/logs/continuation.log"
ALL_BEST_DIR="$REPO_ROOT/All_best"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
LAST_NOTIFY_EPOCH=0

timestamp() { date --iso-8601=seconds; }

write_status() {
  {
    printf 'status=%s\n' "$1"
    printf 'stage=%s\n' "$2"
    printf 'updated_at=%s\n' "$(timestamp)"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'failed_run=%s\n' "$FAILED_RUN"
    printf 'train_gpus=%s\n' "$TRAIN_GPUS"
    printf 'eval_gpus=%s\n' "$EVAL_GPUS"
  } > "$STATUS"
}

log_msg() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$CAMPAIGN_LOG"
}

require_file() {
  [[ -s "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

notify() {
  [[ -s "$FEISHU_SCRIPT" ]] || return 0
  timeout --foreground 15s "$PYTHON" "$FEISHU_SCRIPT" "$1" >/dev/null 2>&1 || true
}

latest_train_metric() {
  local metrics_path="$1"
  if [[ -s "$metrics_path" ]]; then
    tail -n 1 "$metrics_path" | cut -c 1-500
  else
    printf 'metrics=pending'
  fi
}

latest_result_line() {
  if [[ -s "$RUN_ROOT/phase3_online/summary.csv" ]]; then
    tail -n +2 "$RUN_ROOT/phase3_online/summary.csv" | tail -n 3 | cut -c 1-900
  else
    printf 'online_summary=pending'
  fi
}

maybe_notify_progress() {
  local label="$1"
  local now
  now="$(date +%s)"
  if (( LAST_NOTIFY_EPOCH == 0 || now - LAST_NOTIFY_EPOCH >= 2400 )); then
    LAST_NOTIFY_EPOCH="$now"
    notify "NewYork28x7 awt100 attguard progress: stage=${label}; run=${RUN_ROOT}; $(latest_result_line)"
  fi
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

write_report() {
  "$PYTHON" - "$FAILED_RUN" "$RUN_ROOT" "$ALL_BEST_DIR" <<'PY'
import csv
import json
import pathlib
import sys

failed_run = pathlib.Path(sys.argv[1])
run_root = pathlib.Path(sys.argv[2])
all_best = pathlib.Path(sys.argv[3])
baseline = json.load(open(failed_run / "phase1_checkpoint_eval" / "summary.json", encoding="utf-8"))["aggregates"][0]
teacher_rows = json.load(open(failed_run / "phase2_override_grid" / "summary.json", encoding="utf-8"))["aggregates"]
teacher = min(teacher_rows, key=lambda row: row["test_avg_waiting_time_over_mean"])
online_json = run_root / "phase3_online" / "summary.json"
online = json.load(open(online_json, encoding="utf-8"))["aggregates"]

def delta(value: float, ref: float) -> float:
    return 100.0 * (value - ref) / ref

ranked = sorted(
    online,
    key=lambda row: (
        row["test_avg_travel_time_over_mean"] > baseline["test_avg_travel_time_over_mean"] * 1.03,
        row["test_avg_waiting_time_over_mean"],
        row["test_avg_travel_time_over_mean"],
    ),
)
best = ranked[0] if ranked else None
rows = []
for row in ranked:
    rows.append({
        "policy": row["group"],
        "seeds": row["seed_count"],
        "awt": row["test_avg_waiting_time_over_mean"],
        "awt_std": row["test_avg_waiting_time_over_std"],
        "att": row["test_avg_travel_time_over_mean"],
        "att_delta_pct": delta(row["test_avg_travel_time_over_mean"], baseline["test_avg_travel_time_over_mean"]),
        "aql": row["test_avg_queue_len_over_mean"],
        "p95": row["test_vehicle_wait_p95_over_mean"],
        "max_wait": row["test_vehicle_wait_max_over_mean"],
    })

all_best.mkdir(parents=True, exist_ok=True)
csv_path = run_root / "phase3_online" / "comparison.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["policy"])
    writer.writeheader()
    writer.writerows(rows)

report_name = f"{run_root.name}_results.md"
report_path = all_best / report_name
lines = [
    "# NewYork28x7 AWT100 ATT-guard continuation",
    "",
    f"- Run: `{run_root}`",
    f"- Reused failed run data: `{failed_run}`",
    "- Protocol: same NewYork 28x7 flow, online seeds 3031/3032/3033/3034.",
    "- Selection note: primary target is lower AWT; ATT is compared with the epoch-11 baseline.",
    "",
    "## References",
    "",
    "| Reference | AWT | ATT | AQL | P95 wait | Max wait |",
    "|---|---:|---:|---:|---:|---:|",
    f"| epoch_11 baseline | {baseline['test_avg_waiting_time_over_mean']:.2f} | {baseline['test_avg_travel_time_over_mean']:.2f} | {baseline['test_avg_queue_len_over_mean']:.2f} | {baseline['test_vehicle_wait_p95_over_mean']:.2f} | {baseline['test_vehicle_wait_max_over_mean']:.2f} |",
    f"| teacher {teacher['group']} | {teacher['test_avg_waiting_time_over_mean']:.2f} | {teacher['test_avg_travel_time_over_mean']:.2f} | {teacher['test_avg_queue_len_over_mean']:.2f} | {teacher['test_vehicle_wait_p95_over_mean']:.2f} | {teacher['test_vehicle_wait_max_over_mean']:.2f} |",
    "",
    "## Student Results",
    "",
    "| Policy | Seeds | AWT | AWT std | ATT | ATT delta vs baseline | AQL | P95 wait | Max wait |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['policy']} | {row['seeds']} | {row['awt']:.2f} | {row['awt_std']:.2f} | "
        f"{row['att']:.2f} | {row['att_delta_pct']:+.2f}% | {row['aql']:.2f} | "
        f"{row['p95']:.2f} | {row['max_wait']:.2f} |"
    )
if best:
    lines.extend([
        "",
        "## Current Best",
        "",
        f"`{best['group']}` is the current best by the continuation selector: "
        f"AWT={best['test_avg_waiting_time_over_mean']:.2f}, "
        f"ATT={best['test_avg_travel_time_over_mean']:.2f}, "
        f"AQL={best['test_avg_queue_len_over_mean']:.2f}.",
    ])
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report_path)
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
      --memo "online_tests/newyork28x7_awt100_attguard_continue/${label}/seed_${seed}" \
      --projection_ckpt "$ckpt" \
      --llm_path "$MODEL_PATH" \
      --dataset newyork_28x7 \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --num_rounds 1 \
      --seed "$seed" \
      --proj_name "NewYork28x7-awt100-${label}-seed${seed}" \
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

wait_pids_progress() {
  local label="$1"; shift
  local pids=("$@")
  local alive
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    [[ "$alive" -gt 0 ]] || break
    sleep 60
    maybe_notify_progress "$label alive=${alive}"
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

train_variant() {
  local label="$1"
  local awt="$2"
  local aql="$3"
  local tail="$4"
  local rank="$5"
  local att="$6"
  local base_kl="$7"
  local out_dir="$RUN_ROOT/phase3_train/$label/checkpoints"
  local log_path="$RUN_ROOT/logs/phase3_train_${label}.log"
  mkdir -p "$out_dir"
  log_msg "Training ${label} on GPUs ${TRAIN_GPUS}"
  write_status running "train_${label}"
  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONUNBUFFERED=1 "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
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
      --ddp_timeout_seconds 1800
  ) > "$log_path" 2>&1 &
  local pid="$!"
  maybe_notify_progress "train_${label} started"
  if ! wait_pids_progress "train_${label} $(latest_train_metric "$out_dir/metrics.jsonl")" "$pid"; then
    log_msg "Training failed for ${label}; inspect ${log_path}"
    return 1
  fi
  require_file "$out_dir/best.pt"
  log_msg "Training complete for ${label}; best checkpoint: ${out_dir}/best.pt"
}

eval_variant() {
  local label="$1"
  local ckpt="$RUN_ROOT/phase3_train/$label/checkpoints/best.pt"
  require_file "$ckpt"
  log_msg "Evaluating ${label} on same NewYork 28x7 flow with seeds ${EVAL_SEEDS}"
  write_status running "eval_${label}"
  read -r -a eval_gpu_array <<< "$EVAL_GPUS"
  local pids=()
  local job_index=0
  for seed in $EVAL_SEEDS; do
    local gpu="${eval_gpu_array[$((job_index % ${#eval_gpu_array[@]}))]}"
    local out_dir="$RUN_ROOT/phase3_online/$label/seed_${seed}"
    local log_path="$RUN_ROOT/logs/phase3_online_${label}_seed_${seed}.log"
    launch_online "$gpu" "$label" "$seed" "$ckpt" "$out_dir" "$log_path" --disable_snapshots
    pids+=("$LAST_PID")
    job_index=$((job_index + 1))
    if [[ "${#pids[@]}" -eq "${#eval_gpu_array[@]}" ]]; then
      wait_pids_progress "eval_${label}" "${pids[@]}"
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_pids_progress "eval_${label}" "${pids[@]}"
  fi
  summarize_tree "$RUN_ROOT/phase3_online" "$RUN_ROOT/phase3_online/summary.json" "$RUN_ROOT/phase3_online/summary.csv"
  log_msg "Evaluation summarized for ${label}: $(latest_result_line)"
}

finish() {
  local code="$?"
  if [[ "$code" -eq 0 ]]; then
    write_status success complete
    notify "NewYork28x7 awt100 attguard continuation complete: run=${RUN_ROOT}; $(latest_result_line)"
  else
    write_status failed "exit_${code}"
    notify "NewYork28x7 awt100 attguard continuation failed: exit=${code}; run=${RUN_ROOT}; inspect ${CAMPAIGN_LOG}"
  fi
}
trap finish EXIT

mkdir -p "$RUN_ROOT"/{logs,phase1_checkpoint_eval,phase2_override_grid,phase3_collect,phase3_data,phase3_train,phase3_online}
mkdir -p "$ALL_BEST_DIR"
touch "$CAMPAIGN_LOG"
write_status running preflight

require_file "$BASELINE_CKPT"
require_file "$STAGE1_CKPT"
require_file "$COMBINED_DATA"
require_file "$FAILED_RUN/phase1_checkpoint_eval/summary.json"
require_file "$FAILED_RUN/phase2_override_grid/summary.json"

cp -p "$FAILED_RUN/phase1_checkpoint_eval/summary.json" "$RUN_ROOT/phase1_checkpoint_eval/source_summary.json"
cp -p "$FAILED_RUN/phase1_checkpoint_eval/summary.csv" "$RUN_ROOT/phase1_checkpoint_eval/source_summary.csv"
cp -p "$FAILED_RUN/phase2_override_grid/summary.json" "$RUN_ROOT/phase2_override_grid/source_summary.json"
cp -p "$FAILED_RUN/phase2_override_grid/summary.csv" "$RUN_ROOT/phase2_override_grid/source_summary.csv"
cp -p "$FAILED_RUN/phase2_override_grid/selection.env" "$RUN_ROOT/phase2_override_grid/source_selection.env"
cp -p "$FAILED_RUN/phase3_collect/summary.json" "$RUN_ROOT/phase3_collect/source_summary.json"
cp -p "$FAILED_RUN/phase3_collect/summary.csv" "$RUN_ROOT/phase3_collect/source_summary.csv"
ln -s "$COMBINED_DATA" "$RUN_ROOT/phase3_data/base_plus_wait_triggers.source.jsonl"
printf '%s\n' "$PHASE3_VARIANTS" > "$RUN_ROOT/phase3_train/variants.tsv"
{
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'failed_run=%s\n' "$FAILED_RUN"
  printf 'combined_data=%s\n' "$COMBINED_DATA"
  printf 'baseline_ckpt=%s\n' "$BASELINE_CKPT"
  printf 'stage1_ckpt=%s\n' "$STAGE1_CKPT"
  printf 'train_gpus=%s\n' "$TRAIN_GPUS"
  printf 'train_nproc=%s\n' "$TRAIN_NPROC"
  printf 'eval_gpus=%s\n' "$EVAL_GPUS"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'eval_seeds=%s\n' "$EVAL_SEEDS"
  printf 'run_counts=%s\n' "$RUN_COUNTS"
  printf 'protocol=same-flow NewYork28x7\n'
} > "$RUN_ROOT/manifest.env"

log_msg "Continuation run initialized at ${RUN_ROOT}"
notify "NewYork28x7 awt100 attguard continuation started: run=${RUN_ROOT}; train_gpus=${TRAIN_GPUS}; eval_gpus=${EVAL_GPUS}; reused_data=${COMBINED_DATA}"

while IFS=$'\t' read -r label awt aql tail rank att base_kl; do
  [[ "$label" == "label" ]] && continue
  train_variant "$label" "$awt" "$aql" "$tail" "$rank" "$att" "$base_kl"
  eval_variant "$label"
done < "$RUN_ROOT/phase3_train/variants.tsv"

write_report | tee -a "$CAMPAIGN_LOG"
log_msg "Continuation completed"
