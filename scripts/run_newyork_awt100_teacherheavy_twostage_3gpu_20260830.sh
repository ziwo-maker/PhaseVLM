#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/restart_newyork28x7_qwshort_targeted_stage2_4gpu_20260827_104004}"
TEACHER_RUN="${TEACHER_RUN:-$REPO_ROOT/records/pipeline_runs/newyork28x7_awt100_attguard_20260828_205628}"
PREV_STUDENT_RUN="${PREV_STUDENT_RUN:-$REPO_ROOT/records/pipeline_runs/newyork28x7_awt100_attguard_continue_20260829_224552}"
BASE_DATA="${BASE_DATA:-/mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857/raw/traffic_prompt_series_2s.jsonl}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/newyork28x7_awt100_teacherheavy_twostage_$(date +%Y%m%d_%H%M%S)}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
TRAIN_NPROC="${TRAIN_NPROC:-3}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-3031 3032 3033 3034}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_28_7_newyork_real_double.json}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"

STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-2}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-2}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
STAGE2_LR="${STAGE2_LR:-5e-6}"
STAGE2_MIN_LR="${STAGE2_MIN_LR:-1e-6}"

BASELINE_CKPT="$SOURCE_RUN/stage2/checkpoints/epoch_11.pt"
STAGE1_CKPT="$SOURCE_RUN/stage1/best.pt"
STATUS="$RUN_ROOT/status.env"
CAMPAIGN_LOG="$RUN_ROOT/logs/campaign.log"
ALL_BEST_DIR="$REPO_ROOT/All_best"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
LAST_NOTIFY_EPOCH=0

TRIGGER_FILES=(
  "$TEACHER_RUN/phase3_data/seed_2024_triggers.jsonl"
  "$TEACHER_RUN/phase3_data/seed_2025_triggers.jsonl"
  "$TEACHER_RUN/phase3_data/seed_2026_triggers.jsonl"
  "$TEACHER_RUN/phase3_data/seed_2027_triggers.jsonl"
)

timestamp() { date --iso-8601=seconds; }

write_status() {
  {
    printf 'status=%s\n' "$1"
    printf 'stage=%s\n' "$2"
    printf 'updated_at=%s\n' "$(timestamp)"
    printf 'run_root=%s\n' "$RUN_ROOT"
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

latest_result_line() {
  if [[ -s "$RUN_ROOT/phase_online/summary.csv" ]]; then
    tail -n +2 "$RUN_ROOT/phase_online/summary.csv" | tail -n 4 | cut -c 1-1200
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
    notify "NewYork28x7 teacher-heavy two-stage progress: stage=${label}; run=${RUN_ROOT}; $(latest_result_line)"
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
    row = {"group": result_path.parent.parent.name, "seed": result_path.parent.name, "path": str(result_path)}
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
  "$PYTHON" - "$TEACHER_RUN" "$PREV_STUDENT_RUN" "$RUN_ROOT" "$ALL_BEST_DIR" <<'PY'
import csv
import json
import pathlib
import sys

teacher_run = pathlib.Path(sys.argv[1])
prev_run = pathlib.Path(sys.argv[2])
run_root = pathlib.Path(sys.argv[3])
all_best = pathlib.Path(sys.argv[4])

def load_agg(path):
    return {row["group"]: row for row in json.load(open(path, encoding="utf-8"))["aggregates"]}

baseline = load_agg(teacher_run / "phase1_checkpoint_eval" / "summary.json")["epoch_11"]
teacher = load_agg(teacher_run / "phase2_override_grid" / "summary.json")["wait7_p30_g05"]
prev = load_agg(prev_run / "phase3_online" / "summary.json")
online = load_agg(run_root / "phase_online" / "summary.json")

def value(row, key):
    return float(row[f"{key}_mean"])

def pct(value, ref):
    return 100.0 * (value - ref) / ref

rows = []
for name, row in [
    ("epoch11_baseline", baseline),
    ("teacher_wait7_p30_g05", teacher),
    ("prev_awt6_attguard", prev["awt6_attguard"]),
    *sorted(online.items()),
]:
    awt = value(row, "test_avg_waiting_time_over")
    att = value(row, "test_avg_travel_time_over")
    aql = value(row, "test_avg_queue_len_over")
    p95 = value(row, "test_vehicle_wait_p95_over")
    max_wait = value(row, "test_vehicle_wait_max_over")
    rows.append({
        "policy": name,
        "seeds": row["seed_count"],
        "awt": awt,
        "att": att,
        "aql": aql,
        "p95": p95,
        "max_wait": max_wait,
        "awt_delta_pct": pct(awt, value(baseline, "test_avg_waiting_time_over")),
        "att_delta_pct": pct(att, value(baseline, "test_avg_travel_time_over")),
    })

comparison_csv = run_root / "phase_online" / "comparison_with_refs.csv"
with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

new_rows = [row for row in rows if row["policy"].startswith("stage")]
best = min(new_rows, key=lambda row: (row["awt"], row["att"])) if new_rows else None
all_best.mkdir(parents=True, exist_ok=True)
report = all_best / f"{run_root.name}_results.md"
lines = [
    "# NewYork28x7 AWT100 Teacher-Heavy Two-Stage",
    "",
    f"- Run: `{run_root}`",
    "- Protocol: same-flow NewYork 28x7, online seeds 3031/3032/3033/3034.",
    "- Stage A: teacher-heavy wait-first distillation.",
    "- Stage B: initialize from Stage A, then recover ATT using stronger ATT/base-KL guard.",
    "",
    "## Results",
    "",
    "| Policy | Seeds | AWT | ATT | AQL | P95 wait | Max wait | AWT vs baseline | ATT vs baseline |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['policy']} | {row['seeds']} | {row['awt']:.2f} | {row['att']:.2f} | "
        f"{row['aql']:.2f} | {row['p95']:.2f} | {row['max_wait']:.2f} | "
        f"{row['awt_delta_pct']:+.2f}% | {row['att_delta_pct']:+.2f}% |"
    )
if best:
    lines.extend([
        "",
        "## Current Best New Candidate",
        "",
        f"`{best['policy']}` has AWT `{best['awt']:.2f}` and ATT `{best['att']:.2f}`.",
    ])
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
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
      --memo "online_tests/newyork28x7_awt100_teacherheavy_twostage/${label}/seed_${seed}" \
      --projection_ckpt "$ckpt" \
      --llm_path "$MODEL_PATH" \
      --dataset newyork_28x7 \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --num_rounds 1 \
      --seed "$seed" \
      --proj_name "NewYork28x7-awt100-twostage-${label}-seed${seed}" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --traffic_series_interval 2 \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --bf16 \
      --result_json "$out_dir/result.json" \
      --disable_snapshots
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

train_policy() {
  local label="$1"; shift
  local init_ckpt="$1"; shift
  local epochs="$1"; shift
  local awt="$1"; shift
  local aql="$1"; shift
  local att="$1"; shift
  local tail="$1"; shift
  local rank="$1"; shift
  local base_kl="$1"; shift
  local out_dir="$RUN_ROOT/phase_train/$label/checkpoints"
  local log_path="$RUN_ROOT/logs/train_${label}.log"
  local data_paths=("$@")
  mkdir -p "$out_dir"
  write_status running "train_${label}"
  log_msg "Training ${label} epochs=${epochs} awt=${awt} att=${att} tail=${tail} base_kl=${base_kl}"
  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONUNBUFFERED=1 "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
      --data_path "${data_paths[@]}" \
      --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_CKPT" \
      --init_policy_ckpt "$init_ckpt" \
      --output_dir "$out_dir" \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --prompt_mode queue_wait_short \
      --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" \
      --epochs "$epochs" --lr "$STAGE2_LR" --min_lr "$STAGE2_MIN_LR" \
      --warmup_ratio 0.05 --lr_schedule cosine \
      --weight_decay 1e-4 --max_grad_norm 1.0 \
      --val_ratio 0.1 --split_strategy stratified --num_workers 4 \
      --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
      --ranking_loss_weight "$rank" --ranking_min_cost_gap 0.02 \
      --cost_awt_weight "$awt" --cost_aql_weight "$aql" --cost_att_weight "$att" --cost_tail_wait_weight "$tail" \
      --base_policy_kl_weight "$base_kl" \
      --resample_mode targeted \
      --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
      --resample_high_queue_threshold 99999 --resample_high_queue_weight 1 \
      --resample_enable_high_wait --resample_high_wait_threshold 40 --resample_high_wait_weight 4 \
      --resample_max_repeat 6 \
      --best_metric val_expected_cost_regret --best_mode min --best_min_delta 0.0 \
      --log_interval 20 --save_every 1 --checkpoint_slim --seed 2024 --bf16 \
      --ddp_timeout_seconds 1800
  ) > "$log_path" 2>&1 &
  local pid="$!"
  maybe_notify_progress "train_${label} started"
  wait_pids_progress "train_${label}" "$pid"
  require_file "$out_dir/best.pt"
  log_msg "Training complete for ${label}: ${out_dir}/best.pt"
}

eval_policy() {
  local label="$1"
  local ckpt="$RUN_ROOT/phase_train/$label/checkpoints/best.pt"
  require_file "$ckpt"
  write_status running "eval_${label}"
  log_msg "Evaluating ${label} on same NewYork 28x7 flow"
  read -r -a eval_gpu_array <<< "$EVAL_GPUS"
  local pids=()
  local job_index=0
  for seed in $EVAL_SEEDS; do
    local gpu="${eval_gpu_array[$((job_index % ${#eval_gpu_array[@]}))]}"
    launch_online "$gpu" "$label" "$seed" "$ckpt" \
      "$RUN_ROOT/phase_online/$label/seed_${seed}" \
      "$RUN_ROOT/logs/online_${label}_seed_${seed}.log"
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
  summarize_tree "$RUN_ROOT/phase_online" "$RUN_ROOT/phase_online/summary.json" "$RUN_ROOT/phase_online/summary.csv"
  log_msg "Evaluation summarized for ${label}: $(latest_result_line)"
}

finish() {
  local code="$?"
  if [[ "$code" -eq 0 ]]; then
    write_status success complete
    notify "NewYork28x7 teacher-heavy two-stage complete: run=${RUN_ROOT}; $(latest_result_line)"
  else
    write_status failed "exit_${code}"
    notify "NewYork28x7 teacher-heavy two-stage failed: exit=${code}; run=${RUN_ROOT}; inspect ${CAMPAIGN_LOG}"
  fi
}
trap finish EXIT

mkdir -p "$RUN_ROOT"/{logs,phase_train,phase_online,phase_data}
mkdir -p "$ALL_BEST_DIR"
touch "$CAMPAIGN_LOG"
write_status running preflight
require_file "$BASELINE_CKPT"
require_file "$STAGE1_CKPT"
require_file "$BASE_DATA"
require_file "$TEACHER_RUN/phase1_checkpoint_eval/summary.json"
require_file "$TEACHER_RUN/phase2_override_grid/summary.json"
require_file "$PREV_STUDENT_RUN/phase3_online/summary.json"
for trigger_file in "${TRIGGER_FILES[@]}"; do
  require_file "$trigger_file"
done

teacher_heavy_data=("$BASE_DATA")
for _ in 1 2 3 4 5 6; do
  teacher_heavy_data+=("${TRIGGER_FILES[@]}")
done
recover_data=("$BASE_DATA" "${TRIGGER_FILES[@]}" "${TRIGGER_FILES[@]}")

{
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'teacher_run=%s\n' "$TEACHER_RUN"
  printf 'prev_student_run=%s\n' "$PREV_STUDENT_RUN"
  printf 'baseline_ckpt=%s\n' "$BASELINE_CKPT"
  printf 'stage1_ckpt=%s\n' "$STAGE1_CKPT"
  printf 'train_gpus=%s\n' "$TRAIN_GPUS"
  printf 'train_nproc=%s\n' "$TRAIN_NPROC"
  printf 'eval_gpus=%s\n' "$EVAL_GPUS"
  printf 'eval_seeds=%s\n' "$EVAL_SEEDS"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'run_counts=%s\n' "$RUN_COUNTS"
  printf 'stage_a_epochs=%s\n' "$STAGE_A_EPOCHS"
  printf 'stage_b_epochs=%s\n' "$STAGE_B_EPOCHS"
  printf 'teacher_trigger_repeats=6\n'
  printf 'recover_trigger_repeats=2\n'
  printf 'protocol=same-flow NewYork28x7\n'
} > "$RUN_ROOT/manifest.env"

log_msg "Teacher-heavy two-stage run initialized at ${RUN_ROOT}"
notify "NewYork28x7 teacher-heavy two-stage started: run=${RUN_ROOT}; train_gpus=${TRAIN_GPUS}; eval_gpus=${EVAL_GPUS}"

train_policy stageA_wait_teacherheavy "$BASELINE_CKPT" "$STAGE_A_EPOCHS" 8.0 0.0 0.2 4.0 0.08 0.0 "${teacher_heavy_data[@]}"
eval_policy stageA_wait_teacherheavy

stage_a_ckpt="$RUN_ROOT/phase_train/stageA_wait_teacherheavy/checkpoints/best.pt"
train_policy stageB_att_recover_mild "$stage_a_ckpt" "$STAGE_B_EPOCHS" 5.0 0.0 2.0 2.0 0.05 0.20 "${recover_data[@]}"
eval_policy stageB_att_recover_mild

train_policy stageB_att_recover_strong "$stage_a_ckpt" "$STAGE_B_EPOCHS" 4.0 0.0 3.0 1.5 0.04 0.35 "${recover_data[@]}"
eval_policy stageB_att_recover_strong

write_report | tee -a "$CAMPAIGN_LOG"
log_msg "Teacher-heavy two-stage run completed"
