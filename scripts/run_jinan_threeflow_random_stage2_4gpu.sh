#!/usr/bin/env bash
# Train three independent Jinan Stage 2 policies from random projector/policy
# initialization, then evaluate each policy only on its matching traffic flow.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/home/data/model/qwen-vl-8B}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/LLMSTSCrecords/records/sim_expert_validation_search}"
RUN_ID="${RUN_ID:-jinan_threeflow_random_stage2_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/mnt/data/LLMSTSCrecords/records/pipeline_runs/$RUN_ID}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
SEEDS=(2024 42 123 7)
EVAL_GPUS=(0 1 2 3)
FLOW_KEYS=(jinan_real jinan_real_2000 jinan_real_2500)
TRAFFIC_FILES=(
  anon_3_4_jinan_real.json
  anon_3_4_jinan_real_2000.json
  anon_3_4_jinan_real_2500.json
)
DATA_DIRS=(
  anon_3_4_jinan_real
  anon_3_4_jinan_real_2000
  anon_3_4_jinan_real_2500
)
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
CURRENT_PHASE="$RUN_ROOT/current_phase.txt"
NOTIFIER_PID=""

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp() { date --iso-8601=seconds; }

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

notify() {
  local message="$1"
  [[ -f "$FEISHU_SCRIPT" ]] || return 0
  timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

latest_progress() {
  local phase metrics result
  phase="$(sed -n '1p' "$CURRENT_PHASE" 2>/dev/null || printf preflight)"
  metrics="$(find "$RUN_ROOT" -name metrics.jsonl -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  result="$(find "$RUN_ROOT" -name result.json -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -n "$result" ]]; then
    "$PYTHON_BIN" - "$phase" "$result" <<'PY' 2>/dev/null || true
import json, sys
phase, path = sys.argv[1:]
values = json.load(open(path, encoding="utf-8")).get("results", {})
print("phase=%s latest_result=%s AQL=%s AWT=%s ATT=%s" % (
    phase, path,
    values.get("test_avg_queue_len_over", "NA"),
    values.get("test_avg_waiting_time_over", "NA"),
    values.get("test_avg_travel_time_over", "NA"),
))
PY
  elif [[ -n "$metrics" ]]; then
    printf 'phase=%s latest_metrics=%s %s\n' "$phase" "$metrics" "$(tail -n 1 "$metrics" | cut -c1-1000)"
  else
    printf 'phase=%s no_metrics_yet\n' "$phase"
  fi
}

start_notifier() {
  (
    while sleep 2400; do
      notify "Jinan three-flow random Stage 2 progress: $(latest_progress)"
    done
  ) &
  NOTIFIER_PID="$!"
}

stop_notifier() {
  if [[ -n "$NOTIFIER_PID" ]]; then
    kill "$NOTIFIER_PID" 2>/dev/null || true
    wait "$NOTIFIER_PID" 2>/dev/null || true
    NOTIFIER_PID=""
  fi
}

on_exit() {
  local code="$?"
  stop_notifier
  if (( code != 0 )); then
    printf 'state=failed\nexit_status=%s\nfailed_at=%s\nphase=%s\n' \
      "$code" "$(timestamp)" "$(sed -n '1p' "$CURRENT_PHASE" 2>/dev/null || printf unknown)" \
      > "$CAMPAIGN_STATUS"
    notify "Jinan three-flow random Stage 2 FAILED: $(latest_progress)"
  fi
}
trap on_exit EXIT

snapshot() {
  local label="$1" pid="$2" output="$3"
  {
    printf 'captured_at=%s\nlabel=%s\npid=%s\n' "$(timestamp)" "$label" "$pid"
    ps -o pid,ppid,stat,etime,cmd --forest -g "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null | head -n 80 || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  } > "$output"
}

monitor_startup() {
  local label="$1" pid="$2" monitor_dir="$3"
  local minute
  mkdir -p "$monitor_dir"
  snapshot "$label" "$pid" "$monitor_dir/start.txt"
  for minute in 1 2; do
    sleep 60
    snapshot "$label" "$pid" "$monitor_dir/minute_${minute}.txt"
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        log "$label completed during its startup window"
        return 10
      fi
      log "$label exited unsuccessfully during its startup window"
      return 1
    fi
  done
  return 0
}

require_file() {
  [[ -s "$1" ]] || { log "Missing required file: $1"; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "Missing required directory: $1"; return 1; }
}

validate_gpus() {
  local expected=(0 1 2 3) rows gpu used
  mapfile -t rows < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  ((${#rows[@]} >= 4)) || { log "Four visible GPUs are required"; return 1; }
  for gpu in "${expected[@]}"; do
    used="$(printf '%s\n' "${rows[@]}" | awk -F, -v target="$gpu" '$1 + 0 == target {gsub(/ /, "", $2); print $2}')"
    [[ "$used" =~ ^[0-9]+$ ]] || { log "Could not read GPU $gpu memory usage"; return 1; }
    (( used < 1000 )) || { log "GPU $gpu is not idle: ${used} MiB used"; return 1; }
  done
}

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$RUN_ID" "$MODEL_PATH" "$TRAIN_GPUS" \
    "$DATA_ROOT" "${TRAFFIC_FILES[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

output, run_id, model_path, train_gpus, data_root, *flows = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

datasets = []
for flow in flows:
    directory = flow.removesuffix(".json")
    path = Path(data_root) / directory / "best" / "traffic_prompt_series_2s.jsonl"
    config = Path(data_root) / directory / "best_config.json"
    datasets.append({
        "flow": flow,
        "training_data": str(path),
        "training_data_sha256": sha256(path),
        "best_config": str(config),
        "best_config_sha256": sha256(config),
        "evaluation_flow": flow,
    })
payload = {
    "schema": "jinan_threeflow_independent_random_stage2_v1",
    "run_id": run_id,
    "created_at": datetime.now().astimezone().isoformat(),
    "model_path": model_path,
    "init_mode": "random",
    "alignment_checkpoint": None,
    "training_execution": "sequential_independent_four_gpu_ddp",
    "training_gpus": train_gpus,
    "online_seeds": [2024, 42, 123, 7],
    "same_flow_evaluation": True,
    "datasets": datasets,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_training() {
  local key="$1" data_path="$2" task_root="$RUN_ROOT/tasks/$key"
  local stage2_dir="$task_root/stage2" log_path="$task_root/logs/stage2.log"
  local status_path="$task_root/training.status" pid monitor_rc=0
  mkdir -p "$stage2_dir" "$task_root/logs" "$task_root/monitor/training"
  if [[ -s "$stage2_dir/best.pt" ]] && rg -q '^state=completed$' "$status_path" 2>/dev/null; then
    log "Training already complete; skipping $key"
    return 0
  fi
  printf 'training:%s\n' "$key" > "$CURRENT_PHASE"
  printf 'state=running\nstarted_at=%s\ndata=%s\ninit_mode=random\n' \
    "$(timestamp)" "$data_path" > "$status_path"
  log "Starting four-GPU Stage 2 training for $key"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" \
    projection_train/train_projection_bc.py \
    --data_path "$data_path" \
    --llm_path "$MODEL_PATH" --init_mode random \
    --output_dir "$stage2_dir" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --prompt_mode queue_wait_short --action_eval_chunk_size 1 \
    --batch_size 1 --grad_accum_steps 4 \
    --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 \
    --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --best_metric val_top1_cost_regret --best_mode min --best_min_delta 0.0 \
    --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim \
    --ddp_timeout_seconds 1800 > "$log_path" 2>&1 &
  pid="$!"
  printf 'pid=%s\n' "$pid" >> "$status_path"
  monitor_startup "training:$key" "$pid" "$task_root/monitor/training" || monitor_rc="$?"
  if (( monitor_rc == 10 )); then
    :
  elif (( monitor_rc != 0 )); then
    printf 'state=failed\nexit_status=%s\nfailed_at=%s\n' "$monitor_rc" "$(timestamp)" > "$status_path"
    return 1
  elif wait "$pid"; then
    :
  else
    local code="$?"
    printf 'state=failed\nexit_status=%s\nfailed_at=%s\n' "$code" "$(timestamp)" > "$status_path"
    return "$code"
  fi
  require_file "$stage2_dir/best.pt"
  printf 'state=completed\ncompleted_at=%s\nbest_checkpoint=%s\nbest_sha256=%s\n' \
    "$(timestamp)" "$stage2_dir/best.pt" "$(sha256sum "$stage2_dir/best.pt" | awk '{print $1}')" \
    > "$status_path"
}

run_online_group() {
  local key="$1" flow="$2" task_root="$RUN_ROOT/tasks/$key"
  local checkpoint="$task_root/stage2/best.pt" group_status="$task_root/online.status"
  local pids=() logs=() results=() seed gpu index pid monitor_rc=0 failures=0
  if rg -q '^state=completed$' "$group_status" 2>/dev/null; then
    log "Online evaluation already complete; skipping $key"
    return 0
  fi
  require_file "$checkpoint"
  printf 'online:%s\n' "$key" > "$CURRENT_PHASE"
  printf 'state=running\nstarted_at=%s\nflow=%s\ncheckpoint=%s\n' \
    "$(timestamp)" "$flow" "$checkpoint" > "$group_status"
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="${EVAL_GPUS[$index]}"
    local output_dir="$task_root/online/seed_$seed"
    local result_json="$output_dir/result.json"
    local log_path="$task_root/logs/online_seed_${seed}_gpu_${gpu}.log"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" projection_train/run_projection_online.py \
      --memo "online_tests/$RUN_ID/$key/seed_$seed" \
      --projection_ckpt "$checkpoint" --llm_path "$MODEL_PATH" \
      --num_rounds 1 --proj_name "Jinan-$key-random-stage2" \
      --dataset jinan --traffic_file "$flow" --run_counts 3600 \
      --device cuda:0 --seed "$seed" --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 0 \
      --result_json "$result_json" --bf16 --deterministic --disable_snapshots \
      > "$log_path" 2>&1 &
    pid="$!"
    pids+=("$pid")
    logs+=("$log_path")
    results+=("$result_json")
    log "Started online $key seed=$seed gpu=$gpu pid=$pid"
  done
  mkdir -p "$task_root/monitor/online"
  snapshot "online:$key" "${pids[0]}" "$task_root/monitor/online/start.txt"
  for index in 1 2; do
    sleep 60
    snapshot "online:$key" "${pids[0]}" "$task_root/monitor/online/minute_${index}.txt"
    for pid in "${pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        log "Online worker pid=$pid exited during startup window; final status will be checked"
      fi
    done
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failures=$((failures + 1))
      log "Online worker failed: ${logs[$index]}"
    fi
    [[ -s "${results[$index]}" ]] || failures=$((failures + 1))
  done
  (( failures == 0 )) || {
    printf 'state=failed\nfailures=%s\nfailed_at=%s\n' "$failures" "$(timestamp)" > "$group_status"
    return 1
  }
  aggregate_online "$key" "$flow"
  printf 'state=completed\ncompleted_at=%s\nflow=%s\nsummary=%s\n' \
    "$(timestamp)" "$flow" "$task_root/online_summary.json" > "$group_status"
}

aggregate_online() {
  local key="$1" flow="$2" task_root="$RUN_ROOT/tasks/$key"
  "$PYTHON_BIN" - "$task_root" "$flow" "${SEEDS[@]}" <<'PY'
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

task_root, flow, *raw_seeds = sys.argv[1:]
task_root = Path(task_root)
rows = []
for raw_seed in raw_seeds:
    path = task_root / "online" / f"seed_{raw_seed}" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"seed": int(raw_seed), "result_json": str(path), **payload["results"]})
metric_names = sorted({key for row in rows for key in row if key.startswith("test_")})
aggregate = {}
for metric in metric_names:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    aggregate[metric] = {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }
output = {
    "schema": "jinan_sameflow_online_multiseed_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "training_flow": flow,
    "evaluation_flow": flow,
    "same_flow": True,
    "checkpoint": str(task_root / "stage2" / "best.pt"),
    "rows": rows,
    "aggregate": aggregate,
}
(task_root / "online_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
PY
}

aggregate_campaign() {
  "$PYTHON_BIN" - "$RUN_ROOT" "${FLOW_KEYS[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

root, *keys = sys.argv[1:]
root = Path(root)
tasks = {}
for key in keys:
    path = root / "tasks" / key / "online_summary.json"
    tasks[key] = json.loads(path.read_text(encoding="utf-8"))
output = {
    "schema": "jinan_threeflow_independent_random_stage2_aggregate_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "tasks": tasks,
}
(root / "aggregate.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
PY
}

main() {
  cd "$REPO_ROOT"
  mkdir -p "$RUN_ROOT/tasks"
  printf 'preflight\n' > "$CURRENT_PHASE"
  printf 'state=running\nstarted_at=%s\nrun_id=%s\nrun_root=%s\n' \
    "$(timestamp)" "$RUN_ID" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN"
  require_dir "$MODEL_PATH"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  local directory
  for directory in "${DATA_DIRS[@]}"; do
    require_file "$DATA_ROOT/$directory/best/traffic_prompt_series_2s.jsonl"
    require_file "$DATA_ROOT/$directory/best_config.json"
  done
  validate_gpus
  write_manifest
  start_notifier
  notify "Jinan three-flow random Stage 2 started: run_root=$RUN_ROOT"
  local index key flow data_path
  for index in "${!FLOW_KEYS[@]}"; do
    key="${FLOW_KEYS[$index]}"
    flow="${TRAFFIC_FILES[$index]}"
    data_path="$DATA_ROOT/${DATA_DIRS[$index]}/best/traffic_prompt_series_2s.jsonl"
    run_training "$key" "$data_path"
    run_online_group "$key" "$flow"
  done
  printf 'aggregate\n' > "$CURRENT_PHASE"
  aggregate_campaign
  printf 'state=completed\ncompleted_at=%s\naggregate=%s\n' \
    "$(timestamp)" "$RUN_ROOT/aggregate.json" > "$CAMPAIGN_STATUS"
  notify "Jinan three-flow random Stage 2 completed: run_root=$RUN_ROOT aggregate=$RUN_ROOT/aggregate.json"
  stop_notifier
  trap - EXIT
  log "Campaign complete: $RUN_ROOT"
}

main "$@"
