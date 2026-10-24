#!/usr/bin/env bash
# Train independent Jinan Stage 1 + Stage 2 policies, one per traffic flow,
# using the best sim-expert weights found in /mnt/data/LLMSTSCrecords.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/LLMSTSCrecords/records/sim_expert_validation_search}"
RUN_ID="${RUN_ID:-jinan_threeflow_bestweights_stage1_stage2_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/code/LLMSTSC/records/pipeline_runs/$RUN_ID}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
FEISHU_SCRIPT="${FEISHU_SCRIPT:-$REPO_ROOT/scripts/feishu_push.py}"
NOTIFY_INTERVAL_SEC="${NOTIFY_INTERVAL_SEC:-2400}"
FLOW_KEYS=(jinan_real_2000 jinan_real_2500)
DATA_DIRS=(
  anon_3_4_jinan_real_2000
  anon_3_4_jinan_real_2500
)
TRAFFIC_FILES=(
  anon_3_4_jinan_real_2000.json
  anon_3_4_jinan_real_2500.json
)
STAGE1_DATASETS=(
  align_train.stage1_compact_ds.jsonl
  align_train.stage1_compact_ds.jsonl
)
STAGE2_DATASETS=(
  traffic_prompt_series_2s.jsonl
  traffic_prompt_series_2s.jsonl
)
STAGE2_COST_WEIGHTS=(
  "1.0,1.0,2.0,0.2"
  "1.0,1.0,1.0,0.0"
)

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

require_file() {
  [[ -s "$1" ]] || { log "Missing required file: $1"; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "Missing required directory: $1"; return 1; }
}

notify() {
  [[ -s "$FEISHU_SCRIPT" ]] || return 0
  timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" >/dev/null 2>&1 || true
}

latest_metric() {
  local file
  file="$(find "$RUN_ROOT/tasks" -path '*/stage2/metrics.jsonl' -type f -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$file" && -s "$file" ]]; then
    tail -n 1 "$file"
  else
    printf 'no_stage2_metric_yet'
  fi
}

latest_progress() {
  {
    printf 'phase=%s\n' "$(cat "$RUN_ROOT/current_phase.txt" 2>/dev/null || printf unknown)"
    find "$RUN_ROOT/tasks" -maxdepth 2 -name training.status -type f -print 2>/dev/null | sort | while read -r status_file; do
      printf '%s: ' "$(basename "$(dirname "$status_file")")"
      awk -F= '/^(state|flow|stage1_pid|stage2_pid)=/ {printf "%s=%s ", $1, $2} END {print ""}' "$status_file"
    done
    printf 'latest_metric=%s\n' "$(latest_metric)"
  } | tr '\n' '; '
}

notify_loop() {
  while true; do
    sleep "$NOTIFY_INTERVAL_SEC"
    notify "Jinan 2000/2500 bestweights Stage1+2 progress: $(latest_progress) run_root=$RUN_ROOT"
  done
}

notify_pid=""

finish() {
  local code="$?"
  if [[ -n "$notify_pid" ]]; then
    kill "$notify_pid" >/dev/null 2>&1 || true
  fi
  if (( code == 0 )); then
    printf 'state=completed\ncompleted_at=%s\n' "$(timestamp)" > "$RUN_ROOT/campaign.status"
    notify "Jinan 2000/2500 bestweights Stage1+2 completed: $(latest_progress) run_root=$RUN_ROOT"
  else
    printf 'state=failed\nfailed_at=%s\nexit_status=%s\n' "$(timestamp)" "$code" > "$RUN_ROOT/campaign.status"
    notify "Jinan 2000/2500 bestweights Stage1+2 failed: exit=$code; $(latest_progress) run_root=$RUN_ROOT"
  fi
  return "$code"
}
trap finish EXIT

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
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$RUN_ID" "$MODEL_PATH" "$DATA_ROOT" "${#FLOW_KEYS[@]}" \
    "${FLOW_KEYS[@]}" "${DATA_DIRS[@]}" "${TRAFFIC_FILES[@]}" "${STAGE1_DATASETS[@]}" "${STAGE2_DATASETS[@]}" "${STAGE2_COST_WEIGHTS[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

output, run_id, model_path, data_root, flow_count_raw, *rest = sys.argv[1:]
flow_count = int(flow_count_raw)
flows = rest[:flow_count]
data_dirs = rest[flow_count:flow_count * 2]
traffic_files = rest[flow_count * 2:flow_count * 3]
stage1_files = rest[flow_count * 3:flow_count * 4]
stage2_files = rest[flow_count * 4:flow_count * 5]
stage2_cost_weights = rest[flow_count * 5:flow_count * 6]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

datasets = []
for flow, data_dir, traffic_file, stage1_file, stage2_file, cost_weight_str in zip(flows, data_dirs, traffic_files, stage1_files, stage2_files, stage2_cost_weights):
    directory = data_dir
    stage1_path = Path(data_root) / directory / "best" / stage1_file
    stage2_path = Path(data_root) / directory / "best" / stage2_file
    best_config = Path(data_root) / directory / "best_config.json"
    att_w, aql_w, awt_w, spill_w = [float(item) for item in cost_weight_str.split(",")]
    datasets.append({
        "flow": flow,
        "traffic_file": traffic_file,
        "stage1_data": str(stage1_path),
        "stage1_data_sha256": sha256(stage1_path),
        "stage2_data": str(stage2_path),
        "stage2_data_sha256": sha256(stage2_path),
        "best_config": str(best_config),
        "best_config_sha256": sha256(best_config),
        "stage2_cost_weights": {
            "att": att_w,
            "aql": aql_w,
            "awt": awt_w,
            "spillback": spill_w,
        },
    })

payload = {
    "schema": "jinan_twoflow_bestweights_stage1_stage2_v1",
    "run_id": run_id,
    "created_at": datetime.now().astimezone().isoformat(),
    "model_path": model_path,
    "training_execution": "sequential_independent_four_gpu_ddp",
    "training_gpus": os.environ.get("TRAIN_GPUS", "0,1,2,3"),
    "stage1_batch_size": 2,
    "stage1_grad_accum_steps": 16,
    "stage2_batch_size": 2,
    "stage2_grad_accum_steps": 16,
    "stage2_best_metric": "val_top1_cost_regret",
    "stage2_best_mode": "min",
    "same_flow_evaluation": False,
    "datasets": datasets,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_flow() {
  local key="$1" traffic_file="$2" stage1_data="$3" stage2_data="$4" cost_weights="$5"
  local task_root="$RUN_ROOT/tasks/$key"
  local stage1_dir="$task_root/stage1"
  local stage2_dir="$task_root/stage2"
  local logs_dir="$task_root/logs"
  local monitor_dir="$task_root/monitor"
  local training_status="$task_root/training.status"
  local pid monitor_rc=0
  local att_w aql_w awt_w spill_w

  IFS=',' read -r att_w aql_w awt_w spill_w <<< "$cost_weights"

  mkdir -p "$stage1_dir" "$stage2_dir" "$logs_dir" "$monitor_dir/stage1" "$monitor_dir/stage2"
  if [[ -s "$stage2_dir/best.pt" ]] && rg -q '^state=completed$' "$training_status" 2>/dev/null; then
    log "Flow already complete; skipping $key"
    return 0
  fi

  printf 'flow:%s\n' "$key" > "$RUN_ROOT/current_phase.txt"
  printf 'state=running\nstarted_at=%s\nflow=%s\nstage1_data=%s\nstage2_data=%s\nstage2_cost_weights=%s\n' \
    "$(timestamp)" "$key" "$stage1_data" "$stage2_data" "$cost_weights" > "$training_status"

  if [[ ! -s "$stage1_dir/best.pt" ]]; then
    log "Starting Stage 1 for $key"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" \
      train_align.py \
      --data_path "$stage1_data" \
      --model_path "$MODEL_PATH" \
      --output_dir "$stage1_dir" \
      --decision_only \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix \
      --temperature 0.05 \
      --batch_size 2 --grad_accum_steps 16 \
      --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
      --epochs 15 --num_workers 4 --val_ratio 0.1 \
      --eval_max_samples 1000 --log_every_steps 50 --save_every_epochs 1 \
      --seed 2024 --bf16 \
      > "$logs_dir/stage1.log" 2>&1 &
    pid="$!"
    printf 'stage1_pid=%s\n' "$pid" >> "$training_status"
    monitor_startup "stage1:$key" "$pid" "$monitor_dir/stage1" || monitor_rc="$?"
    if (( monitor_rc == 10 )); then
      :
    elif (( monitor_rc != 0 )); then
      printf 'state=failed\nexit_status=%s\nfailed_at=%s\n' "$monitor_rc" "$(timestamp)" > "$training_status"
      return 1
    elif wait "$pid"; then
      :
    else
      local code="$?"
      printf 'state=failed\nexit_status=%s\nfailed_at=%s\n' "$code" "$(timestamp)" > "$training_status"
      return "$code"
    fi
    require_file "$stage1_dir/best.pt"
  else
    log "Stage 1 best.pt already present for $key"
  fi

  if [[ ! -s "$stage2_dir/best.pt" ]]; then
    log "Starting Stage 2 for $key"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" \
      projection_train/train_projection_bc.py \
      --data_path "$stage2_data" \
      --llm_path "$MODEL_PATH" --align_ckpt "$stage1_dir/best.pt" \
      --output_dir "$stage2_dir" \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --prompt_mode queue_wait_short \
      --batch_size 2 --grad_accum_steps 16 \
      --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
      --weight_decay 1e-4 --max_grad_norm 1.0 \
      --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 \
      --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
      --cost_att_weight "$att_w" --cost_aql_weight "$aql_w" --cost_awt_weight "$awt_w" --cost_spillback_weight "$spill_w" \
      --best_metric val_top1_cost_regret --best_mode min --best_min_delta 0.0 \
      --log_interval 20 --save_every 1 --seed 2024 --bf16 \
      --checkpoint_slim --ddp_timeout_seconds 1800 \
      > "$logs_dir/stage2.log" 2>&1 &
    pid="$!"
    printf 'stage2_pid=%s\n' "$pid" >> "$training_status"
    monitor_startup "stage2:$key" "$pid" "$monitor_dir/stage2" || monitor_rc="$?"
    if (( monitor_rc == 10 )); then
      :
    elif (( monitor_rc != 0 )); then
      printf 'state=failed\nexit_status=%s\nfailed_at=%s\n' "$monitor_rc" "$(timestamp)" > "$training_status"
      return 1
    elif wait "$pid"; then
      :
    else
      local code="$?"
      printf 'state=failed\nexit_status=%s\nfailed_at=%s\n' "$code" "$(timestamp)" > "$training_status"
      return "$code"
    fi
    require_file "$stage2_dir/best.pt"
  else
    log "Stage 2 best.pt already present for $key"
  fi

  printf 'state=completed\ncompleted_at=%s\nstage1_checkpoint=%s\nstage2_checkpoint=%s\n' \
    "$(timestamp)" "$stage1_dir/best.pt" "$stage2_dir/best.pt" > "$training_status"
}

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

main() {
  cd "$REPO_ROOT"
  mkdir -p "$RUN_ROOT/tasks"
  printf 'preflight\n' > "$RUN_ROOT/current_phase.txt"
  printf 'state=running\nstarted_at=%s\nrun_id=%s\nrun_root=%s\n' \
    "$(timestamp)" "$RUN_ID" "$RUN_ROOT" > "$RUN_ROOT/campaign.status"
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN"
  require_dir "$MODEL_PATH"
  require_file "$REPO_ROOT/train_align.py"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  local directory
  for directory in "${DATA_DIRS[@]}"; do
    require_file "$DATA_ROOT/$directory/best/traffic_prompt_series_2s.jsonl"
    require_file "$DATA_ROOT/$directory/best_config.json"
  done
  validate_gpus
  write_manifest
  log "Launching Jinan 2000/2500 Stage 1 + Stage 2 campaign at $RUN_ROOT"
  notify "Jinan 2000/2500 bestweights Stage1+2 launched: flows=${FLOW_KEYS[*]}, gpus=$TRAIN_GPUS, batch_size=2, grad_accum_steps=16, prompt=queue_wait_short, run_root=$RUN_ROOT"
  notify_loop &
  notify_pid="$!"
  local index key data_dir traffic_file stage1_data cost_weights
  local stage2_data
  for index in "${!FLOW_KEYS[@]}"; do
    key="${FLOW_KEYS[$index]}"
    data_dir="${DATA_DIRS[$index]}"
    traffic_file="${TRAFFIC_FILES[$index]}"
    stage1_data="$DATA_ROOT/$data_dir/best/${STAGE1_DATASETS[$index]}"
    stage2_data="$DATA_ROOT/$data_dir/best/${STAGE2_DATASETS[$index]}"
    cost_weights="${STAGE2_COST_WEIGHTS[$index]}"
    log "Flow=$key data_dir=$data_dir traffic_file=$traffic_file stage1_data=$stage1_data stage2_data=$stage2_data"
    run_flow "$key" "$traffic_file" "$stage1_data" "$stage2_data" "$cost_weights"
  done
  printf 'state=completed\ncompleted_at=%s\n' "$(timestamp)" > "$RUN_ROOT/campaign.status"
  log "Campaign complete: $RUN_ROOT"
}

main "$@"
