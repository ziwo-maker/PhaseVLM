#!/usr/bin/env bash
# Validate the Jinan-2500 H30/epoch-2 candidate across training and simulator seeds.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TRAIN_SEEDS_RAW="${TRAIN_SEEDS:-2024 42 123}"
SIM_SEEDS_RAW="${SIM_SEEDS:-2024 42 123 7 2026}"
RUN_ID="${RUN_ID:-jinan2500_h30_epoch2_3trainx5sim_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/paper_validation/$RUN_ID}"
FLOW="anon_3_4_jinan_real_2500.json"
RUN_COUNTS="${RUN_COUNTS:-3600}"
FEISHU_SCRIPT="${FEISHU_SCRIPT:-$REPO_ROOT/scripts/feishu_push.py}"

DATA1="$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DATA2="$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
FLOW_PATH="$REPO_ROOT/data/Jinan/3_4/$FLOW"
ROADNET="$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
MONITOR_PID=""

IFS=',' read -r -a TRAIN_GPUS <<< "$GPU_LIST"
read -r -a TRAIN_SEEDS <<< "$TRAIN_SEEDS_RAW"
read -r -a SIM_SEEDS <<< "$SIM_SEEDS_RAW"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

notify_feishu() {
  [[ -s "$FEISHU_SCRIPT" ]] || return 0
  timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" >/dev/null 2>&1 || true
}

progress_message() {
  local trained candidates online
  trained="$(find "$RUN_ROOT" -path '*/stage2/epoch_4.pt' -type f 2>/dev/null | wc -l)"
  candidates="$(find "$RUN_ROOT" -path '*/counterfactual_h30/training/responsive_h30/epoch_2.pt' -type f 2>/dev/null | wc -l)"
  online="$(find "$RUN_ROOT" -path '*/online/simulator_seed_*/result.json' -type f 2>/dev/null | wc -l)"
  printf 'Jinan-2500 H30 epoch-2 3-train x 5-simulator validation is running. stage2_epoch4=%s/3 h30_epoch2=%s/3 online=%s/15 run=%s' \
    "$trained" "$candidates" "$online" "$RUN_ROOT"
}

start_progress_monitor() {
  (
    while true; do
      sleep 2400
      notify_feishu "$(progress_message)"
    done
  ) &
  MONITOR_PID="$!"
}

finish_notification() {
  local status="$1"
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [[ "$status" -eq 0 && -s "$RUN_ROOT/aggregate.json" ]]; then
    local summary
    summary="$(jq -r '.overall | "AWT=\(.AWT) ATT=\(.ATT) AQL=\(.AQL) n=\(.n)"' "$RUN_ROOT/aggregate.json" 2>/dev/null || true)"
    notify_feishu "Jinan-2500 H30 epoch-2 3-train x 5-simulator validation completed. ${summary:-aggregate available}. run=$RUN_ROOT"
  elif [[ "$status" -ne 0 ]]; then
    notify_feishu "Jinan-2500 H30 epoch-2 3-train x 5-simulator validation failed with exit=$status. $(progress_message)."
  fi
}

require_file() {
  [[ -s "$1" ]] || { log "Missing required file: $1"; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "Missing required directory: $1"; exit 1; }
}

require_exact_matrix() {
  [[ "${#TRAIN_GPUS[@]}" -eq 4 ]] || { log "GPU_LIST must contain four GPU ids: $GPU_LIST"; exit 1; }
  [[ "${#TRAIN_SEEDS[@]}" -eq 3 ]] || { log "TRAIN_SEEDS must contain exactly three values"; exit 1; }
  [[ "${#SIM_SEEDS[@]}" -eq 5 ]] || { log "SIM_SEEDS must contain exactly five values"; exit 1; }
}

write_static_manifest() {
  RUN_ROOT="$RUN_ROOT" RUN_ID="$RUN_ID" GPU_LIST="$GPU_LIST" TRAIN_SEEDS_RAW="$TRAIN_SEEDS_RAW" \
    SIM_SEEDS_RAW="$SIM_SEEDS_RAW" MODEL_PATH="$MODEL_PATH" FLOW="$FLOW" RUN_COUNTS="$RUN_COUNTS" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone

manifest = {
    "schema": "jinan2500_h30_epoch2_3trainx5sim_v1",
    "run_id": os.environ["RUN_ID"],
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scope": "same_flow",
    "flow": os.environ["FLOW"],
    "run_counts": int(os.environ["RUN_COUNTS"]),
    "training_seeds": [int(x) for x in os.environ["TRAIN_SEEDS_RAW"].split()],
    "simulator_seeds": [int(x) for x in os.environ["SIM_SEEDS_RAW"].split()],
    "gpus": [int(x) for x in os.environ["GPU_LIST"].split(",")],
    "model_path": os.environ["MODEL_PATH"],
    "architecture": {
        "seq_len": 16,
        "image_size": 224,
        "periodicity": 16,
        "input_dim": 8,
        "hidden_dim": 64,
        "prefix_tokens": 8,
        "prefix_mode": "prefix",
        "max_length": 512,
    },
    "stage1": {"epochs": 15, "batch_size_per_rank": 1, "grad_accum_steps": 16},
    "stage2": {
        "source_epoch": 4,
        "epochs": 15,
        "batch_size_per_rank": 1,
        "grad_accum_steps": 16,
    },
    "counterfactual": {
        "teacher": "policy_responsive_h30",
        "teacher_seed": 0,
        "collection_seed_per_training_run": True,
        "adaptation_epoch": 2,
        "epochs": 5,
        "batch_size": 1,
        "grad_accum_steps": 16,
    },
    "online": {
        "deterministic": True,
        "phase_overrides": False,
        "worker_count": 4,
        "jobs": 15,
    },
}
with open(os.path.join(os.environ["RUN_ROOT"], "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
PY
}

train_seed_pipeline() {
  local seed="$1"
  local seed_root="$RUN_ROOT/training_seed_${seed}"
  local stage1="$seed_root/stage1"
  local stage2="$seed_root/stage2"
  local seed_log="$LOG_DIR/training_seed_${seed}"

  mkdir -p "$stage1" "$stage2" "$seed_log"
  if [[ ! -s "$stage1/best.pt" ]]; then
    log "training_seed=$seed stage1 start on GPUs $GPU_LIST"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 \
      train_align.py \
      --data_path "$DATA1" "$DATA2" --model_path "$MODEL_PATH" --output_dir "$stage1" --decision_only \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix \
      --temperature 0.07 --batch_size 1 --grad_accum_steps 16 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 \
      --lr_schedule cosine --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
      --log_every_steps 50 --save_every_epochs 1 --seed "$seed" --bf16 \
      2>&1 | tee "$seed_log/stage1.log"
  fi
  require_file "$stage1/best.pt"

  if [[ ! -s "$stage2/epoch_4.pt" ]]; then
    log "training_seed=$seed stage2 start on GPUs $GPU_LIST with max_length=512 batch=1 grad_accum=16"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 \
      projection_train/train_projection_bc.py \
      --data_path "$DATA1" "$DATA2" --llm_path "$MODEL_PATH" --align_ckpt "$stage1/best.pt" --output_dir "$stage2" \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix \
      --max_length 512 --prompt_mode queue_wait_short --batch_size 1 --grad_accum_steps 16 --epochs 15 \
      --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
      --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --num_workers 4 \
      --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
      --best_metric val_top1_cost_regret --best_mode min --best_min_delta 0.0 \
      --log_interval 20 --save_every 1 --seed "$seed" --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
      2>&1 | tee "$seed_log/stage2.log"
  fi
  require_file "$stage2/epoch_4.pt"
  sha256sum "$stage1/best.pt" "$stage2/epoch_4.pt" > "$seed_root/checkpoint_sha256.txt"
  log "training_seed=$seed complete"
}

prepare_h30_candidate() {
  local seed="$1"
  local gpu="$2"
  local seed_root="$RUN_ROOT/training_seed_${seed}"
  local stage1="$seed_root/stage1/best.pt"
  local source="$seed_root/stage2/epoch_4.pt"
  local cf_root="$seed_root/counterfactual_h30"
  local collected="$cf_root/collect/collected_states.jsonl"
  local labels="$cf_root/relabel/responsive_h30.jsonl"
  local candidate="$cf_root/training/responsive_h30/epoch_2.pt"

  mkdir -p "$cf_root/collect" "$cf_root/relabel" "$cf_root/training/responsive_h30" "$LOG_DIR/training_seed_${seed}"
  if [[ ! -s "$collected" ]]; then
    log "training_seed=$seed collect start on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
      --memo "$RUN_ID/training_seed_${seed}/collect" --projection_ckpt "$source" --llm_path "$MODEL_PATH" \
      --dataset jinan --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 \
      --inference_batch_size 64 --bf16 --deterministic --collect_states_output "$collected" \
      > "$LOG_DIR/training_seed_${seed}/collect.log" 2>&1
  fi
  require_file "$collected"

  if [[ ! -s "$labels" ]]; then
    log "training_seed=$seed responsive-H30 relabel start on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
      --input "$collected" --output "$labels" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" --dataset jinan \
      --num_rows 3 --num_cols 4 --source_policy_ckpt "$source" --llm_path "$MODEL_PATH" --work_dir "$cf_root/runtime_h30" \
      --sim_horizon 30 --min_action_time 30 --traffic_series_interval 2 --cost_scope global --neighbor-mode responsive \
      --cost_att_weight 1 --cost_aql_weight 1 --cost_awt_weight 1 --cost_spillback_weight 0.2 \
      --inference_batch_size 16 --bf16 --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
      > "$LOG_DIR/training_seed_${seed}/relabel_h30.log" 2>&1
  fi
  require_file "$labels"

  if [[ ! -s "$candidate" ]]; then
    log "training_seed=$seed responsive-H30 adaptation start on GPU $gpu with batch=1 grad_accum=16"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/train_projection_bc.py \
      --data_path "$labels" --llm_path "$MODEL_PATH" --align_ckpt "$stage1" --init_policy_ckpt "$source" \
      --output_dir "$cf_root/training/responsive_h30" --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix \
      --max_length 512 --batch_size 1 --grad_accum_steps 16 --epochs 5 --lr 1e-5 --min_lr 1e-6 \
      --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
      --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --num_workers 4 \
      --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
      --best_metric val_top1_cost_regret --best_mode min --log_interval 20 --save_every 1 --seed "$seed" \
      --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
      > "$LOG_DIR/training_seed_${seed}/train_responsive_h30.log" 2>&1
  fi
  require_file "$candidate"
  sha256sum "$candidate" > "$cf_root/candidate_sha256.txt"
  log "training_seed=$seed H30 epoch-2 candidate complete"
}

build_online_queue() {
  local jobs="$STATUS_DIR/online_jobs.tsv"
  : > "$jobs"
  local seed sim checkpoint
  for seed in "${TRAIN_SEEDS[@]}"; do
    checkpoint="$RUN_ROOT/training_seed_${seed}/counterfactual_h30/training/responsive_h30/epoch_2.pt"
    require_file "$checkpoint"
    for sim in "${SIM_SEEDS[@]}"; do
      printf '%s\t%s\t%s\n' "$seed" "$sim" "$checkpoint" >> "$jobs"
    done
  done
  printf '1\n' > "$STATUS_DIR/online_next_index"
  : > "$STATUS_DIR/online_queue.lock"
}

online_worker() {
  local gpu="$1"
  local jobs="$STATUS_DIR/online_jobs.tsv"
  local next="$STATUS_DIR/online_next_index"
  local lock="$STATUS_DIR/online_queue.lock"
  local line index training_seed sim_seed checkpoint output result

  while true; do
    exec 9>"$lock"
    flock -x 9
    index="$(<"$next")"
    line="$(sed -n "${index}p" "$jobs")"
    if [[ -n "$line" ]]; then
      printf '%s\n' "$((index + 1))" > "$next"
    fi
    flock -u 9
    exec 9>&-
    [[ -n "$line" ]] || break

    IFS=$'\t' read -r training_seed sim_seed checkpoint <<< "$line"
    output="$RUN_ROOT/training_seed_${training_seed}/online/simulator_seed_${sim_seed}"
    result="$output/result.json"
    mkdir -p "$output"
    if [[ -s "$result" ]]; then
      log "online worker gpu=$gpu skip existing training_seed=$training_seed simulator_seed=$sim_seed"
      continue
    fi
    log "online worker gpu=$gpu start training_seed=$training_seed simulator_seed=$sim_seed"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
      --memo "$RUN_ID/training_seed_${training_seed}/simulator_seed_${sim_seed}" --projection_ckpt "$checkpoint" --llm_path "$MODEL_PATH" \
      --dataset jinan --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$sim_seed" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 \
      --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
      --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
      --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
      --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
      --result_json "$result" > "$LOG_DIR/online_gpu${gpu}_training${training_seed}_sim${sim_seed}.log" 2>&1
    require_file "$result"
    log "online worker gpu=$gpu complete training_seed=$training_seed simulator_seed=$sim_seed"
  done
}

write_aggregate() {
  RUN_ROOT="$RUN_ROOT" TRAIN_SEEDS_RAW="$TRAIN_SEEDS_RAW" SIM_SEEDS_RAW="$SIM_SEEDS_RAW" "$PYTHON_BIN" - <<'PY'
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
training_seeds = [int(x) for x in os.environ["TRAIN_SEEDS_RAW"].split()]
simulator_seeds = [int(x) for x in os.environ["SIM_SEEDS_RAW"].split()]
metric_keys = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
    "queued": "test_queuing_vehicle_num_over",
    "reward": "test_reward_over",
    "p95_wait": "test_vehicle_wait_p95_over",
    "max_wait": "test_vehicle_wait_max_over",
}
rows = []
for training_seed in training_seeds:
    for simulator_seed in simulator_seeds:
        path = root / f"training_seed_{training_seed}" / "online" / f"simulator_seed_{simulator_seed}" / "result.json"
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload["results"]
        rows.append({
            "training_seed": training_seed,
            "simulator_seed": simulator_seed,
            **{name: float(values[key]) for name, key in metric_keys.items()},
            "result_path": str(path),
        })

def summarize(group):
    return {
        "n": len(group),
        **{name: statistics.fmean(row[name] for row in group) for name in metric_keys},
        **{f"{name}_std": statistics.pstdev(row[name] for row in group) if len(group) > 1 else 0.0 for name in metric_keys},
    }

by_training_seed = []
for seed in training_seeds:
    by_training_seed.append({"training_seed": seed, **summarize([row for row in rows if row["training_seed"] == seed])})

reference = {"ATT": 255.85311248634875, "AQL": 127.06666666666666, "AWT": 40.97650171944849}
overall = summarize(rows)
aggregate = {
    "reference_archived_h30_epoch2_3seed": reference,
    "per_job": rows,
    "by_training_seed": by_training_seed,
    "overall": overall,
    "delta_from_reference": {name: overall[name] - value for name, value in reference.items()},
    "selection": {
        "pre_registered_checkpoint": "responsive_h30/epoch_2",
        "primary_metric": "AWT",
        "tie_breakers": ["ATT", "AQL"],
    },
}
with open(root / "aggregate.json", "w", encoding="utf-8") as handle:
    json.dump(aggregate, handle, indent=2)
with open(root / "status" / "completed.txt", "w", encoding="utf-8") as handle:
    handle.write(f"completed_jobs={len(rows)}\n")
    handle.write(f"overall_AWT={overall['AWT']:.6f}\n")
    handle.write(f"overall_ATT={overall['ATT']:.6f}\n")
    handle.write(f"overall_AQL={overall['AQL']:.6f}\n")
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN"
  require_file "$REPO_ROOT/train_align.py"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_file "$REPO_ROOT/scripts/relabel_policy_responsive.py"
  require_file "$DATA1"
  require_file "$DATA2"
  require_file "$FLOW_PATH"
  require_file "$ROADNET"
  require_dir "$MODEL_PATH"
  require_exact_matrix
  mkdir -p "$RUN_ROOT" "$LOG_DIR" "$STATUS_DIR"
  write_static_manifest
  trap 'status=$?; finish_notification "$status"' EXIT
  notify_feishu "Jinan-2500 H30 epoch-2 3-train x 5-simulator validation started. max_length=512, batch_size=1, grad_accum_steps=16, four-GPU DDP training, then four online workers. run=$RUN_ROOT"
  start_progress_monitor
  log "run_id=$RUN_ID run_root=$RUN_ROOT"
  log "phase=training training_seeds=$TRAIN_SEEDS_RAW gpus=$GPU_LIST"
  local seed
  for seed in "${TRAIN_SEEDS[@]}"; do
    train_seed_pipeline "$seed"
  done

  log "phase=counterfactual_h30 start three independent seed workers"
  local pids=() index=0 gpu
  for seed in "${TRAIN_SEEDS[@]}"; do
    gpu="${TRAIN_GPUS[$index]}"
    prepare_h30_candidate "$seed" "$gpu" > "$LOG_DIR/h30_seed_${seed}_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { log "one or more H30 candidate pipelines failed"; exit 1; }

  build_online_queue
  log "phase=online start 4 GPU workers for 15 jobs"
  pids=()
  for gpu in "${TRAIN_GPUS[@]}"; do
    online_worker "$gpu" > "$LOG_DIR/online_worker_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { log "one or more online workers failed"; exit 1; }
  write_aggregate
  log "completed run_root=$RUN_ROOT"
}

main "$@"
