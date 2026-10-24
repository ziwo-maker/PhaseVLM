#!/usr/bin/env bash
# Reproduce the jinan_2500 archived four-GPU Stage 1 -> Stage 2 recipe.
# The missing historical checkpoint is intentionally not reused: this trains
# both stages from scratch with the current /mnt/data Qwen3-VL-8B-Instruct model.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_ID="${RUN_ID:-jinan2500_stage1_stage2_4gpu_maxlen512_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/repro/$RUN_ID}"

DATA1="$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DATA2="$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
STAGE1_DIR="${STAGE1_DIR:-$RUN_ROOT/stage1}"
STAGE2_DIR="${STAGE2_DIR:-$RUN_ROOT/stage2}"
STAGE2_INIT_CKPT="${STAGE2_INIT_CKPT:-}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-16}"
LOG_DIR="$RUN_ROOT/logs"
MANIFEST="$RUN_ROOT/manifest.env"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

require_file() {
  [[ -s "$1" ]] || { log "Missing required file: $1"; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "Missing required directory: $1"; exit 1; }
}

validate_four_gpus() {
  local gpu_count gpu
  IFS=',' read -r -a TRAIN_GPUS <<< "$GPU_LIST"
  gpu_count="${#TRAIN_GPUS[@]}"
  [[ "$gpu_count" -eq 4 ]] || {
    log "GPU_LIST must contain exactly four comma-separated GPU ids; got: $GPU_LIST"
    exit 1
  }

  for gpu in "${TRAIN_GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { log "Invalid GPU id: $gpu"; exit 1; }
  done
  [[ "$NPROC_PER_NODE" -eq 4 ]] || {
    log "NPROC_PER_NODE must be 4 for this four-GPU recipe; got: $NPROC_PER_NODE"
    exit 1
  }
}

write_manifest() {
  umask 077
  {
    printf 'run_id=%q\n' "$RUN_ID"
    printf 'run_root=%q\n' "$RUN_ROOT"
    printf 'model_path=%q\n' "$MODEL_PATH"
    printf 'gpu_list=%q\n' "$GPU_LIST"
    printf 'stage1_data=%q\n' "$DATA1 $DATA2"
    printf 'stage2_data=%q\n' "$DATA1 $DATA2"
    printf 'traffic_file=%q\n' 'anon_3_4_jinan_real_2500.json'
    printf 'prompt_mode=%q\n' 'queue_wait_short'
    printf 'seq_len=%q\n' '16'
    printf 'image_size=%q\n' '224'
    printf 'periodicity=%q\n' '16'
    printf 'input_dim=%q\n' '8'
    printf 'hidden_dim=%q\n' '64'
    printf 'prefix_tokens=%q\n' '8'
    printf 'prefix_mode=%q\n' 'prefix'
    printf 'max_length=%q\n' '512'
    printf 'stage2_init_ckpt=%q\n' "$STAGE2_INIT_CKPT"
    printf 'stage2_batch_size=%q\n' "$STAGE2_BATCH_SIZE"
    printf 'stage2_grad_accum_steps=%q\n' "$STAGE2_GRAD_ACCUM_STEPS"
    printf 'seed=%q\n' '2024'
    printf 'epochs=%q\n' '15'
  } > "$MANIFEST"
}

run_stage1() {
  if [[ -s "$STAGE1_DIR/best.pt" ]]; then
    log "Reusing existing Stage 1 checkpoint: $STAGE1_DIR/best.pt"
    return
  fi

  log "Starting four-GPU Stage 1 alignment training"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 \
    train_align.py \
    --data_path "$DATA1" "$DATA2" \
    --model_path "$MODEL_PATH" \
    --output_dir "$STAGE1_DIR" \
    --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix \
    --temperature 0.07 \
    --batch_size 2 --grad_accum_steps 2 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 \
    2>&1 | tee "$LOG_DIR/stage1.log"
  require_file "$STAGE1_DIR/best.pt"
}

run_stage2() {
  if [[ -s "$STAGE2_DIR/best.pt" && -s "$STAGE2_DIR/epoch_15.pt" ]]; then
    log "Reusing existing Stage 2 checkpoints in $STAGE2_DIR"
    return
  fi

  log "Starting four-GPU Stage 2 BC training with max_length=512, batch_size=$STAGE2_BATCH_SIZE, grad_accum_steps=$STAGE2_GRAD_ACCUM_STEPS"
  local -a stage2_init_args=()
  if [[ -n "$STAGE2_INIT_CKPT" ]]; then
    require_file "$STAGE2_INIT_CKPT"
    stage2_init_args=(--init_policy_ckpt "$STAGE2_INIT_CKPT")
    log "Initializing Stage 2 trainable weights from: $STAGE2_INIT_CKPT"
  fi
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 \
    projection_train/train_projection_bc.py \
    --data_path "$DATA1" "$DATA2" \
    --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_DIR/best.pt" "${stage2_init_args[@]}" \
    --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --prompt_mode queue_wait_short \
    --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 \
    --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --best_metric val_top1_cost_regret --best_mode min --best_min_delta 0.0 \
    --log_interval 20 --save_every 1 --seed 2024 --bf16 \
    --checkpoint_slim --ddp_timeout_seconds 1800 \
    2>&1 | tee "$LOG_DIR/stage2.log"
  require_file "$STAGE2_DIR/best.pt"
  require_file "$STAGE2_DIR/epoch_15.pt"
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN"
  require_file "$REPO_ROOT/train_align.py"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  require_file "$DATA1"
  require_file "$DATA2"
  require_dir "$MODEL_PATH"
  validate_four_gpus
  mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$LOG_DIR"
  write_manifest
  log "Run root: $RUN_ROOT"
  log "Model: $MODEL_PATH; GPUs: $GPU_LIST; Stage 2 max_length: 512"
  run_stage1
  run_stage2
  log "Completed Stage 1 and Stage 2: $RUN_ROOT"
}

main "$@"
