#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
REPO_ROOT="/home/code/LLMSTSC"
PYTHON="/home/lijn99/miniconda3/envs/lab/bin/python"
TORCHRUN="/home/lijn99/miniconda3/envs/lab/bin/torchrun"
MODEL_PATH="/home/data/model/qwen-vl-8B"
DOUBLE_DATA="/mnt/data/LLMSTSC/records/sim_expert_validation_search/newyork28x7_full3600_archive_aggregate_grid_20260625_070657/best/traffic_prompt_series_2s.jsonl"
TRIPLE_DATA="/mnt/data/LLMSTSC/generated/newyork28x7_triple_stage2_raw_20260729_005955/raw/traffic_prompt_series_2s.jsonl"
STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"
LOG_DIR="$RUN_ROOT/logs"

notify_completion() {
  local status="$?"
  local completed_at
  local stage1_state="missing"
  local stage2_state="missing"
  local webhook=""

  completed_at="$(date --iso-8601=seconds)"
  [[ -s "$STAGE1_DIR/best.pt" ]] && stage1_state="checkpoint_ready"
  [[ -s "$STAGE2_DIR/best.pt" ]] && stage2_state="checkpoint_ready"

  mkdir -p "$LOG_DIR" 2>/dev/null || true
  {
    printf 'exit_status=%s\n' "$status"
    printf 'completed_at=%s\n' "$completed_at"
    printf 'stage1=%s\n' "$stage1_state"
    printf 'stage2=%s\n' "$stage2_state"
  } > "$LOG_DIR/completion.status" 2>/dev/null || true

  if [[ -r /root/.claude/feishu.json ]]; then
    webhook="$(sed -n 's/^[[:space:]]*"webhook_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' /root/.claude/feishu.json | sed -n '1p')"
  fi

  if [[ -n "$webhook" ]]; then
    curl --silent --show-error --connect-timeout 5 --max-time 10 --output /dev/null \
      --request POST "$webhook" \
      --header 'Content-Type: application/json' \
      --data "{\"msg_type\":\"interactive\",\"card\":{\"header\":{\"title\":{\"tag\":\"plain_text\",\"content\":\"NewYork Training Finished\"},\"template\":\"$( [[ "$status" -eq 0 ]] && printf 'green' || printf 'red' )\"},\"elements\":[{\"tag\":\"markdown\",\"content\":\"NewYork Stage 1 + Stage 2 training exited with status $status at $completed_at. Stage 1: $stage1_state. Stage 2: $stage2_state. Run: $RUN_ROOT\"}]}}" \
      || true
  fi

  trap - EXIT
  exit "$status"
}

trap notify_completion EXIT

for required_path in "$PYTHON" "$TORCHRUN" "$MODEL_PATH" "$DOUBLE_DATA" "$TRIPLE_DATA"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done

if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$STAGE1_DIR" "$STAGE2_DIR"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "run_root=$RUN_ROOT"
echo "started_at=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "double_data=$DOUBLE_DATA"
echo "triple_data=$TRIPLE_DATA"
echo "stage1_epochs=15"
echo "stage2_epochs=15"

"$TORCHRUN" --standalone --nproc_per_node=2 train_align.py \
  --data_path "$DOUBLE_DATA" "$TRIPLE_DATA" \
  --model_path "$MODEL_PATH" \
  --output_dir "$STAGE1_DIR" \
  --decision_only \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix \
  --temperature 0.07 \
  --batch_size 4 --grad_accum_steps 2 \
  --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --epochs 15 --num_workers 4 --val_ratio 0.1 \
  --eval_max_samples 2000 --log_every_steps 50 --save_every_epochs 5 \
  --seed 2024 --bf16 \
  2>&1 | tee "$LOG_DIR/stage1.log"

ALIGN_CKPT="$STAGE1_DIR/best.pt"
if [[ ! -s "$ALIGN_CKPT" ]]; then
  echo "Stage 1 completed without a valid checkpoint: $ALIGN_CKPT" >&2
  exit 1
fi

"$TORCHRUN" --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
  --data_path "$DOUBLE_DATA" "$TRIPLE_DATA" \
  --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" \
  --output_dir "$STAGE2_DIR" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode image_only_short \
  --batch_size 4 --grad_accum_steps 4 \
  --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --val_ratio 0.1 --split_strategy blocked --num_workers 4 \
  --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
  --best_metric val_macro_f1 --best_mode auto --best_min_delta 0.0 \
  --log_interval 20 --save_every 1 --seed 2024 --bf16 \
  --ddp_timeout_seconds 1800 \
  2>&1 | tee "$LOG_DIR/stage2.log"

echo "finished_at=$(date --iso-8601=seconds)"
