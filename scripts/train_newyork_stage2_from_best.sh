#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
REPO_ROOT="/home/code/LLMSTSC"
TORCHRUN="/home/lijn99/miniconda3/envs/lab/bin/torchrun"
MODEL_PATH="/home/data/model/qwen-vl-8B"
DOUBLE_DATA="/mnt/data/LLMSTSC/records/sim_expert_validation_search/newyork28x7_full3600_archive_aggregate_grid_20260625_070657/best/traffic_prompt_series_2s.jsonl"
TRIPLE_DATA="/mnt/data/LLMSTSC/generated/newyork28x7_triple_stage2_raw_20260729_005955/raw/traffic_prompt_series_2s.jsonl"
ALIGN_CKPT="$RUN_ROOT/stage1/best.pt"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"

for path in "$TORCHRUN" "$MODEL_PATH" "$DOUBLE_DATA" "$TRIPLE_DATA" "$ALIGN_CKPT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT/logs" "$STAGE2_DIR"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "stage=stage2"
echo "run_root=$RUN_ROOT"
echo "started_at=$(date --iso-8601=seconds)"
echo "align_ckpt=$ALIGN_CKPT"
sha256sum "$ALIGN_CKPT"

exec "$TORCHRUN" --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
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
  --ddp_timeout_seconds 1800
