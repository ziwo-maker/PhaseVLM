#!/usr/bin/env bash
set -Eeuo pipefail

# Ablation B: Pure BC on qwctx SimExpert data — reproduce baseline
# Hypothesis: Pure BC on qwctx data should match original qwctx result (Sum~2822)
# This confirms data quality is the root cause of DAgger regression

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_ROOT" >&2
    exit 2
fi

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/lijn99/miniconda3/envs/lab/bin/python
TORCHRUN_BIN=/home/lijn99/miniconda3/envs/lab/bin/torchrun
MODEL_PATH=/home/data/model/qwen-vl-8B
DATA_PATH=/mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857/raw/traffic_prompt_series_2s.jsonl
ALIGN_CKPT=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_double_qwctx_20260731_040000/stage1/best.pt
RUN_ROOT="$1"
LOG_DIR="$RUN_ROOT/logs"
STATUS_PATH="$RUN_ROOT/completion.status"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"

on_exit() {
    local status="$?"
    if [[ "$status" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\nstage2_best=%s\n' "$(date -Is)" "$STAGE2_DIR/best.pt" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\n' "$(date -Is)" "$status" > "$STATUS_PATH"
    fi
    exit "$status"
}

check_preflight() {
    [[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; return 1; }
    [[ -s "$DATA_PATH" ]] || { echo "Missing data: $DATA_PATH" >&2; return 1; }
    [[ -s "$ALIGN_CKPT" ]] || { echo "Missing Stage 1: $ALIGN_CKPT" >&2; return 1; }
    for gpu in 0 1; do
        local used_memory
        used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
        [[ -n "$used_memory" && "$used_memory" -lt 500 ]] || { echo "GPU $gpu is busy: ${used_memory:-unknown} MiB" >&2; return 1; }
    done
}

check_preflight
mkdir -p "$LOG_DIR" "$STAGE2_DIR"
exec > "$LOG_DIR/pipeline.log" 2>&1
trap on_exit EXIT
{
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'status=running\n'
    printf 'experiment=ablation_b_pure_bc_qwctx_data\n'
    printf 'data=%s\n' "$DATA_PATH"
    printf 'stage1=%s\n' "$ALIGN_CKPT"
    printf 'target_mode=hard\n'
} > "$STATUS_PATH"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$TORCHRUN_BIN" --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
    --data_path "$DATA_PATH" \
    --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" \
    --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
    --batch_size 4 --grad_accum_steps 8 --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 \
    --log_interval 20 --save_every 1 --best_metric val_acc --best_mode max \
    --target_mode hard \
    --ddp_timeout_seconds 1800

printf 'stage2_completed_at=%s\n' "$(date -Is)" >> "$STATUS_PATH"
