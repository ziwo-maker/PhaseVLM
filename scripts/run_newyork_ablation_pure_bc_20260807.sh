#!/usr/bin/env bash
set -Eeuo pipefail

# Ablation A: Pure BC on DAgger data — isolate data quality vs loss function
# Hypothesis: If pure BC on DAgger data matches qwctx baseline (Sum~2822),
#   then the DAgger data is fine and the cost ranking + KL losses are the problem.
#   If pure BC is still bad, the DAgger data itself is the root cause.

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_ROOT" >&2
    exit 2
fi

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/lijn99/miniconda3/envs/lab/bin/python
TORCHRUN_BIN=/home/lijn99/miniconda3/envs/lab/bin/torchrun
MODEL_PATH=/home/data/model/qwen-vl-8B
DATA_PATH=/home/code/LLMSTSC/records/dagger/newyork28x7_double_awt_priority_20260806_workers4/stage2_train.jsonl
ALIGN_CKPT=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_dagger_costrank_20260806_run01/stage1/best.pt
RUN_ROOT="$1"
LOG_DIR="$RUN_ROOT/logs"
STATUS_PATH="$RUN_ROOT/completion.status"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"

notify_feishu() {
    local title="$1"
    local body="$2"
    local config=/root/.claude/feishu.json
    local mode webhook payload

    [[ -r "$config" ]] || return 0
    mode="$(jq -r '.mode // "off"' "$config" 2>/dev/null || true)"
    [[ "$mode" == push ]] || return 0
    webhook="$(jq -r '.webhook_url // ""' "$config" 2>/dev/null || true)"
    [[ -n "$webhook" ]] || return 0
    payload="$(jq -n --arg title "$title" --arg body "$body" '{msg_type:"interactive",card:{header:{title:{tag:"plain_text",content:$title},template:"blue"},elements:[{tag:"markdown",content:$body}]}}')"
    curl -fsS --connect-timeout 5 --max-time 10 -X POST "$webhook" \
        -H 'Content-Type: application/json' -d "$payload" >/dev/null || true
}

latest_metric() {
    local metrics_path="$1"
    if [[ -s "$metrics_path" ]]; then
        tail -n 1 "$metrics_path"
    else
        printf 'initializing'
    fi
}

on_exit() {
    local status="$?"
    if [[ "$status" -eq 0 ]]; then
        {
            printf 'completed_at=%s\n' "$(date -Is)"
            printf 'status=success\n'
            printf 'stage2_best=%s\n' "$STAGE2_DIR/best.pt"
            printf 'stage2_last=%s\n' "$STAGE2_DIR/last.pt"
        } > "$STATUS_PATH"
        notify_feishu "Ablation Pure BC Complete" "Pure BC on DAgger data completed. Best: $STAGE2_DIR/best.pt. Run: $RUN_ROOT"
    else
        {
            printf 'completed_at=%s\n' "$(date -Is)"
            printf 'status=failed\n'
            printf 'exit_code=%s\n' "$status"
        } > "$STATUS_PATH"
        notify_feishu "Ablation Pure BC Failed" "Exit code: $status. Inspect: $LOG_DIR/pipeline.log. Run: $RUN_ROOT"
    fi
    exit "$status"
}

check_preflight() {
    [[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; return 1; }
    [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 1; }
    [[ -x "$TORCHRUN_BIN" ]] || { echo "Missing torchrun: $TORCHRUN_BIN" >&2; return 1; }
    [[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; return 1; }
    [[ -s "$DATA_PATH" ]] || { echo "Missing data: $DATA_PATH" >&2; return 1; }
    [[ -s "$ALIGN_CKPT" ]] || { echo "Missing Stage 1 checkpoint: $ALIGN_CKPT" >&2; return 1; }
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
    printf 'experiment=ablation_pure_bc\n'
    printf 'data_path=%s\n' "$DATA_PATH"
    printf 'align_ckpt=%s\n' "$ALIGN_CKPT"
    printf 'gpus=0,1\n'
    printf 'target_mode=hard (pure BC, no cost ranking, no KL, no soft target)\n'
} > "$STATUS_PATH"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

notify_feishu "Ablation Pure BC Started" "Pure BC on DAgger data (no ranking, no KL, no soft target). GPUs: 0,1. Run: $RUN_ROOT"

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
