#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_ROOT" >&2
    exit 2
fi

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/lijn99/miniconda3/envs/lab/bin/python
TORCHRUN_BIN=/home/lijn99/miniconda3/envs/lab/bin/torchrun
MODEL_PATH=/home/data/model/qwen-vl-8B
DATA_PATH=/home/code/LLMSTSC/records/dagger/newyork28x7_double_awt_priority_20260806_workers4/stage2_train.jsonl
RUN_ROOT="$1"
LOG_DIR="$RUN_ROOT/logs"
STATUS_PATH="$RUN_ROOT/completion.status"
STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"
MONITOR_PID=""

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

start_progress_monitor() {
    local pipeline_pid="$$"
    (
        while kill -0 "$pipeline_pid" 2>/dev/null; do
            sleep 2400
            kill -0 "$pipeline_pid" 2>/dev/null || break
            if [[ -d "$STAGE2_DIR" ]]; then
                notify_feishu "NewYork DAgger Training Progress" "Stage 2 running. Latest metric: $(latest_metric "$STAGE2_DIR/metrics.jsonl"). Run: $RUN_ROOT"
            else
                notify_feishu "NewYork DAgger Training Progress" "Stage 1 running. Latest metric: $(latest_metric "$STAGE1_DIR/metrics.jsonl"). Run: $RUN_ROOT"
            fi
        done
    ) &
    MONITOR_PID="$!"
}

on_exit() {
    local status="$?"
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    if [[ "$status" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\n' "$(date -Is)" > "$STATUS_PATH"
        notify_feishu "NewYork DAgger Stage 1 + Stage 2 Complete" "Both stages completed. Stage 1: $STAGE1_DIR/best.pt. Stage 2: $STAGE2_DIR/best.pt. Run: $RUN_ROOT"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\n' "$(date -Is)" "$status" > "$STATUS_PATH"
        notify_feishu "NewYork DAgger Stage 1 + Stage 2 Failed" "Exit code: $status. Inspect: $LOG_DIR/pipeline.log. Run: $RUN_ROOT"
    fi
    exit "$status"
}

check_preflight() {
    [[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; return 1; }
    [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; return 1; }
    [[ -x "$TORCHRUN_BIN" ]] || { echo "Missing torchrun: $TORCHRUN_BIN" >&2; return 1; }
    [[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; return 1; }
    [[ -s "$DATA_PATH" ]] || { echo "Missing data: $DATA_PATH" >&2; return 1; }
    for gpu in 0 1; do
        local used_memory
        used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
        [[ -n "$used_memory" && "$used_memory" -lt 500 ]] || { echo "GPU $gpu is busy: ${used_memory:-unknown} MiB" >&2; return 1; }
    done
}

check_preflight
mkdir -p "$LOG_DIR" "$STAGE1_DIR" "$STAGE2_DIR"
exec > "$LOG_DIR/pipeline.log" 2>&1
trap on_exit EXIT
printf 'started_at=%s\nstatus=running\ndata_path=%s\ngpus=0,1\n' "$(date -Is)" "$DATA_PATH" > "$STATUS_PATH"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

notify_feishu "NewYork DAgger Stage 1 + Stage 2 Started" "Data validation passed: 35,480 action labels and 43,512 captions. GPUs: 0,1. Run: $RUN_ROOT"
start_progress_monitor

printf 'stage1_started_at=%s\n' "$(date -Is)" >> "$STATUS_PATH"
"$TORCHRUN_BIN" --standalone --nproc_per_node=2 train_align.py \
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$STAGE1_DIR" \
    --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --temperature 0.07 \
    --batch_size 4 --grad_accum_steps 2 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16

[[ -s "$STAGE1_DIR/best.pt" ]] || { echo "Stage 1 completed without best.pt" >&2; exit 1; }
printf 'stage1_completed_at=%s\nstage2_started_at=%s\n' "$(date -Is)" "$(date -Is)" >> "$STATUS_PATH"
notify_feishu "NewYork DAgger Stage 1 Complete" "Stage 1 checkpoint ready: $STAGE1_DIR/best.pt. Starting Stage 2."

"$TORCHRUN_BIN" --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
    --data_path "$DATA_PATH" \
    --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_DIR/best.pt" \
    --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
    --batch_size 4 --grad_accum_steps 8 --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 \
    --log_interval 20 --save_every 1 --best_metric val_top1_cost_regret --best_mode min \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 \
    --ranking_loss_weight 0.15 --ranking_min_cost_gap 0.02 \
    --base_policy_kl_weight 0.02 --base_policy_prior_mass 0.70 \
    --base_policy_kl_mode match_or_low_weight --base_policy_kl_low_weight_threshold 1.0 \
    --ddp_timeout_seconds 1800

printf 'stage2_completed_at=%s\n' "$(date -Is)" >> "$STATUS_PATH"
