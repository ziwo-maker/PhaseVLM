#!/usr/bin/env bash
# Train the frozen Jinan SimExpert reference arm and evaluate it on the same flow only.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT ARM TRAINING_SEED}"
ARM="${2:?Usage: $0 RUN_ROOT ARM TRAINING_SEED}"
SEED="${3:?Usage: $0 RUN_ROOT ARM TRAINING_SEED}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${FROZEN_SIMEXPERT_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /home/sysu/anaconda3/envs/hnr5090/bin/python ]]; then
        PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
TORCHRUN_BIN="${FROZEN_SIMEXPERT_TORCHRUN:-}"
if [[ -z "$TORCHRUN_BIN" ]]; then
    if [[ -x /home/sysu/anaconda3/envs/hnr5090/bin/torchrun ]]; then
        TORCHRUN_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/torchrun
    else
        TORCHRUN_BIN="$(command -v torchrun)"
    fi
fi
if [[ -n "${FROZEN_SIMEXPERT_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$FROZEN_SIMEXPERT_LD_PRELOAD"
fi

MODEL_PATH=/home/data/model/qwen-vl-8B
DATA_PATH="$REPO_ROOT/records/sim_expert_validation_search/anon_3_4_jinan_real/best/traffic_prompt_series_2s.jsonl"
ALIGN_CKPT="$REPO_ROOT/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_frozen_simexpert_online_suite.sh"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
ARM_ROOT="$RUN_ROOT/arms/$ARM/seed_$SEED"
STAGE2_DIR="$ARM_ROOT/stage2"
LOG_DIR="$ARM_ROOT/logs"
STATUS_PATH="$ARM_ROOT/status.txt"
GLOBAL_EFFECTIVE_BATCH=96
PER_RANK_BATCH=4
LAST_NOTIFICATION_EPOCH=0
SELECTED_GPUS=()

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

discover_idle_gpus() {
    local gpu used candidate
    SELECTED_GPUS=()
    if [[ -n "${FROZEN_SIMEXPERT_GPUS:-}" ]]; then
        IFS=',' read -r -a SELECTED_GPUS <<< "$FROZEN_SIMEXPERT_GPUS"
    else
        while IFS=, read -r gpu used; do
            gpu="${gpu//[[:space:]]/}"
            used="${used//[[:space:]]/}"
            [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
            if (( used < 500 )); then
                SELECTED_GPUS+=("$gpu")
            fi
        done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    fi
    ((${#SELECTED_GPUS[@]} > 0)) || { echo "No idle GPU is available for Stage 2" >&2; return 1; }
    for candidate in "${SELECTED_GPUS[@]}"; do
        [[ "$candidate" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $candidate" >&2; return 1; }
    done
}

latest_train_metric() {
    local progress="$STAGE2_DIR/train_progress.log"
    if [[ -f "$progress" ]]; then
        timeout --foreground 10s rg '\[epoch [0-9]+\] val_' "$progress" | tail -n 1 || true
    else
        printf 'offline_metric=not_available'
    fi
}

maybe_progress_notification() {
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "jinan_real frozen SimExpert reference progress: arm=${ARM} train_seed=${SEED}; $(latest_train_metric)"
}

record_snapshot() {
    local label="$1"
    local pid="$2"
    local snapshot="$LOG_DIR/startup_monitor.log"
    {
        printf '\n[%s] %s child_pid=%s\n' "$(timestamp)" "$label" "$pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | \
            awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 120 || true
        if [[ -f "$LOG_DIR/pipeline.log" ]]; then
            timeout --foreground 10s tail -n 100 "$LOG_DIR/pipeline.log" || true
            timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$LOG_DIR/pipeline.log" | tail -n 40 || true
        fi
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$snapshot" 2>&1
}

monitor_training_startup() {
    local pid="$1"
    local minute
    record_snapshot startup_t0 "$pid"
    for minute in 1 2 3 4 5; do
        sleep 60
        record_snapshot "startup_t${minute}m" "$pid"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            return $?
        fi
        if timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$LOG_DIR/pipeline.log"; then
            kill -- -"$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        fi
    done
}

case "$ARM" in
    frozen_simexpert_mixed_rank)
        TARGET_ARGS=(
            --target_mode mixed
            --soft_target_alpha 0.3
            --soft_label_temperature 0.35
            --ranking_loss_weight 0.05
            --ranking_min_cost_gap 0.02
            --base_policy_kl_weight 0
            --cost_att_weight 1
            --cost_aql_weight 1
            --cost_awt_weight 1
            --cost_spillback_weight 0
        )
        ;;
    *)
        echo "Unknown arm: $ARM" >&2
        exit 2
        ;;
esac

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\narm=%s\ntraining_seed=%s\ntraining_flow=anon_3_4_jinan_real.json\nstage2_best=%s\nstage2_best_sha256=%s\nonline_status=success\n' \
            "$(timestamp)" "$ARM" "$SEED" "$STAGE2_DIR/best.pt" "$(sha256sum "$STAGE2_DIR/best.pt" | awk '{print $1}')" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\narm=%s\ntraining_seed=%s\ntraining_flow=anon_3_4_jinan_real.json\n' \
            "$(timestamp)" "$code" "$ARM" "$SEED" > "$STATUS_PATH"
    fi
    trap - EXIT
    exit "$code"
}
trap on_exit EXIT

[[ -s "$DATA_PATH" ]] || { echo "Missing frozen SimExpert data: $DATA_PATH" >&2; exit 1; }
[[ -s "$ALIGN_CKPT" ]] || { echo "Missing Jinan Stage 1 checkpoint: $ALIGN_CKPT" >&2; exit 1; }
[[ -x "$ONLINE_SUITE" ]] || { echo "Missing same-flow online suite: $ONLINE_SUITE" >&2; exit 1; }
[[ ! -e "$ARM_ROOT" ]] || { echo "Run directory already exists: $ARM_ROOT" >&2; exit 1; }
discover_idle_gpus
GPU_COUNT="${#SELECTED_GPUS[@]}"
(( GLOBAL_EFFECTIVE_BATCH % (GPU_COUNT * PER_RANK_BATCH) == 0 )) || {
    echo "Cannot preserve effective batch ${GLOBAL_EFFECTIVE_BATCH} on ${GPU_COUNT} GPUs" >&2
    exit 1
}
GRAD_ACCUM_STEPS="$(( GLOBAL_EFFECTIVE_BATCH / (GPU_COUNT * PER_RANK_BATCH) ))"
CUDA_DEVICE_LIST="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]}")"

mkdir -p "$STAGE2_DIR" "$LOG_DIR"
printf 'started_at=%s\nstatus=running\narm=%s\ntraining_seed=%s\ntraining_flow=anon_3_4_jinan_real.json\ndata=%s\ndata_sha256=%s\nalign_ckpt=%s\narchitecture=pt8_hd64\n' \
    "$(timestamp)" "$ARM" "$SEED" "$DATA_PATH" "$(sha256sum "$DATA_PATH" | awk '{print $1}')" "$ALIGN_CKPT" > "$STATUS_PATH"

cd "$REPO_ROOT"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "jinan_real frozen SimExpert Stage 2 launched: arm=${ARM}, train_seed=${SEED}, GPUs=${CUDA_DEVICE_LIST}, grad_accum=${GRAD_ACCUM_STEPS}."
setsid env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPU_COUNT" projection_train/train_projection_bc.py \
    --data_path "$DATA_PATH" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
    --batch_size "$PER_RANK_BATCH" --grad_accum_steps "$GRAD_ACCUM_STEPS" --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 \
    --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 24 \
    --num_workers 4 --seed "$SEED" --bf16 --disable_class_balance --log_interval 20 --save_every 1 \
    --checkpoint_slim --best_metric val_acc --best_mode max --ddp_timeout_seconds 1800 "${TARGET_ARGS[@]}" \
    > "$LOG_DIR/pipeline.log" 2>&1 &
TRAIN_PID=$!
monitor_training_startup "$TRAIN_PID"
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
        maybe_progress_notification
    done
    wait "$TRAIN_PID"
fi
[[ -s "$STAGE2_DIR/best.pt" ]] || { echo "Stage 2 finished without best.pt" >&2; exit 1; }
printf 'stage2_completed_at=%s\nstage2_status=success\nstage2_best_sha256=%s\n' \
    "$(timestamp)" "$(sha256sum "$STAGE2_DIR/best.pt" | awk '{print $1}')" >> "$STATUS_PATH"

"$ONLINE_SUITE" "$RUN_ROOT" "$ARM_ROOT" "$STAGE2_DIR/best.pt" "jinan_frozen_simexpert_${ARM}_train${SEED}"
printf 'online_completed_at=%s\nonline_status=success\n' "$(timestamp)" >> "$STATUS_PATH"
notify "jinan_real frozen SimExpert arm completed: arm=${ARM}, train_seed=${SEED}; $(latest_train_metric)"
