#!/usr/bin/env bash
# Train one local wait25 refinement arm and evaluate it on Jinan real only.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT ARM TRAINING_SEED}"
ARM="${2:?Usage: $0 RUN_ROOT ARM TRAINING_SEED}"
SEED="${3:?Usage: $0 RUN_ROOT ARM TRAINING_SEED}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${WAIT25_IMPROVEMENT_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN_BIN="${WAIT25_IMPROVEMENT_TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH=/home/data/model/qwen-vl-8B
DATA_PATH="${WAIT25_IMPROVEMENT_DATA_PATH:-$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl}"
ALIGN_CKPT="${WAIT25_IMPROVEMENT_ALIGN_CKPT:-$REPO_ROOT/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt}"
HIDDEN_DIM="${WAIT25_IMPROVEMENT_HIDDEN_DIM:-64}"
PREFIX_TOKENS="${WAIT25_IMPROVEMENT_PREFIX_TOKENS:-8}"
ARCHITECTURE_TAG="pt${PREFIX_TOKENS}_hd${HIDDEN_DIM}"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_wait25_improvement_online_suite.sh"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
EXTRA_TRAIN_ARGS_RAW="${WAIT25_IMPROVEMENT_EXTRA_TRAIN_ARGS:-}"
ARM_ROOT="$RUN_ROOT/arms/$ARM/seed_$SEED"
STAGE2_DIR="$ARM_ROOT/stage2"
LOG_DIR="$ARM_ROOT/logs"
STATUS_PATH="$ARM_ROOT/status.txt"
GLOBAL_EFFECTIVE_BATCH=96
PER_RANK_BATCH=4
SELECTED_GPUS=()
EXTRA_TRAIN_ARGS=()

timestamp() { date -Is; }

notify() {
    if [[ -f "$FEISHU_SCRIPT" ]]; then
        set +e
        timeout --kill-after=5s 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1"
        set -e
    fi
    return 0
}

latest_train_metric() {
    [[ -s "$STAGE2_DIR/metrics.jsonl" ]] || { printf 'train_metric=not_available'; return 0; }
    "$PYTHON_BIN" - "$STAGE2_DIR/metrics.jsonl" <<'PY' 2>/dev/null || printf 'train_metric=not_available'
import json, sys

last = None
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            last = json.loads(line)
if last is None:
    print("train_metric=not_available")
else:
    print(
        "epoch={epoch} val_acc={val_acc} val_loss={val_loss}".format(
            epoch=last.get("epoch"),
            val_acc=last.get("val_acc"),
            val_loss=last.get("val_loss"),
        )
    )
PY
}

discover_idle_gpus() {
    local gpu used
    SELECTED_GPUS=()
    if [[ -n "${WAIT25_IMPROVEMENT_GPUS:-}" ]]; then
        IFS=',' read -r -a SELECTED_GPUS <<< "$WAIT25_IMPROVEMENT_GPUS"
    else
        while IFS=, read -r gpu used; do
            gpu="${gpu//[[:space:]]/}"
            used="${used//[[:space:]]/}"
            [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
            (( used < 500 )) && SELECTED_GPUS+=("$gpu")
        done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    fi
    ((${#SELECTED_GPUS[@]} > 0)) || { echo "No idle GPU is available" >&2; return 1; }
}

snapshot() {
    local label="$1" pid="$2" output="$LOG_DIR/startup_monitor.log"
    {
        printf '\n[%s] %s child_pid=%s\n' "$(timestamp)" "$label" "$pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 120 || true
        [[ -f "$LOG_DIR/pipeline.log" ]] && timeout --foreground 10s tail -n 80 "$LOG_DIR/pipeline.log" || true
        [[ -f "$LOG_DIR/pipeline.log" ]] && timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$LOG_DIR/pipeline.log" | tail -n 40 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$output" 2>&1
}

terminate_train() {
    local pid="$1"
    timeout --foreground 10s pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    timeout --foreground 10s pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
}

monitor_startup() {
    local pid="$1"
    snapshot startup_t0 "$pid"
    if timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$LOG_DIR/pipeline.log"; then
        terminate_train "$pid"
        wait "$pid" || true
        return 1
    fi
}

case "$ARM" in
    hard_bc)
        TARGET_ARGS=(--target_mode hard --ranking_loss_weight 0 --base_policy_kl_weight 0)
        ;;
    wait2_mixed_rank)
        TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
        ;;
    wait225_mixed_rank)
        TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.25 --cost_spillback_weight 0)
        ;;
    wait25_conservative_rank)
        TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.15 --soft_label_temperature 0.35 --ranking_loss_weight 0.02 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.5 --cost_spillback_weight 0)
        ;;
    wait25_strong_rank)
        TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.45 --soft_label_temperature 0.35 --ranking_loss_weight 0.08 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.5 --cost_spillback_weight 0)
        ;;
    tail_p95_mixed_rank)
        TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2 --cost_spillback_weight 0 --cost_tail_wait_weight 1)
        ;;
    *)
        echo "Unknown wait25 refinement arm: $ARM" >&2
        exit 2
        ;;
esac

on_exit() {
    local code="$?"
    # Prerequisite failures can occur before the arm directory is initialized.
    mkdir -p "$ARM_ROOT"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\narm=%s\ntraining_seed=%s\ntraining_flow=anon_3_4_jinan_real.json\nstage2_best=%s\nstage2_best_sha256=%s\narchitecture=%s\nextra_train_args=%s\nonline_status=success\n' "$(timestamp)" "$ARM" "$SEED" "$STAGE2_DIR/best.pt" "$(sha256sum "$STAGE2_DIR/best.pt" | awk '{print $1}')" "$ARCHITECTURE_TAG" "$EXTRA_TRAIN_ARGS_RAW" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\narm=%s\ntraining_seed=%s\ntraining_flow=anon_3_4_jinan_real.json\narchitecture=%s\nextra_train_args=%s\n' "$(timestamp)" "$code" "$ARM" "$SEED" "$ARCHITECTURE_TAG" "$EXTRA_TRAIN_ARGS_RAW" > "$STATUS_PATH"
    fi
    trap - EXIT
    exit "$code"
}
trap on_exit EXIT

[[ -x "$PYTHON_BIN" && -x "$TORCHRUN_BIN" ]] || { echo "Missing hnr5090 Python or torchrun" >&2; exit 1; }
[[ -s "$DATA_PATH" && -s "$ALIGN_CKPT" && -x "$ONLINE_SUITE" ]] || { echo "Missing training prerequisite" >&2; exit 1; }
[[ ! -e "$ARM_ROOT" ]] || { echo "Refusing to overwrite existing arm: $ARM_ROOT" >&2; exit 1; }
discover_idle_gpus
GPU_COUNT="${#SELECTED_GPUS[@]}"
(( GLOBAL_EFFECTIVE_BATCH % (GPU_COUNT * PER_RANK_BATCH) == 0 )) || { echo "Cannot preserve effective batch on ${GPU_COUNT} GPUs" >&2; exit 1; }
GRAD_ACCUM_STEPS="$(( GLOBAL_EFFECTIVE_BATCH / (GPU_COUNT * PER_RANK_BATCH) ))"
GPU_LIST="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]}")"

mkdir -p "$STAGE2_DIR" "$LOG_DIR"
if [[ -n "$EXTRA_TRAIN_ARGS_RAW" ]]; then
    read -r -a EXTRA_TRAIN_ARGS <<< "$EXTRA_TRAIN_ARGS_RAW"
fi
printf 'started_at=%s\nstatus=running\narm=%s\ntraining_seed=%s\ntraining_flow=anon_3_4_jinan_real.json\ndata=%s\ndata_sha256=%s\nalign_ckpt=%s\narchitecture=%s\nextra_train_args=%s\n' "$(timestamp)" "$ARM" "$SEED" "$DATA_PATH" "$(sha256sum "$DATA_PATH" | awk '{print $1}')" "$ALIGN_CKPT" "$ARCHITECTURE_TAG" "$EXTRA_TRAIN_ARGS_RAW" > "$STATUS_PATH"
cd "$REPO_ROOT"
notify "jinan_real Stage 2 launched: arm=${ARM}, train_seed=${SEED}, architecture=${ARCHITECTURE_TAG}, gpus=${GPU_LIST}, grad_accum=${GRAD_ACCUM_STEPS}."
env CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPU_COUNT" projection_train/train_projection_bc.py --data_path "$DATA_PATH" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --output_dir "$STAGE2_DIR" --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim "$HIDDEN_DIM" --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short --batch_size "$PER_RANK_BATCH" --grad_accum_steps "$GRAD_ACCUM_STEPS" --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 24 --num_workers 4 --seed "$SEED" --bf16 --disable_class_balance --log_interval 20 --save_every 1 --checkpoint_slim --best_metric val_acc --best_mode max --ddp_timeout_seconds 1800 "${TARGET_ARGS[@]}" "${EXTRA_TRAIN_ARGS[@]}" > "$LOG_DIR/pipeline.log" 2>&1
[[ -s "$STAGE2_DIR/best.pt" ]] || { echo "Stage 2 finished without best.pt" >&2; exit 1; }
printf 'stage2_completed_at=%s\nstage2_status=success\nstage2_best_sha256=%s\n' "$(timestamp)" "$(sha256sum "$STAGE2_DIR/best.pt" | awk '{print $1}')" >> "$STATUS_PATH"
"$ONLINE_SUITE" "$RUN_ROOT" "$ARM_ROOT" "$STAGE2_DIR/best.pt" "jinan_wait25_${ARM}_train${SEED}"
printf 'online_completed_at=%s\nonline_status=success\n' "$(timestamp)" >> "$STATUS_PATH"
notify "jinan_real wait25 refinement arm completed: arm=${ARM}, train_seed=${SEED}; $(latest_train_metric)"
