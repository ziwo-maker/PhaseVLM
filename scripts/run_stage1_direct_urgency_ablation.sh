#!/usr/bin/env bash
# Train one Stage-1 alignment initializer for the direct-urgency ablations.
set -Eeuo pipefail

OUT_DIR="${1:?Usage: $0 OUT_DIR DATA_MODE HIDDEN_DIM PREFIX_TOKENS}"
DATA_MODE="${2:?Usage: $0 OUT_DIR DATA_MODE HIDDEN_DIM PREFIX_TOKENS}"
HIDDEN_DIM="${3:?Usage: $0 OUT_DIR DATA_MODE HIDDEN_DIM PREFIX_TOKENS}"
PREFIX_TOKENS="${4:?Usage: $0 OUT_DIR DATA_MODE HIDDEN_DIM PREFIX_TOKENS}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
TORCHRUN_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/torchrun
MODEL_PATH=/home/data/model/qwen-vl-8B
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/status.env"
RUN_LABEL="pt${PREFIX_TOKENS}_hd${HIDDEN_DIM}_${DATA_MODE}"
GPUS=(0 1)
DATA=()

case "$DATA_MODE" in
    same_flow)
        DATA=("$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl")
        ;;
    three_flow)
        DATA=(
            "$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
            "$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
            "$REPO_ROOT/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl"
        )
        ;;
    *) echo "Unsupported DATA_MODE: $DATA_MODE" >&2; exit 2 ;;
esac

[[ ! -e "$OUT_DIR" ]] || { echo "Refusing to overwrite: $OUT_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" && -x "$TORCHRUN_BIN" && -d "$MODEL_PATH" ]] || { echo "Missing runtime prerequisite" >&2; exit 1; }
for path in "${DATA[@]}"; do [[ -s "$path" ]] || { echo "Missing data: $path" >&2; exit 1; }; done
for gpu in "${GPUS[@]}"; do
    used="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
    [[ -n "$used" && "$used" -lt 500 ]] || { echo "GPU $gpu is busy" >&2; exit 1; }
done

mkdir -p "$LOG_DIR"
printf 'started_at=%s\nstatus=running\narchitecture=pt%s_hd%s\ndata_mode=%s\n' "$(date -Is)" "$PREFIX_TOKENS" "$HIDDEN_DIM" "$DATA_MODE" > "$STATUS"
printf 'data_sha256=%s\n' "$(for path in "${DATA[@]}"; do sha256sum "$path"; done | sha256sum | awk '{print $1}')" >> "$STATUS"

snapshot() {
    local label="$1" pid="$2"
    {
        printf '\n[%s] %s pid=%s\n' "$(date -Is)" "$label" "$pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 100 || true
        timeout --foreground 10s tail -n 80 "$LOG_DIR/train.log" || true
        timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory' "$LOG_DIR/train.log" | tail -n 40 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$LOG_DIR/startup_monitor.log" 2>&1
}

cd "$REPO_ROOT"
setsid env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
    "$TORCHRUN_BIN" --standalone --nproc_per_node=2 train_align.py \
    --data_path "${DATA[@]}" --model_path "$MODEL_PATH" --output_dir "$OUT_DIR" --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim "$HIDDEN_DIM" --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix \
    --temperature 0.05 --batch_size 4 --grad_accum_steps 2 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 1000 --log_every_steps 20 --save_every_epochs 5 --seed 2024 --bf16 \
    > "$LOG_DIR/train.log" 2>&1 &
PID=$!
snapshot startup_t0 "$PID"
if timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory' "$LOG_DIR/train.log"; then
    kill -- -"$PID" 2>/dev/null || true
    wait "$PID" || true
    exit 1
fi
wait "$PID"
[[ -s "$OUT_DIR/best.pt" ]] || { echo "Stage 1 completed without best.pt" >&2; exit 1; }
printf 'completed_at=%s\nstatus=success\nbest_checkpoint=%s\nbest_checkpoint_sha256=%s\n' "$(date -Is)" "$OUT_DIR/best.pt" "$(sha256sum "$OUT_DIR/best.pt" | awk '{print $1}')" > "$STATUS"
printf 'data_mode=%s\narchitecture=pt%s_hd%s\n' "$DATA_MODE" "$PREFIX_TOKENS" "$HIDDEN_DIM" >> "$STATUS"
