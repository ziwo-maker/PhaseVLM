#!/usr/bin/env bash
# jinan_real_2500 held-out source policy — Stage 1 (align) + Stage 2 (BC policy).
#
# Trains a FRESH pt32/hd256 source policy on jinan_real + jinan_real_2000 ONLY,
# leaving jinan_real_2500 as a held-out target. This is the "从 stage 1 开始"
# step; "训练集不满足也要生成" = the source training set intentionally excludes
# 2500 (target out-of-distribution wrt source), yet we still generate the full
# experiment matrix.
#
# DDP across GPUs 0/1/2 (both trainers support torchrun + rank-0 checkpoint save).
# Idempotent: skips a stage if its best.pt already exists and is non-truncated.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
TORCHRUN="$(dirname "$PY")/torchrun"
LLM=/home/data/model/qwen-vl-8B

STAGE1_OUT="$REPO/records/pipeline_runs/stage1_pt32_hd256_jinan2500_20260815"
STAGE2_OUT="$REPO/records/pipeline_runs/stage2_pt32_hd256_jinan2500_20260815"

# Source training data = jinan_real + jinan_real_2000 (NOT jinan_real_2500).
DATA1="$REPO/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DATA2="$REPO/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"

GPU_COUNT=3
MIN_CKPT_BYTES=200000000   # 200MB; complete slim checkpoints are ~303MB

cd "$REPO"

log() { echo "[$(date '+%F %T')] $*"; }

is_complete() {
    local p="$1"
    [[ -s "$p" ]] && [[ "$(stat -c%s "$p")" -ge "$MIN_CKPT_BYTES" ]]
}

# ---------------------------------------------------------------------------
# Stage 1 — alignment
# ---------------------------------------------------------------------------
if is_complete "$STAGE1_OUT/best.pt"; then
    log "Stage 1 best.pt exists and is non-truncated; skip. ($STAGE1_OUT/best.pt)"
else
    log "=== Stage 1: alignment (DDP x${GPU_COUNT}) ==="
    mkdir -p "$STAGE1_OUT"
    "$TORCHRUN" --nproc_per_node="$GPU_COUNT" --master_port=29501 \
        train_align.py \
        --data_path "$DATA1" "$DATA2" \
        --model_path "$LLM" \
        --output_dir "$STAGE1_OUT" \
        --decision_only \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix \
        --temperature 0.07 \
        --batch_size 2 --grad_accum_steps 2 \
        --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
        --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
        --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 \
        > "$STAGE1_OUT/train.log" 2>&1
    log "Stage 1 finished -> $STAGE1_OUT/best.pt"
fi

# ---------------------------------------------------------------------------
# Stage 2 — BC policy (warm-start from Stage 1)
# ---------------------------------------------------------------------------
if is_complete "$STAGE2_OUT/best.pt"; then
    log "Stage 2 best.pt exists and is non-truncated; skip. ($STAGE2_OUT/best.pt)"
else
    if ! is_complete "$STAGE1_OUT/best.pt"; then
        log "ERROR: Stage 1 best.pt missing/truncated; cannot run Stage 2."
        exit 1
    fi
    log "=== Stage 2: BC policy (DDP x${GPU_COUNT}, warm-start Stage 1) ==="
    mkdir -p "$STAGE2_OUT"
    "$TORCHRUN" --nproc_per_node="$GPU_COUNT" --master_port=29502 \
        projection_train/train_projection_bc.py \
        --data_path "$DATA1" "$DATA2" \
        --llm_path "$LLM" \
        --align_ckpt "$STAGE1_OUT/best.pt" \
        --output_dir "$STAGE2_OUT" \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 64 \
        --prompt_mode queue_wait_short \
        --batch_size 2 --grad_accum_steps 8 --epochs 15 \
        --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
        --weight_decay 1e-4 --max_grad_norm 1.0 \
        --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 \
        --ranking_loss_weight 0.5 --cost_awt_weight 2.0 --cost_aql_weight 1.0 \
        --best_metric val_top1_cost_regret --best_mode min \
        --enable_class_balance --overconf_penalty_weight 0.1 \
        --save_every 3 --checkpoint_slim --ddp_timeout_seconds 1800 \
        > "$STAGE2_OUT/train.log" 2>&1
    log "Stage 2 finished -> $STAGE2_OUT/best.pt"
fi

log "STAGE 1+2 DONE. Stage1=$STAGE1_OUT/best.pt Stage2=$STAGE2_OUT/best.pt"
