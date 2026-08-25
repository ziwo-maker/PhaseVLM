#!/usr/bin/env bash
# Stage 1 Alignment Retraining — Improved Architecture
# prefix_tokens=32 (was 8), hidden_dim=256 (was 64)
# Estimated: ~10 hours for 15 epochs on 1× RTX 5090
set -euo pipefail
cd /home/code/LLMSTSC

OUTDIR="records/pipeline_runs/stage1_pt32_hd256_20260812"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "Stage 1: Alignment Training (pt=32, hd=256)"
echo "Start: $(date)"
echo "Output: ${OUTDIR}"
echo "GPU: 0"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python3 -u train_align.py \
  --data_path \
    data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl \
    data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl \
    data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl \
  --model_path /home/data/model/qwen-vl-8B \
  --output_dir "${OUTDIR}" \
  --decision_only \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
  --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix \
  --temperature 0.07 \
  --batch_size 2 --grad_accum_steps 2 \
  --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
  --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 \
  > "${OUTDIR}/train.log" 2>&1

echo "============================================================"
echo "Stage 1 Complete: $(date)"
echo "Checkpoint: ${OUTDIR}/best.pt"
echo "============================================================"

# Auto-launch Stage 2 if Stage 1 succeeded
if [[ -s "${OUTDIR}/best.pt" ]]; then
  echo "Stage 1 succeeded. Launching Stage 2..."
  STAGE2DIR="records/pipeline_runs/stage2_pt32_hd256_20260812"
  mkdir -p "${STAGE2DIR}"

  CUDA_VISIBLE_DEVICES=0 python3 -u projection_train/train_projection_bc.py \
    --data_path \
      data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl \
      data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl \
      data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl \
    --llm_path /home/data/model/qwen-vl-8B \
    --align_ckpt "${OUTDIR}/best.pt" \
    --output_dir "${STAGE2DIR}" \
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
    --save_every 3 \
    > "${STAGE2DIR}/train.log" 2>&1

  echo "Stage 2 Complete: $(date)"
fi
