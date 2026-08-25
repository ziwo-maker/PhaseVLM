#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
OUTDIR="records/pipeline_runs/ablation_cb_rank_20260812"
mkdir -p "${OUTDIR}"
echo "=== Class Balance + Ranking (fp32) on GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 python3 projection_train/train_projection_bc.py \
  --data_path data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl \
  --val_ratio 0.1 --llm_path /home/data/model/qwen-vl-8B \
  --align_ckpt records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt \
  --output_dir "${OUTDIR}" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
  --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode queue_wait_short --epochs 10 --batch_size 2 --grad_accum_steps 8 \
  --lr 1e-4 --weight_decay 0.01 --seed 42 --save_every 2 \
  --ranking_loss_weight 0.5 --cost_awt_weight 2.0 --cost_aql_weight 1.0 \
  --best_metric val_top1_cost_regret --best_mode min \
  --enable_class_balance --overconf_penalty_weight 0.2 \
  --soft_target_alpha 0.5 --soft_label_temperature 0.5 \
  --fp16 \
  > "${OUTDIR}/train.log" 2>&1
echo "=== Done at $(date) ==="
