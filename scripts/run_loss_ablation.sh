#!/usr/bin/env bash
# Loss function ablation training
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${1:-0}"
RUN_NAME="${2:-ablation_v1}"
EXTRA_ARGS="${3:-}"

OUTDIR="records/pipeline_runs/${RUN_NAME}_20260812"
mkdir -p "${OUTDIR}"

echo "=== Starting ${RUN_NAME} on GPU ${GPU_ID} ==="
echo "Output: ${OUTDIR}"
echo "Time: $(date)"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 projection_train/train_projection_bc.py \
  --data_path data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl \
  --val_ratio 0.1 \
  --llm_path /home/data/model/qwen-vl-8B \
  --align_ckpt records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt \
  --output_dir "${OUTDIR}" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
  --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode queue_wait_short \
  --epochs 10 --batch_size 4 --grad_accum_steps 4 \
  --lr 1e-4 --weight_decay 0.01 \
  --seed 42 --save_every 2 \
  --best_metric val_top1_cost_regret --best_mode min \
  ${EXTRA_ARGS} \
  2>&1 | tee "${OUTDIR}/train.log"

echo "=== Done ${RUN_NAME} at $(date) ==="
