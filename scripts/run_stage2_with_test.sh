#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-}"
if [[ -z "${TORCHRUN_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/torchrun" ]]; then
    TORCHRUN_BIN="${REPO_ROOT}/.venv/bin/torchrun"
  else
    TORCHRUN_BIN="torchrun"
  fi
fi

cd "${REPO_ROOT}"

LOG_DIR="records/pipeline_runs/exp_jinan/stage2_bc"
LOG_PATH="${LOG_DIR}/nohup.log"
mkdir -p "${LOG_DIR}"

nohup "${TORCHRUN_BIN}" --nproc_per_node=8 projection_train/train_projection_bc.py \
  --data_path data/recorded/align_train.cleaned.llm_caption.jsonl \
  --llm_path /home/data/model/qwen-vl-8B/ \
  --align_ckpt records/align/align_epoch_40.pt \
  --output_dir records/projection_bc_margin1_t1 \
  --intersection_id "" \
  --batch_size 1 \
  --seq_len 16 \
  --epochs 15 \
  --max_length 64 \
  --prompt_mode image_only_short \
  --lr 1e-4 \
  --grad_accum_steps 4 \
  --bf16 \
  --image_size 224 \
  --prefix_tokens 8 \
  --prefix_mode prefix \
  --target_mode mixed \
  --soft_target_alpha 0.3 \
  --soft_label_temperature 1.0 \
  --min_action_margin 1.0 \
  > "${LOG_PATH}" 2>&1 &

echo "Started Stage 2 BC training in background."
echo "PID: $!"
echo "Log: ${LOG_PATH}"
