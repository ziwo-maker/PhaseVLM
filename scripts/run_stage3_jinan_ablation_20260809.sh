#!/usr/bin/env bash
# Stage 3 PPO ablation for Jinan — proper train/eval/test separation.
#
# Protocol:
#   Train on jinan_real (3600s)
#   Checkpoint selection on jinan_real_2000 (held-out flow, same network)
#   Final test on jinan_real_2500 (held-out from both train and eval)
#
# Ablation matrix (4 experiments, ~2-3 GPU-hours each on 1× RTX 5090):
#   A: default PPO (stable_delta, KL=0.02, travel proxy, starvation)
#   B: no KL          (kl_coef=0)
#   C: no travel proxy (selection_travel_weight=0)
#   D: no starvation   (starvation_penalty_coef=0)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage3_jinan_ablation_20260809.sh A
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN="${1:-A}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="records/pipeline_runs/stage3_jinan_ablation_20260809"
OUTPUT_DIR="${EXP_ROOT}/${RUN}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# ── Shared config ──────────────────────────────────────────────────
STAGE2_CKPT="${STAGE2_CKPT:-${REPO_ROOT}/records/pipeline_runs/exp_jinan/stage2_bc/checkpoints/best.pt}"
LLM_PATH="${LLM_PATH:-/home/data/model/qwen-vl-8B}"
DATASET="jinan"
TRAIN_FILE="anon_3_4_jinan_real.json"
EVAL_FILE="anon_3_4_jinan_real_2000.json"
TEST_FILE="anon_3_4_jinan_real_2500.json"
ROADNET="data/Jinan/3_4/roadnet_3_4.json"
TRAFFIC_ROOT="data/Jinan/3_4"
SEED=2024

COMMON_ARGS=(
  --stage2_ckpt "${STAGE2_CKPT}"
  --dataset "${DATASET}"
  --traffic_file "${TRAIN_FILE}"
  --eval_traffic_file "${EVAL_FILE}"
  --test_traffic_file "${TEST_FILE}"
  --roadnet_path "${ROADNET}"
  --traffic_root "${TRAFFIC_ROOT}"
  --llm_path "${LLM_PATH}"
  --seq_len 16
  --image_size 224
  --periodicity 16
  --input_dim 8
  --hidden_dim 64
  --prefix_tokens 8
  --prefix_mode prefix
  --max_length 64
  --value_hidden_dim 512
  --prompt_mode queue_wait_short
  --train_run_counts 3600
  --eval_run_counts 3600
  --run_counts 3600
  --num_updates 80
  --rollout_steps 8
  --ppo_epochs 4
  --mini_batch_size 1
  --grad_accum_steps 8
  --action_eval_chunk_size 1
  --lr 5e-5
  --weight_decay 1e-4
  --gamma 0.99
  --gae_lambda 0.95
  --clip_range 0.2
  --value_coef 0.5
  --entropy_coef 0.01
  --max_grad_norm 1.0
  --reward_alpha 0.1
  --switch_penalty_beta 0.02
  --reward_mode stable_delta
  --queue_delta_scale 4.0
  --wait_delta_scale 30.0
  --active_delta_scale 4.0
  --throughput_scale 2.0
  --reward_clip 3.0
  --global_reward_scale 1.0
  --selection_queue_weight 1.0
  --selection_wait_weight 1.0
  --selection_travel_weight 1.0
  --eval_every 10
  --model_selection_metric composite
  --seed "${SEED}"
  --bf16
  --output_dir "${OUTPUT_DIR}"
)

# ── Ablation-specific overrides ────────────────────────────────────
# In stable_delta mode, the four coefficient args act as MAGNITUDE
# multipliers (abs() is used).  1.0 = full strength, 0.0 = disabled.
case "${RUN}" in
  A)
    # Default PPO — all components active at full strength
    EXTRA_ARGS=(
      --kl_coef 0.02
      --queue_reward_coef -1.0
      --waiting_reward_coef -1.0
      --system_vehicle_reward_coef -1.0
      --throughput_reward_coef 1.0
      --starvation_penalty_coef 0.05
    )
    ;;
  B)
    # No KL to base policy
    EXTRA_ARGS=(
      --kl_coef 0.0
      --queue_reward_coef -1.0
      --waiting_reward_coef -1.0
      --system_vehicle_reward_coef -1.0
      --throughput_reward_coef 1.0
      --starvation_penalty_coef 0.05
    )
    ;;
  C)
    # No travel proxy (zero out travel components)
    EXTRA_ARGS=(
      --kl_coef 0.02
      --queue_reward_coef -1.0
      --waiting_reward_coef -1.0
      --system_vehicle_reward_coef 0.0
      --throughput_reward_coef 0.0
      --selection_travel_weight 0.0
      --starvation_penalty_coef 0.05
    )
    ;;
  D)
    # No starvation penalty
    EXTRA_ARGS=(
      --kl_coef 0.02
      --queue_reward_coef -1.0
      --waiting_reward_coef -1.0
      --system_vehicle_reward_coef -1.0
      --throughput_reward_coef 1.0
      --starvation_penalty_coef 0.0
    )
    ;;
  *)
    echo "Unknown ablation: ${RUN}. Use A, B, C, or D." >&2
    exit 1
    ;;
esac

echo "=== Stage 3 PPO Ablation ${RUN} ==="
echo "Output: ${OUTPUT_DIR}"
echo "Train:  ${TRAIN_FILE} (seed=${SEED})"
echo "Eval:   ${EVAL_FILE}"
echo "Test:   ${TEST_FILE}"
echo "Command:"
echo "torchrun --standalone --nproc_per_node=1 projection_train/train_projection_ppo_traj.py ${COMMON_ARGS[*]} ${EXTRA_ARGS[*]}"

exec torchrun --standalone --nproc_per_node=1 \
  projection_train/train_projection_ppo_traj.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
