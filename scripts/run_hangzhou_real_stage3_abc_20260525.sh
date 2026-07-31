#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/LLMSTSC}"
PY_BIN="${PY_BIN:-/root/miniconda3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-abc_$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-records/pipeline_runs/exp_hangzhou_real}"
ABC_ROOT="${EXP_ROOT}/${RUN_TAG}"
mkdir -p "${ABC_ROOT}/logs" "${ABC_ROOT}/artifacts"

BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints/best.pt}"
LLM_PATH="${LLM_PATH:-/gemini/pretrain2/qwen-vl-8B/}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_4_4_hangzhou_real.json}"
DATASET="${DATASET:-hangzhou}"
ROADNET_PATH="${ROADNET_PATH:-data/Hangzhou/4_4/roadnet_4_4.json}"
TRAFFIC_ROOT="${TRAFFIC_ROOT:-data/Hangzhou/4_4}"
BASE_PROMPT_SERIES="${BASE_PROMPT_SERIES:-data/recorded/anon_4_4_hangzhou_real.json_05_24_17_08_12/traffic_prompt_series_2s.jsonl}"

COMMON_RUNTIME_ARGS=(
  --dataset "${DATASET}"
  --traffic_file "${TRAFFIC_FILE}"
  --llm_path "${LLM_PATH}"
  --seq_len 16
  --image_size 224
  --periodicity 16
  --input_dim 8
  --hidden_dim 64
  --prefix_tokens 8
  --prefix_mode prefix
  --max_length 64
  --prompt_mode image_only_short
  --bf16
)

COMMON_TRAIN_ARGS=(
  --stage2_ckpt "${BASE_CKPT}"
  --dataset "${DATASET}"
  --traffic_file "${TRAFFIC_FILE}"
  --eval_traffic_file "${TRAFFIC_FILE}"
  --roadnet_path "${ROADNET_PATH}"
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
  --prompt_mode image_only_short
  --train_run_counts 3600
  --eval_run_counts 3600
  --ppo_epochs 1
  --mini_batch_size 8
  --grad_accum_steps 2
  --action_eval_chunk_size 1
  --eval_every 2
  --save_every 1
  --lr 5e-7
  --clip_range 0.01
  --kl_coef 1.2
  --entropy_coef 0.0
  --value_coef 0.2
  --reward_alpha 0.1
  --switch_penalty_beta 0.02
  --queue_reward_coef -0.25
  --pressure_reward_coef 0.0
  --waiting_reward_coef -0.01
  --system_vehicle_reward_coef -0.01
  --throughput_reward_coef 0.02
  --starvation_penalty_coef 0.1
  --starvation_threshold_steps 2
  --starvation_queue_threshold 8.0
  --starvation_wait_threshold 45.0
  --wait_reward_scope neighborhood
  --wait_reward_reducer max
  --reward_mode stable_delta
  --queue_delta_scale 4.0
  --wait_delta_scale 45.0
  --wait_abs_coef 0.3
  --wait_abs_scale 60.0
  --active_delta_scale 2.0
  --throughput_scale 0.5
  --reward_clip 3.0
  --global_reward_scale 1.0
  --model_selection_metric composite
  --selection_queue_weight 1.0
  --selection_wait_weight 2.0
  --selection_travel_weight 1.5
  --selection_max_queue_ratio 1.02
  --selection_max_wait_ratio 1.03
  --selection_max_travel_ratio 1.01
  --seed 2024
  --bf16
)

run_online_override() {
  local gap="$1"
  local root="${ABC_ROOT}/A_override_gap${gap}"
  mkdir -p "${root}/logs" "${root}/artifacts"
  echo "===== A override gap=${gap} start $(date '+%F %T') ====="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PY_BIN}" projection_train/run_projection_online.py \
    --memo "${EXP_ROOT}/${RUN_TAG}/A_override_gap${gap}/test_runs" \
    --projection_ckpt "${BASE_CKPT}" \
    --device auto \
    --seed 2024 \
    --num_rounds 1 \
    --proj_name LLM-TSCS-extreme \
    --run_counts 3600 \
    --result_json "${root}/artifacts/test_result.json" \
    --max_phase_override_gap "${gap}" \
    --max_phase_override_left_wait_protect 45 \
    "${COMMON_RUNTIME_ARGS[@]}" \
    > "${root}/logs/test.log" 2>&1
  echo "===== A override gap=${gap} end $(date '+%F %T') ====="
}

run_stage3() {
  local name="$1"
  shift
  local root="${EXP_ROOT}/${name}_${RUN_TAG}"
  mkdir -p "${root}"
  echo "===== ${name} start $(date '+%F %T') root=${root} ====="
  PYTHON_BIN="${PY_BIN}" "${PY_BIN}" scripts/run_projection_stage.py stage3 \
    --stage_root "${root}" \
    --report_md "${root}/report.md" \
    --python_bin "${PY_BIN}" \
    --torchrun_bin "${TORCHRUN_BIN}" \
    --nproc_per_node 1 \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
    --test_dataset "${DATASET}" \
    --test_traffic_file "${TRAFFIC_FILE}" \
    --test_device auto \
    --test_num_rounds 1 \
    --test_proj_name LLM-TSCS-extreme \
    --test_run_counts 3600 \
    --test_precision bf16 \
    --pytorch_cuda_alloc_conf expandable_segments:True \
    -- "$@" \
    > "${ABC_ROOT}/logs/${name}.wrapper.log" 2>&1
  echo "===== ${name} end $(date '+%F %T') root=${root} ====="
}

echo "ABC root: ${ABC_ROOT}"
echo "Base checkpoint: ${BASE_CKPT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

for gap in 2 3 5 8; do
  run_online_override "${gap}"
done

SNAPSHOT_JSONL="${ABC_ROOT}/snapshot_seeds/hzreal_worst_15_45.jsonl"
"${PY_BIN}" scripts/prepare_snapshot_seeds_from_prompt_series.py \
  --input "${BASE_PROMPT_SERIES}" \
  --output "${SNAPSHOT_JSONL}" \
  --start_s 900 \
  --end_s 2700 \
  --max_snapshots 48 \
  > "${ABC_ROOT}/logs/prepare_snapshot_seeds.log" 2>&1

run_stage3 "B_longroll60_lr5e7_clip001_kl12_ent0" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --num_updates 4 \
  --rollout_steps 60 \
  --snapshot_reset_prob 0

run_stage3 "C_snapshot_worst15_45_roll32_lr5e7_clip001_kl12_ent0" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --num_updates 4 \
  --rollout_steps 32 \
  --snapshot_jsonl "${SNAPSHOT_JSONL}" \
  --snapshot_reset_prob 0.5 \
  --snapshot_traffic_file "${TRAFFIC_FILE}" \
  --snapshot_max_samples 48

echo "All A/B/C experiments finished at $(date '+%F %T')"
