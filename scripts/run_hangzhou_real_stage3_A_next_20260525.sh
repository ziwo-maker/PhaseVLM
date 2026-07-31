#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/LLMSTSC}"
PY_BIN="${PY_BIN:-/root/miniconda3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-A_next_$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-records/pipeline_runs/exp_hangzhou_real}"
A_ROOT="${EXP_ROOT}/${RUN_TAG}"
mkdir -p "${A_ROOT}/logs"

BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints/best.pt}"
LLM_PATH="${LLM_PATH:-/gemini/pretrain2/qwen-vl-8B/}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_4_4_hangzhou_real.json}"
DATASET="${DATASET:-hangzhou}"

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

run_variant() {
  local name="$1"
  local gap="$2"
  local protect="$3"
  local target_floor="$4"
  local total_floor="$5"
  local seed="$6"
  local root="${A_ROOT}/${name}"
  mkdir -p "${root}/logs" "${root}/artifacts"
  printf '[%s] start %s gap=%s protect=%s target_floor=%s total_floor=%s seed=%s\n' \
    "$(date '+%F %T')" "${name}" "${gap}" "${protect}" "${target_floor}" "${total_floor}" "${seed}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PY_BIN}" projection_train/run_projection_online.py \
    --memo "${A_ROOT}/${name}/test_runs" \
    --projection_ckpt "${BASE_CKPT}" \
    --device auto \
    --seed "${seed}" \
    --num_rounds 1 \
    --proj_name LLM-TSCS-extreme \
    --run_counts 3600 \
    --result_json "${root}/artifacts/test_result.json" \
    --max_phase_override_gap "${gap}" \
    --max_phase_override_left_wait_protect "${protect}" \
    --max_phase_override_target_queue_floor "${target_floor}" \
    --max_phase_override_total_queue_floor "${total_floor}" \
    "${COMMON_RUNTIME_ARGS[@]}" \
    > "${root}/logs/test.log" 2>&1
  printf '[%s] end %s\n' "$(date '+%F %T')" "${name}"
}

printf 'A-next root: %s\n' "${A_ROOT}"
printf 'Base checkpoint: %s\n' "${BASE_CKPT}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES}"

run_variant "gap2_p45_s2024" 2 45 0 0 2024
run_variant "gap2_p45_s2025" 2 45 0 0 2025
run_variant "gap2_p45_s2026" 2 45 0 0 2026

run_variant "gap3_p45_s2024" 3 45 0 0 2024
run_variant "gap3_p45_s2025" 3 45 0 0 2025
run_variant "gap3_p45_s2026" 3 45 0 0 2026

run_variant "gap2_p30_s2024" 2 30 0 0 2024
run_variant "gap2_p60_s2024" 2 60 0 0 2024

run_variant "gap2_p45_tq3_s2024" 2 45 3 0 2024
run_variant "gap2_p45_total5_s2024" 2 45 0 5 2024
run_variant "gap2_p45_tq3_total5_s2024" 2 45 3 5 2024

"${PY_BIN}" scripts/summarize_hangzhou_real_A_next_20260525.py \
  --repo_root "${REPO_ROOT}" \
  --root "${A_ROOT}" \
  --output "${A_ROOT}/summary.md"
