#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/bin/torchrun}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LLM_PATH="${LLM_PATH:-/gemini/pretrain2/qwen-vl-8B/}"
ALIGN_CKPT="${ALIGN_CKPT:-records/pipeline_runs/exp_hangzhou_real/three_stage_20260520_035801/stage1_gated_ep10_lr7e5_temp010_bs24/best.pt}"
DATA_PATH="${DATA_PATH:-records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl}"
STAGE_ROOT_BASE="${STAGE_ROOT_BASE:-records/pipeline_runs/exp_hangzhou_real}"
RUN_NAME="${RUN_NAME:-stage2_w2_guarded_online3600_$(date +%Y%m%d_%H%M%S)}"
STAGE_ROOT="${STAGE_ROOT_BASE}/${RUN_NAME}"
REPORT_MD="${REPORT_MD:-${STAGE_ROOT}/report.md}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-2024}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing TORCHRUN_BIN=${TORCHRUN_BIN}" >&2
  exit 2
fi
if [[ ! -d "${LLM_PATH}" ]]; then
  echo "Missing LLM_PATH=${LLM_PATH}" >&2
  exit 2
fi
if [[ ! -f "${ALIGN_CKPT}" ]]; then
  echo "Missing ALIGN_CKPT=${ALIGN_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing DATA_PATH=${DATA_PATH}" >&2
  exit 2
fi

CKPT_DIR="${STAGE_ROOT}/checkpoints"
LOG_DIR="${STAGE_ROOT}/logs"
ARTIFACT_DIR="${STAGE_ROOT}/artifacts"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${ARTIFACT_DIR}"

{
  echo "# Hangzhou Real Stage 2 W2 Guarded Run"
  echo
  echo "- started_at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "- stage_root: \`${STAGE_ROOT}\`"
  echo "- data_path: \`${DATA_PATH}\`"
  echo "- llm_path: \`${LLM_PATH}\`"
  echo "- align_ckpt: \`${ALIGN_CKPT}\`"
  echo "- run_counts: \`${RUN_COUNTS}\`"
  echo "- seed: \`${SEED}\`"
  echo
  echo "## W2 Training Recipe"
  echo
  echo "- soft_target_alpha: \`0.55\`"
  echo "- soft_label_temperature: \`0.55\`"
  echo "- global_action_weights: \`0.90,0.95,1.15,1.45\`"
  echo "- online selection guardrails: \`NLSL>=6%, max-phase-match>=45%, strong_misses<=60, no monotonic late-window queue rise\`"
  echo
} > "${REPORT_MD}"

TRAIN_ARGS=(
  --data_path "${DATA_PATH}"
  --llm_path "${LLM_PATH}"
  --align_ckpt "${ALIGN_CKPT}"
  --output_dir "${CKPT_DIR}"
  --seq_len 16
  --image_size 224
  --periodicity 16
  --input_dim 8
  --hidden_dim 64
  --prefix_tokens 8
  --prefix_mode prefix
  --max_length 64
  --prompt_mode image_only_short
  --batch_size 8
  --grad_accum_steps 8
  --epochs 14
  --lr 7e-5
  --min_lr 1e-5
  --warmup_ratio 0.05
  --lr_schedule cosine
  --weight_decay 1e-4
  --max_grad_norm 1.0
  --val_ratio 0.1
  --split_strategy blocked
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --bf16
  --save_every 1
  --target_mode mixed
  --soft_target_alpha 0.55
  --soft_label_temperature 0.55
  --disable_class_balance
  --global_action_weights 0.90,0.95,1.15,1.45
  --best_metric val_loss
  --best_mode min
)

echo "===== TRAIN ${RUN_NAME} =====" | tee -a "${REPORT_MD}"
"${TORCHRUN_BIN}" --nproc_per_node 1 projection_train/train_projection_bc.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "===== FULL ONLINE EVAL ${RUN_NAME} =====" | tee -a "${REPORT_MD}"
seen=""
for ckpt in "${CKPT_DIR}"/epoch_*.pt "${CKPT_DIR}"/best.pt "${CKPT_DIR}"/last.pt; do
  [[ -f "${ckpt}" ]] || continue
  case " ${seen} " in
    *" ${ckpt} "*) continue ;;
  esac
  seen="${seen} ${ckpt}"
  stem="$(basename "${ckpt}" .pt)"
  "${PYTHON_BIN}" projection_train/run_projection_online.py \
    --memo "pipeline_runs/exp_hangzhou_real/${RUN_NAME}/eval_${stem}" \
    --projection_ckpt "${ckpt}" \
    --dataset hangzhou \
    --traffic_file anon_4_4_hangzhou_real.json \
    --device auto \
    --seed "${SEED}" \
    --num_rounds 1 \
    --proj_name LLM-TSCS-hangzhou-stage2-w2 \
    --run_counts "${RUN_COUNTS}" \
    --result_json "${ARTIFACT_DIR}/full_eval_${stem}.json" \
    --llm_path "${LLM_PATH}" \
    --seq_len 16 \
    --image_size 224 \
    --periodicity 16 \
    --input_dim 8 \
    --hidden_dim 64 \
    --prefix_tokens 8 \
    --prefix_mode prefix \
    --max_length 64 \
    --prompt_mode image_only_short \
    --bf16 \
    2>&1 | tee "${LOG_DIR}/full_eval_${stem}.log"
done

echo "===== SELECT ONLINE BEST ${RUN_NAME} =====" | tee -a "${REPORT_MD}"
"${PYTHON_BIN}" scripts/select_projection_checkpoint.py \
  --stage_root "${STAGE_ROOT}" \
  --ranking_metric reward \
  --eval_glob "artifacts/full_eval_*.json" \
  --min_nlsl 6 \
  --min_max_phase_match 45 \
  --max_strong_misses 60 \
  --strong_miss_gap 10 \
  --forbid_monotonic_late_queue_rise \
  --promote_to_best \
  --output_json "${ARTIFACT_DIR}/online_checkpoint_selection.json" \
  2>&1 | tee "${LOG_DIR}/select_online_best.log"

echo "===== FINAL ONLINE TEST ${RUN_NAME} =====" | tee -a "${REPORT_MD}"
"${PYTHON_BIN}" projection_train/run_projection_online.py \
  --memo "pipeline_runs/exp_hangzhou_real/${RUN_NAME}/final_test" \
  --projection_ckpt "${CKPT_DIR}/best.pt" \
  --dataset hangzhou \
  --traffic_file anon_4_4_hangzhou_real.json \
  --device auto \
  --seed "${SEED}" \
  --num_rounds 1 \
  --proj_name LLM-TSCS-hangzhou-stage2-w2 \
  --run_counts "${RUN_COUNTS}" \
  --result_json "${ARTIFACT_DIR}/final_test_result.json" \
  --llm_path "${LLM_PATH}" \
  --seq_len 16 \
  --image_size 224 \
  --periodicity 16 \
  --input_dim 8 \
  --hidden_dim 64 \
  --prefix_tokens 8 \
  --prefix_mode prefix \
  --max_length 64 \
  --prompt_mode image_only_short \
  --bf16 \
  2>&1 | tee "${LOG_DIR}/final_test.log"

{
  echo
  echo "## Outputs"
  echo
  echo "- train_log: \`${LOG_DIR}/train.log\`"
  echo "- selection_log: \`${LOG_DIR}/select_online_best.log\`"
  echo "- selection_json: \`${ARTIFACT_DIR}/online_checkpoint_selection.json\`"
  echo "- final_result: \`${ARTIFACT_DIR}/final_test_result.json\`"
  echo
  echo "Finished at $(date '+%Y-%m-%d %H:%M:%S %z')"
} >> "${REPORT_MD}"
