#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/lijn99/miniconda3/envs/lab/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/lijn99/miniconda3/envs/lab/bin/torchrun}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LLM_PATH="${LLM_PATH:-/home/data/model/qwen-vl-8B}"
ALIGN_CKPT="${ALIGN_CKPT:-/data/datacode/records/align_qwenvl8b_bs16_ga4_lr1e-4_t007_pt8_rebalanced/align_epoch_40.pt}"
DATA_PATH="${DATA_PATH:-records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl}"
STAGE_ROOT_BASE="${STAGE_ROOT_BASE:-records/pipeline_runs/exp_hangzhou_real}"
REPORT_MD="${REPORT_MD:-${STAGE_ROOT_BASE}/stage2_optimization_report.md}"
SHORT_RUN_COUNTS="${SHORT_RUN_COUNTS:-1200}"
FINAL_RUN_COUNTS="${FINAL_RUN_COUNTS:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"

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

run_one() {
  local name="$1"
  local lr="$2"
  local soft_alpha="$3"
  local soft_temp="$4"
  local min_margin="$5"
  local stage_root="${STAGE_ROOT_BASE}/${name}"
  local ckpt_dir="${stage_root}/checkpoints"
  local log_dir="${stage_root}/logs"
  local artifact_dir="${stage_root}/artifacts"
  mkdir -p "${ckpt_dir}" "${log_dir}" "${artifact_dir}"

  local train_args=(
    --data_path "${DATA_PATH}"
    --llm_path "${LLM_PATH}"
    --align_ckpt "${ALIGN_CKPT}"
    --output_dir "${ckpt_dir}"
    --seq_len 16
    --image_size 224
    --periodicity 16
    --input_dim 8
    --hidden_dim 64
    --prefix_tokens 8
    --prefix_mode prefix
    --max_length 128
    --prompt_mode image_only_short
    --batch_size 8
    --grad_accum_steps 8
    --epochs 12
    --lr "${lr}"
    --min_lr 1e-5
    --warmup_ratio 0.1
    --lr_schedule cosine
    --weight_decay 1e-4
    --max_grad_norm 1.0
    --val_ratio 0.1
    --split_strategy blocked
    --num_workers "${NUM_WORKERS}"
    --seed 2024
    --bf16
    --save_every 1
    --target_mode mixed
    --soft_target_alpha "${soft_alpha}"
    --soft_label_temperature "${soft_temp}"
    --disable_class_balance
    --best_metric val_loss
    --best_mode min
  )
  if [[ "${min_margin}" != "none" ]]; then
    train_args+=(--min_action_margin "${min_margin}")
  fi

  echo "===== TRAIN ${name} ====="
  "${TORCHRUN_BIN}" --nproc_per_node 1 projection_train/train_projection_bc.py "${train_args[@]}" \
    2>&1 | tee "${log_dir}/train.log"

  echo "===== SHORT ONLINE EVAL ${name} ====="
  local seen=""
  for ckpt in "${ckpt_dir}"/epoch_*.pt "${ckpt_dir}"/best.pt "${ckpt_dir}"/last.pt; do
    [[ -f "${ckpt}" ]] || continue
    local stem
    stem="$(basename "${ckpt}" .pt)"
    case " ${seen} " in
      *" ${ckpt} "*) continue ;;
    esac
    seen="${seen} ${ckpt}"
    "${PYTHON_BIN}" projection_train/run_projection_online.py \
      --memo "pipeline_runs/exp_hangzhou_real/${name}/short_eval_${stem}" \
      --projection_ckpt "${ckpt}" \
      --dataset hangzhou \
      --traffic_file anon_4_4_hangzhou_real.json \
      --device auto \
      --seed 2024 \
      --num_rounds 1 \
      --proj_name LLM-TSCS-hangzhou-stage2-opt \
      --run_counts "${SHORT_RUN_COUNTS}" \
      --result_json "${artifact_dir}/short_eval_${stem}.json" \
      --llm_path "${LLM_PATH}" \
      --seq_len 16 \
      --image_size 224 \
      --periodicity 16 \
      --input_dim 8 \
      --hidden_dim 64 \
      --prefix_tokens 8 \
      --prefix_mode prefix \
      --max_length 128 \
      --prompt_mode image_only_short \
      --bf16 \
      2>&1 | tee "${log_dir}/short_eval_${stem}.log"
  done

  echo "===== SELECT ONLINE BEST ${name} ====="
  "${PYTHON_BIN}" scripts/select_projection_checkpoint.py \
    --stage_root "${stage_root}" \
    --ranking_metric reward \
    --eval_glob "artifacts/short_eval*.json" \
    --promote_to_best \
    --fallback_to_unconstrained \
    --output_json "${artifact_dir}/online_checkpoint_selection.json" \
    2>&1 | tee "${log_dir}/select_online_best.log"

  echo "===== FINAL ONLINE TEST ${name} ====="
  "${PYTHON_BIN}" projection_train/run_projection_online.py \
    --memo "pipeline_runs/exp_hangzhou_real/${name}/final_test" \
    --projection_ckpt "${ckpt_dir}/best.pt" \
    --dataset hangzhou \
    --traffic_file anon_4_4_hangzhou_real.json \
    --device auto \
    --seed 2024 \
    --num_rounds 1 \
    --proj_name LLM-TSCS-hangzhou-stage2-opt \
    --run_counts "${FINAL_RUN_COUNTS}" \
    --result_json "${artifact_dir}/final_test_result.json" \
    --llm_path "${LLM_PATH}" \
    --seq_len 16 \
    --image_size 224 \
    --periodicity 16 \
    --input_dim 8 \
    --hidden_dim 64 \
    --prefix_tokens 8 \
    --prefix_mode prefix \
    --max_length 128 \
    --prompt_mode image_only_short \
    --bf16 \
    2>&1 | tee "${log_dir}/final_test.log"

  {
    echo "## ${name}"
    echo
    echo "- stage_root: \`${stage_root}\`"
    echo "- train_log: \`${log_dir}/train.log\`"
    echo "- selection: \`${artifact_dir}/online_checkpoint_selection.json\`"
    echo "- final_result: \`${artifact_dir}/final_test_result.json\`"
    echo
  } >> "${REPORT_MD}"
}

mkdir -p "$(dirname "${REPORT_MD}")"
{
  echo "# Hangzhou Real Stage 2 Optimization"
  echo
  echo "- started_at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "- data_path: \`${DATA_PATH}\`"
  echo "- llm_path: \`${LLM_PATH}\`"
  echo "- align_ckpt: \`${ALIGN_CKPT}\`"
  echo "- cuda_device: \`${CUDA_DEVICE}\`"
  echo "- short_run_counts: \`${SHORT_RUN_COUNTS}\`"
  echo
} > "${REPORT_MD}"

run_one "nobal_lr7e5_a02_t07_ep12" "7e-5" "0.2" "0.7" "none"
run_one "nobal_lr1e4_a02_t07_ep12" "1e-4" "0.2" "0.7" "none"
run_one "nobal_margin05_lr7e5_a02_t07_ep12" "7e-5" "0.2" "0.7" "0.5"
run_one "nobal_lr7e5_a035_t07_ep12" "7e-5" "0.35" "0.7" "none"

echo "Finished at $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${REPORT_MD}"
