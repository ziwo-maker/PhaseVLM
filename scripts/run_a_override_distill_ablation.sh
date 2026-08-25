#!/bin/bash
# A-Override Distillation Ablation Experiment
# R1: BC on uncorrected data (baseline) -> eval WITHOUT A-override
# R2: BC on override-corrected data (distill) -> eval WITHOUT A-override
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LLM_PATH="${LLM_PATH:-/home/data/model/qwen-vl-8B}"
ALIGN_CKPT="${ALIGN_CKPT:-records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt}"
OUT_BASE="records/ablation/a_override_distill"
TS=$(date +%Y%m%d_%H%M%S)

R1_DATA="data/ablation/a_override_distill/R1_baseline_uncorrected.jsonl"
R2_DATA="data/ablation/a_override_distill/R2_distill_corrected.jsonl"

# Shared params
BATCH=4 GA=8 EPOCHS=15 LR=7e-5 MIN_LR=7e-6 SEED=2024
PROMPT_MODE=queue_wait_short MAX_LEN=64 PREFIX=8

# Eval params
TRAFFIC="anon_4_4_hangzhou_real.json"
ROADNET="data/Hangzhou/4_4/roadnet_4_4.json"
RUNS=3600 MIN_ACT=30

mkdir -p "${OUT_BASE}"

echo "===== A-Override Distill Ablation ${TS} ====="

# ---- R1: Baseline (uncorrected) ----
R1_OUT="${OUT_BASE}/R1_baseline_${TS}"
echo "[$(date +%H:%M:%S)] R1 Training (uncorrected data)..."
torchrun --standalone --nproc_per_node=2 \
    projection_train/train_projection_bc.py \
    --data_path "${R1_DATA}" \
    --llm_path "${LLM_PATH}" \
    --align_ckpt "${ALIGN_CKPT}" \
    --output_dir "${R1_OUT}" \
    --prompt_mode "${PROMPT_MODE}" \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --epochs ${EPOCHS} --lr ${LR} --min_lr ${MIN_LR} --seed ${SEED} \
    --max_length ${MAX_LEN} --prefix_tokens ${PREFIX} \
    --target_mode hard --best_metric val_acc --best_mode max \
    --log_interval 20 --ddp_timeout_seconds 1800 \
    2>&1 | tee "${OUT_BASE}/R1_train_${TS}.log"
echo "[$(date +%H:%M:%S)] R1 training done."

R1_CKPT="${R1_OUT}/best.pt"
echo "[$(date +%H:%M:%S)] R1 Eval (NO override)..."
R1_EVAL_DIR="${OUT_BASE}/R1_eval_${TS}"
mkdir -p "${R1_EVAL_DIR}"
CUDA_VISIBLE_DEVICES=2 /home/data/sysu/anaconda3/envs/hnr5090/bin/python projection_train/run_projection_online.py \
    --memo "R1_baseline_eval" \
    --llm_path "${LLM_PATH}" --projection_ckpt "${R1_CKPT}" \
    --dataset hangzhou --traffic_file "${TRAFFIC}" \
    --run_counts ${RUNS} --min_action_time ${MIN_ACT} \
    --prompt_mode "${PROMPT_MODE}" --max_length ${MAX_LEN} --prefix_tokens ${PREFIX} \
    --result_json "${R1_EVAL_DIR}/test_result.json" \
    --max_phase_override_gap 0.0 \
    --proj_name "ablation_a_override_R1" \
    2>&1 | tee "${OUT_BASE}/R1_eval_${TS}.log"
echo "[$(date +%H:%M:%S)] R1 eval done."

# ---- R2: Distill (corrected) ----
R2_OUT="${OUT_BASE}/R2_distill_${TS}"
echo "[$(date +%H:%M:%S)] R2 Training (corrected data)..."
torchrun --standalone --nproc_per_node=2 \
    projection_train/train_projection_bc.py \
    --data_path "${R2_DATA}" \
    --llm_path "${LLM_PATH}" \
    --align_ckpt "${ALIGN_CKPT}" \
    --output_dir "${R2_OUT}" \
    --prompt_mode "${PROMPT_MODE}" \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --epochs ${EPOCHS} --lr ${LR} --min_lr ${MIN_LR} --seed ${SEED} \
    --max_length ${MAX_LEN} --prefix_tokens ${PREFIX} \
    --target_mode hard --best_metric val_acc --best_mode max \
    --log_interval 20 --ddp_timeout_seconds 1800 \
    2>&1 | tee "${OUT_BASE}/R2_train_${TS}.log"
echo "[$(date +%H:%M:%S)] R2 training done."

R2_CKPT="${R2_OUT}/best.pt"
echo "[$(date +%H:%M:%S)] R2 Eval (NO override)..."
R2_EVAL_DIR="${OUT_BASE}/R2_eval_${TS}"
mkdir -p "${R2_EVAL_DIR}"
CUDA_VISIBLE_DEVICES=2 /home/data/sysu/anaconda3/envs/hnr5090/bin/python projection_train/run_projection_online.py \
    --memo "R2_distill_eval" \
    --llm_path "${LLM_PATH}" --projection_ckpt "${R2_CKPT}" \
    --dataset hangzhou --traffic_file "${TRAFFIC}" \
    --run_counts ${RUNS} --min_action_time ${MIN_ACT} \
    --prompt_mode "${PROMPT_MODE}" --max_length ${MAX_LEN} --prefix_tokens ${PREFIX} \
    --result_json "${R2_EVAL_DIR}/test_result.json" \
    --max_phase_override_gap 0.0 \
    --proj_name "ablation_a_override_R2" \
    2>&1 | tee "${OUT_BASE}/R2_eval_${TS}.log"
echo "[$(date +%H:%M:%S)] R2 eval done."

# ---- Summary ----
echo ""
echo "===== RESULTS ====="
echo "R1 (baseline, uncorrected):"
python3 -c "
import json
try:
    with open('${OUT_BASE}/R1_eval_${TS}/test_result.json') as f:
        d=json.load(f)
    r=d.get('results',d)
    print(f'  Queue={r[\"test_avg_queue_len_over\"]:.2f} Wait={r[\"test_avg_waiting_time_over\"]:.2f} Travel={r[\"test_avg_travel_time_over\"]:.2f} Reward={r[\"test_reward_over\"]:.2f}')
except Exception as e:
    print(f'  Error: {e}')
"
echo "R2 (distill, corrected):"
python3 -c "
import json
try:
    with open('${OUT_BASE}/R2_eval_${TS}/test_result.json') as f:
        d=json.load(f)
    r=d.get('results',d)
    print(f'  Queue={r[\"test_avg_queue_len_over\"]:.2f} Wait={r[\"test_avg_waiting_time_over\"]:.2f} Travel={r[\"test_avg_travel_time_over\"]:.2f} Reward={r[\"test_reward_over\"]:.2f}')
except Exception as e:
    print(f'  Error: {e}')
"
echo "===== DONE ====="
