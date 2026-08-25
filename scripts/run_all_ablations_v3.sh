#!/bin/bash
# Master Ablation Launcher v3 - Fixed stdout capture
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

CONDA_PY="/home/data/sysu/anaconda3/envs/hnr5090/bin/python"
LLM_PATH="/home/data/model/qwen-vl-8B"
ALIGN_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
OUT_BASE="records/ablation/a_override_distill"
DATA_DIR="data/ablation/a_override_distill"
TS=$(date +%Y%m%d_%H%M%S)

BATCH=4 GA=8 EPOCHS=15 LR=7e-5 MIN_LR=7e-6
PM=queue_wait_short ML=64 PT=8
TRAFFIC="anon_4_4_hangzhou_real.json"
RUNS=3600 MA=30

R1_CKPT="${OUT_BASE}/R1_baseline_20260812_002702/best.pt"
R2_CKPT="${OUT_BASE}/R2_distill_20260812_005352/best.pt"

mkdir -p "${OUT_BASE}"
CURRENT_CKPT=""

train() {
    local NAME=$1 DATA=$2 SEED=${3:-2024}
    local OUT="${OUT_BASE}/${NAME}_seed${SEED}_${TS}"
    echo "[TRAIN] ${NAME} seed=${SEED}"
    torchrun --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
        --data_path "${DATA}" --llm_path "${LLM_PATH}" --align_ckpt "${ALIGN_CKPT}" \
        --output_dir "${OUT}" --prompt_mode "${PM}" --batch_size ${BATCH} \
        --grad_accum_steps ${GA} --epochs ${EPOCHS} --lr ${LR} --min_lr ${MIN_LR} \
        --seed ${SEED} --max_length ${ML} --prefix_tokens ${PT} \
        --target_mode hard --best_metric val_acc --best_mode max \
        --log_interval 20 --ddp_timeout_seconds 1800 \
        2>&1 | tee "${OUT_BASE}/${NAME}_train_seed${SEED}_${TS}.log"
    CURRENT_CKPT="${OUT}/best.pt"
    echo "[TRAIN DONE] ${NAME} -> ${CURRENT_CKPT}"
}

eval_model() {
    local NAME=$1 CKPT=$2 SEED=${3:-2024} GAP=${4:-0.0} PROT=${5:-0.0}
    local EVAL_DIR="${OUT_BASE}/${NAME}_eval_seed${SEED}_${TS}"
    mkdir -p "${EVAL_DIR}"
    echo "[EVAL] ${NAME} seed=${SEED} gap=${GAP} prot=${PROT}"
    CUDA_VISIBLE_DEVICES=2 ${CONDA_PY} projection_train/run_projection_online.py \
        --memo "${NAME}_s${SEED}" --llm_path "${LLM_PATH}" --projection_ckpt "${CKPT}" \
        --dataset hangzhou --traffic_file "${TRAFFIC}" --run_counts ${RUNS} \
        --min_action_time ${MA} --prompt_mode "${PM}" --max_length ${ML} \
        --prefix_tokens ${PT} --result_json "${EVAL_DIR}/test_result.json" \
        --seed ${SEED} --max_phase_override_gap ${GAP} \
        --max_phase_override_left_wait_protect ${PROT} \
        --proj_name "abl_${NAME}" \
        2>&1 | tee "${OUT_BASE}/${NAME}_eval_seed${SEED}_${TS}.log"
    python3 -c "
import json; r=json.load(open('${EVAL_DIR}/test_result.json')).get('results',{})
print('    Q={:.1f} W={:.1f} T={:.1f} Sum={:.1f}'.format(
    r['test_avg_queue_len_over'], r['test_avg_waiting_time_over'], r['test_avg_travel_time_over'],
    r['test_avg_queue_len_over']+r['test_avg_waiting_time_over']+r['test_avg_travel_time_over']))
"
}

echo "===== ALL ABLATIONS v3 $(date) ====="

# PHASE 2: A4 protect=0
train "A4_protect0" "${DATA_DIR}/A4_protect0.jsonl" 2024
eval_model "A4_protect0" "${CURRENT_CKPT}" 2024 0.0 0.0

# B1 gap=3
train "B1_gap3" "${DATA_DIR}/B1_gap3.jsonl" 2024
eval_model "B1_gap3" "${CURRENT_CKPT}" 2024 0.0 0.0

# B1 gap=5
train "B1_gap5" "${DATA_DIR}/B1_gap5.jsonl" 2024
eval_model "B1_gap5" "${CURRENT_CKPT}" 2024 0.0 0.0

# B2 protect=30
train "B2_protect30" "${DATA_DIR}/B2_protect30.jsonl" 2024
eval_model "B2_protect30" "${CURRENT_CKPT}" 2024 0.0 0.0

# B2 protect=45
train "B2_protect45" "${DATA_DIR}/B2_protect45.jsonl" 2024
eval_model "B2_protect45" "${CURRENT_CKPT}" 2024 0.0 0.0

# A2: Override verification multi-seed
for s in 2024 2025 2026; do
    eval_model "A2_R2_override" "${R2_CKPT}" $s 2.0 60.0
done

# SUMMARY
echo ""
echo "===== COMPLETE $(date) ====="
python3 << 'PYSUM'
import json, os, glob
base = "records/ablation/a_override_distill"
print(f"{'Experiment':<35} {'Q':>7} {'W':>7} {'T':>7} {'Sum':>8}")
print("-" * 65)
seen = {}
for f in sorted(glob.glob(f"{base}/*eval*/test_result.json")):
    try:
        r = json.load(open(f)).get("results", {})
        q = r["test_avg_queue_len_over"]
        w = r["test_avg_waiting_time_over"]
        t = r["test_avg_travel_time_over"]
        name = os.path.basename(os.path.dirname(f))[:33]
        seen[name] = (q, w, t)
    except: pass
for name, (q, w, t) in sorted(seen.items()):
    print(f"  {name:<33} {q:>7.1f} {w:>7.1f} {t:>7.1f} {q+w+t:>8.1f}")
PYSUM
echo "===== DONE ====="
