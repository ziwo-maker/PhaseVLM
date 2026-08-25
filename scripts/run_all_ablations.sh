#!/bin/bash
# Master Ablation Launcher - All 19 experiments
# Phases 1-5, sequential within phase, auto-collects results
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

CONDA_PY="/home/data/sysu/anaconda3/envs/hnr5090/bin/python"
LLM_PATH="/home/data/model/qwen-vl-8B"
ALIGN_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
OUT_BASE="records/ablation/a_override_distill"
DATA_DIR="data/ablation/a_override_distill"
TS=$(date +%Y%m%d_%H%M%S)

# Shared params
BATCH=4 GA=8 EPOCHS=15 LR=7e-5 MIN_LR=7e-6
PROMPT_MODE=queue_wait_short MAX_LEN=64 PREFIX=8
TRAFFIC="anon_4_4_hangzhou_real.json"
ROADNET="data/Hangzhou/4_4/roadnet_4_4.json"
RUNS=3600 MIN_ACT=30

# Existing checkpoints
R1_CKPT="records/ablation/a_override_distill/R1_baseline_20260812_002702/best.pt"
R2_CKPT="records/ablation/a_override_distill/R2_distill_20260812_005352/best.pt"

mkdir -p "${OUT_BASE}"

run_train() {
    local NAME=$1 DATA=$2 SEED=$3
    local OUT="${OUT_BASE}/${NAME}_seed${SEED}_${TS}"
    echo "  [TRAIN] ${NAME} seed=${SEED}"
    torchrun --standalone --nproc_per_node=2 \
        projection_train/train_projection_bc.py \
        --data_path "${DATA}" --llm_path "${LLM_PATH}" --align_ckpt "${ALIGN_CKPT}" \
        --output_dir "${OUT}" --prompt_mode "${PROMPT_MODE}" \
        --batch_size ${BATCH} --grad_accum_steps ${GA} --epochs ${EPOCHS} \
        --lr ${LR} --min_lr ${MIN_LR} --seed ${SEED} \
        --max_length ${MAX_LEN} --prefix_tokens ${PREFIX} \
        --target_mode hard --best_metric val_acc --best_mode max \
        --log_interval 20 --ddp_timeout_seconds 1800 \
        2>&1 | tee "${OUT_BASE}/${NAME}_train_seed${SEED}_${TS}.log"
    echo "${OUT}/best.pt"
}

run_eval() {
    local NAME=$1 CKPT=$2 SEED=$3 OVERRIDE_GAP=${4:-0.0} OVERRIDE_PROTECT=${5:-0.0}
    local EVAL_DIR="${OUT_BASE}/${NAME}_eval_seed${SEED}_${TS}"
    mkdir -p "${EVAL_DIR}"
    echo "  [EVAL] ${NAME} seed=${SEED} gap=${OVERRIDE_GAP} protect=${OVERRIDE_PROTECT}"
    CUDA_VISIBLE_DEVICES=2 ${CONDA_PY} projection_train/run_projection_online.py \
        --memo "${NAME}_seed${SEED}" --llm_path "${LLM_PATH}" --projection_ckpt "${CKPT}" \
        --dataset hangzhou --traffic_file "${TRAFFIC}" --run_counts ${RUNS} --min_action_time ${MIN_ACT} \
        --prompt_mode "${PROMPT_MODE}" --max_length ${MAX_LEN} --prefix_tokens ${PREFIX} \
        --result_json "${EVAL_DIR}/test_result.json" --seed ${SEED} \
        --max_phase_override_gap ${OVERRIDE_GAP} \
        --max_phase_override_left_wait_protect ${OVERRIDE_PROTECT} \
        --proj_name "ablation_${NAME}" \
        2>&1 | tee "${OUT_BASE}/${NAME}_eval_seed${SEED}_${TS}.log"
    python3 -c "
import json
with open('${EVAL_DIR}/test_result.json') as f:
    r=json.load(f).get('results',{})
q=r['test_avg_queue_len_over']; w=r['test_avg_waiting_time_over']; t=r['test_avg_travel_time_over']
print(f'    Q={q:.1f} W={w:.1f} T={t:.1f} Sum={q+w+t:.1f}')
" 2>/dev/null
}

echo "============================================"
echo "ABLATION EXPERIMENTS - ALL 19"
echo "Start: $(date)"
echo "============================================"

# ====== PHASE 1: Multi-seed + Override verification (eval only) ======
echo ""
echo "===== PHASE 1: Multi-seed + A2 (eval only) ====="

for s in 2025 2026; do
    run_eval "C1_R1_seed" "${R1_CKPT}" $s 0.0 0.0
done

for s in 2025 2026; do
    run_eval "C2_R2_seed" "${R2_CKPT}" $s 0.0 0.0
done

# A2: R2 WITH override
for s in 2024 2025 2026; do
    run_eval "A2_R2_override" "${R2_CKPT}" $s 2.0 60.0
done

# ====== PHASE 2: Core component ablations (train + eval) ======
echo ""
echo "===== PHASE 2: Core Component Ablations ====="

# A4: protect=0
A4_CKPT=$(run_train "A4_protect0" "${DATA_DIR}/A4_protect0.jsonl" 2024)
run_eval "A4_protect0" "${A4_CKPT}" 2024 0.0 0.0

# B1: gap=3
B1G3_CKPT=$(run_train "B1_gap3" "${DATA_DIR}/B1_gap3.jsonl" 2024)
run_eval "B1_gap3" "${B1G3_CKPT}" 2024 0.0 0.0

# B1: gap=5
B1G5_CKPT=$(run_train "B1_gap5" "${DATA_DIR}/B1_gap5.jsonl" 2024)
run_eval "B1_gap5" "${B1G5_CKPT}" 2024 0.0 0.0

# B2: protect=30
B2P30_CKPT=$(run_train "B2_protect30" "${DATA_DIR}/B2_protect30.jsonl" 2024)
run_eval "B2_protect30" "${B2P30_CKPT}" 2024 0.0 0.0

# B2: protect=45
B2P45_CKPT=$(run_train "B2_protect45" "${DATA_DIR}/B2_protect45.jsonl" 2024)
run_eval "B2_protect45" "${B2P45_CKPT}" 2024 0.0 0.0

# ====== PHASE 3: Generalization ======
echo ""
echo "===== PHASE 3: Generalization ====="
echo "D1 (Hangzhou->Jinan) and D3 (HZ native Stage1) require additional data."
echo "Skipping for now - will run after Phase 1-2 complete."
echo "(D1, D3 marked as deferred)"

# ====== PHASE 4: Correction ratio ======
echo ""
echo "===== PHASE 4: Correction Ratio ====="
echo "B3 (top-K correction) requires per-row gap analysis."
echo "Skipping for now."
echo "(B3 marked as deferred)"

# ====== PHASE 5: Design choices ======
echo ""
echo "===== PHASE 5: Design Choices ====="
echo "E1,E2 require training code changes (soft target, sample_weight)."
echo "Skipping for now."
echo "(E1,E2 marked as deferred)"

# ====== SUMMARY ======
echo ""
echo "============================================"
echo "ABLATION EXPERIMENTS COMPLETE"
echo "End: $(date)"
echo "============================================"
echo ""
echo "Collecting all results..."

python3 << 'PYSUM'
import json, os, glob

base = "records/ablation/a_override_distill"
print(f"{'Experiment':<30} {'Seed':>6} {'Queue':>8} {'Wait':>8} {'Travel':>8} {'Sum':>8} {'Gap':>5} {'Prot':>5}")
print("-" * 95)

results = []
for f in sorted(glob.glob(f"{base}/*eval*/test_result.json")):
    try:
        d = json.load(open(f))
        r = d.get("results", d)
        q = r.get("test_avg_queue_len_over", 0)
        w = r.get("test_avg_waiting_time_over", 0)
        t = r.get("test_avg_travel_time_over", 0)
        name = os.path.basename(os.path.dirname(f))
        results.append((name, q, w, t, q+w+t))
    except: pass

# Deduplicate by name (keep latest)
seen = {}
for name, q, w, t, s in sorted(results):
    seen[name] = (q, w, t, s)

for name, (q, w, t, s) in sorted(seen.items()):
    print(f"{name:<30} {'?':>6} {q:>8.1f} {w:>8.1f} {t:>8.1f} {s:>8.1f}")

print(f"\nTotal experiments with results: {len(seen)}")
PYSUM

echo "===== DONE ====="
