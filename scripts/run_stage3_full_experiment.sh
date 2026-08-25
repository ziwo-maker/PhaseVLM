#!/bin/bash
# Stage 3 Cross-Network Adaptation: Full Experiment Launcher
# Generated: 2026-08-10
# Ablations: A (RuleOnly+HardBC), B (SimExpert+HardBC), C (SimExpert+Soft), D (SimExpert+Pareto)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Use conda env with cityflow + PyTorch
source /home/sysu/anaconda3/etc/profile.d/conda.sh
conda activate hnr5090
export PYTHON="$(which python3)"
echo "Using Python: $PYTHON"

TIMESTAMP="20260810_204128"
BASE_OUTPUT="records/pipeline_runs/stage3_ablation_${TIMESTAMP}"
COLLECTED="${BASE_OUTPUT}/collected_states.jsonl"

SOURCE_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt"
STAGE1_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
LLM_PATH="/home/data/model/qwen-vl-8B"
TRAIN_GPUS="0,1"
COLLECTION_FLOW="anon_4_4_hangzhou_real.json"
TEST_FLOW="anon_4_4_hangzhou_real_5816.json"
LABEL_BUDGETS="8 32 120"
SEEDS="2024 42 123"
SIM_HORIZON=30

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ================================================================
# Phase 2: Zero-shot evaluation
# ================================================================
log "=== Phase 2: Zero-Shot Evaluation ==="
ZERO_SHOT_DIR="${BASE_OUTPUT}/zero_shot"
if [ -f "${ZERO_SHOT_DIR}/all_seeds.json" ]; then
    log "Zero-shot results exist, skipping"
else
    mkdir -p "${ZERO_SHOT_DIR}"
    for seed in ${SEEDS}; do
        log "Zero-shot seed=${seed}"
        python3 projection_train/run_projection_online.py \
            --memo "stage3_ablation_zs" \
            --projection_ckpt "${SOURCE_CKPT}" \
            --llm_path "${LLM_PATH}" \
            --dataset hangzhou \
            --traffic_file "${TEST_FLOW}" \
            --run_counts 3600 --num_rounds 1 \
            --seed "${seed}" \
            --prompt_mode queue_wait_short \
            --seq_len 16 --image_size 224 \
            --periodicity 16 --input_dim 8 --hidden_dim 64 \
            --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
            --inference_batch_size 64 \
            --bf16 \
            --result_json "${ZERO_SHOT_DIR}/result_seed${seed}.json" \
            >> "${BASE_OUTPUT}/pipeline.log" 2>&1
    done
    # Merge results
    python3 -c "
import json, os
results = {}
for seed in [2024, 42, 123]:
    path = f'${ZERO_SHOT_DIR}/result_seed{seed}.json'
    if os.path.exists(path):
        with open(path) as f:
            payload = json.load(f)
        r = payload.get('results', payload)
        results[f'seed_{seed}'] = {
            'ATT': r.get('test_avg_travel_time_over'),
            'AQL': r.get('test_avg_queue_len_over'),
            'AWT': r.get('test_avg_waiting_time_over'),
        }
with open('${ZERO_SHOT_DIR}/all_seeds.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Zero-shot merged:', json.dumps(results, indent=2))
" 2>&1 | tee -a "${BASE_OUTPUT}/pipeline.log"
fi
log "Phase 2 complete"

# ================================================================
# Phase 4a: RuleOnly relabeling (for ablation A)
# ================================================================
log "=== Phase 4a: RuleOnly Relabeling (Ablation A) ==="
A_DIR="${BASE_OUTPUT}/A"
mkdir -p "${A_DIR}"
RELABELED_A="${A_DIR}/relabeled_states.jsonl"

if [ -f "${RELABELED_A}" ]; then
    log "RuleOnly relabeled data exists, skipping"
else
    log "Running RuleOnly relabeling..."
    python3 -c "
import json, os
from projection_train.action_space import ACTION_LIST

states_path = '${COLLECTED}'
output_path = '${RELABELED_A}'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cost_aql = 1.0
cost_awt = 2.0

count = 0
with open(states_path) as fin, open(output_path, 'w') as fout:
    for line in fin:
        if not line.strip(): continue
        rec = json.loads(line)
        pq = rec.get('phase_queue', {})
        pw = rec.get('phase_wait', {})
        best = ACTION_LIST[0]
        best_score = -1.0
        for a in ACTION_LIST:
            q = float(pq.get(a, 0) or 0)
            w = float(pw.get(a, 0) or 0)
            s = cost_aql * q + cost_awt * w
            if s > best_score:
                best_score = s
                best = a
        rec['expert_action'] = best
        rec['relabel_method'] = 'ruleonly_awt2.0_aql1.0'
        rec['candidate_action_metrics'] = {}
        rec['candidate_action_waits'] = {}
        rec['expert_action_margin'] = 0.0
        rec['base_action'] = rec.get('policy_action', ACTION_LIST[0])
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        count += 1
print(f'RuleOnly relabeled {count} records -> {output_path}')
" 2>&1 | tee -a "${BASE_OUTPUT}/pipeline.log"
fi
log "RuleOnly relabeling complete"

# ================================================================
# Phase 4b: SimExpert relabeling (shared for ablations B/C/D)
# ================================================================
log "=== Phase 4b: SimExpert Relabeling (Ablations B/C/D) ==="
RELABELED_SE="${BASE_OUTPUT}/relabeled_simexpert.jsonl"

if [ -f "${RELABELED_SE}" ]; then
    log "SimExpert relabeled data exists, skipping"
else
    log "Running SimExpert relabeling (this takes ~30 min)..."
    source /home/sysu/anaconda3/etc/profile.d/conda.sh
    conda activate hnr5090
    python3 scripts/relabel_simexpert.py \
        --input "${COLLECTED}" \
        --output "${RELABELED_SE}" \
        --roadnet data/Hangzhou/4_4/roadnet_4_4.json \
        --traffic_file data/Hangzhou/4_4/anon_4_4_hangzhou_real.json \
        --dataset hangzhou \
        --work_dir "${BASE_OUTPUT}/simexpert_work" \
        --sim_horizon "${SIM_HORIZON}" \
        --cost_att_weight 1.0 \
        --cost_aql_weight 1.0 \
        --cost_awt_weight 1.0 \
        --cost_spillback_weight 0.2 \
        --awt_constraint_epsilon 5.0 \
        --spillback_constraint_epsilon 0.05 \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
    log "SimExpert relabeling complete"
fi

# ================================================================
# Phase 5+6: Adaptation + Eval for each ablation
# ================================================================
run_adaptation() {
    local ABLATION="$1"
    local DATA_PATH="$2"
    local TARGET_MODE="$3"
    local RANKING_WEIGHT="$4"
    local SOFT_ALPHA="$5"
    local BEST_METRIC="$6"
    local EXTRA_FLAGS="$7"

    local ADAPT_DIR="${BASE_OUTPUT}/${ABLATION}"

    for K in ${LABEL_BUDGETS}; do
        log "=== ${ABLATION} K=${K} target_mode=${TARGET_MODE} ==="
        local K_DIR="${ADAPT_DIR}/adapted_k${K}"

        if [ -f "${K_DIR}/evaluation/eval_summary.json" ]; then
            log "${ABLATION} K=${K} already done, skipping"
            continue
        fi

        mkdir -p "${K_DIR}"

        # Subsample K snapshots
        log "Subsampling K=${K}..."
        python3 -c "
import json, random, os
random.seed(2024)
with open('${DATA_PATH}') as f:
    all_recs = [json.loads(l) for l in f if l.strip()]
time_groups = {}
for rec in all_recs:
    t = rec.get('time_s', 0.0)
    time_groups.setdefault(t, []).append(rec)
times = sorted(time_groups.keys())
selected = sorted(random.sample(times, min(${K}, len(times))))
selected_recs = []
for t in selected:
    selected_recs.extend(time_groups[t])
sub_path = '${K_DIR}/train_data.jsonl'
with open(sub_path, 'w') as f:
    for rec in selected_recs:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(f'Subsampled K=${K}: {len(selected)} snapshots -> {len(selected_recs)} records (out of {len(times)} available)')
" 2>&1 | tee -a "${BASE_OUTPUT}/pipeline.log"

        # Get world size from GPU string
        local WORLD_SIZE=$(echo "${TRAIN_GPUS}" | tr ',' '\n' | wc -l)

        # Training
        log "Training ${ABLATION} K=${K}..."
        CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --nproc_per_node=${WORLD_SIZE} \
            projection_train/train_projection_bc.py \
            --data_path "${K_DIR}/train_data.jsonl" \
            --llm_path "${LLM_PATH}" \
            --align_ckpt "${STAGE1_CKPT}" \
            --init_policy_ckpt "${SOURCE_CKPT}" \
            --output_dir "${K_DIR}/checkpoints" \
            --seq_len 16 --image_size 224 \
            --periodicity 16 --input_dim 8 --hidden_dim 64 \
            --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
            --prompt_mode queue_wait_short \
            --batch_size 2 --grad_accum_steps 8 \
            --epochs 5 \
            --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine \
            --weight_decay 1e-4 --max_grad_norm 1.0 \
            --val_ratio 0.1 --split_strategy stratified \
            --num_workers 2 --seed 2024 --bf16 \
            --log_interval 10 --save_every 1 \
            --best_metric "${BEST_METRIC}" \
            --best_mode min \
            --ddp_timeout_seconds 1800 \
            --target_mode "${TARGET_MODE}" \
            --soft_target_alpha "${SOFT_ALPHA}" \
            --soft_label_temperature 0.35 \
            --ranking_loss_weight "${RANKING_WEIGHT}" \
            --cost_awt_weight 1.0 \
            --cost_aql_weight 1.0 \
            --cost_att_weight 1.0 \
            --cost_spillback_weight 0.2 \
            --ranking_min_cost_gap 0.02 \
            --base_policy_kl_weight 0.0 \
            ${EXTRA_FLAGS} \
            >> "${BASE_OUTPUT}/pipeline.log" 2>&1

        local BEST_PT="${K_DIR}/checkpoints/best.pt"
        if [ ! -f "${BEST_PT}" ]; then
            log "ERROR: ${ABLATION} K=${K} training failed (no best.pt)"
            continue
        fi

        # Evaluation
        log "Evaluating ${ABLATION} K=${K}..."
        local EVAL_DIR="${K_DIR}/evaluation"
        mkdir -p "${EVAL_DIR}"

        for seed in ${SEEDS}; do
            python3 projection_train/run_projection_online.py \
                --memo "stage3_${ABLATION}_k${K}" \
                --projection_ckpt "${BEST_PT}" \
                --llm_path "${LLM_PATH}" \
                --dataset hangzhou \
                --traffic_file "${TEST_FLOW}" \
                --run_counts 3600 --num_rounds 1 \
                --seed "${seed}" \
                --prompt_mode queue_wait_short \
                --seq_len 16 --image_size 224 \
                --periodicity 16 --input_dim 8 --hidden_dim 64 \
                --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
                --inference_batch_size 64 \
                --bf16 \
                --result_json "${EVAL_DIR}/result_seed${seed}.json" \
                >> "${BASE_OUTPUT}/pipeline.log" 2>&1
        done

        # Merge evaluation results with statistics
        python3 -c "
import json, os, numpy as np
from scipy import stats as scipy_stats

eval_dir = '${EVAL_DIR}'
results = {}
metric_keys = {
    'ATT': 'test_avg_travel_time_over',
    'AQL': 'test_avg_queue_len_over',
    'AWT': 'test_avg_waiting_time_over',
}
all_values = {m: [] for m in metric_keys}
for seed in [2024, 42, 123]:
    path = os.path.join(eval_dir, f'result_seed{seed}.json')
    if os.path.exists(path):
        with open(path) as f:
            payload = json.load(f)
        r = payload.get('results', payload)
        for label, key in metric_keys.items():
            all_values[label].append(float(r.get(key, 0)))

summary = {}
for metric, values in all_values.items():
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_val = scipy_stats.t.ppf(0.975, n - 1)
        ci_half = t_val * std_v / np.sqrt(n)
    else:
        ci_half = 0.0
    summary[metric] = {
        'mean': round(mean_v, 2), 'std': round(std_v, 2), 'n': n,
        'ci95_low': round(mean_v - ci_half, 2),
        'ci95_high': round(mean_v + ci_half, 2),
    }

with open(os.path.join(eval_dir, 'eval_summary.json'), 'w') as f:
    json.dump({'ablation': '${ABLATION}', 'K': ${K}, 'summary': summary, 'raw_values': {m: [round(v,2) for v in vs] for m, vs in all_values.items()}}, f, indent=2)
print(f'${ABLATION} K=${K} eval: ' + json.dumps(summary, indent=2))
" 2>&1 | tee -a "${BASE_OUTPUT}/pipeline.log"

        log "${ABLATION} K=${K} complete"
    done
}

# ---- Run all ablations ----
log "============================================="
log "Starting Ablation Matrix"
log "============================================="

# A: RuleOnly + Hard BC
run_adaptation "A" "${RELABELED_A}" "hard" "0.0" "0.0" "val_acc" ""

# B: SimExpert + Hard BC
run_adaptation "B" "${RELABELED_SE}" "hard" "0.0" "0.0" "val_acc" ""

# C: SimExpert + Soft preference distillation
run_adaptation "C" "${RELABELED_SE}" "soft" "0.0" "0.3" "val_top1_cost_regret" ""

# D: SimExpert + Mixed (hard+soft+ranking) with Pareto
run_adaptation "D" "${RELABELED_SE}" "mixed" "0.05" "0.3" "val_top1_cost_regret" ""

# ================================================================
# Final Summary
# ================================================================
log "============================================="
log "Stage 3 Ablation Matrix Complete!"
log "Results: ${BASE_OUTPUT}"
log "============================================="

# Generate summary
python3 -c "
import json, os, glob

base = '${BASE_OUTPUT}'
print()
print('=' * 60)
print('FINAL RESULTS SUMMARY')
print('=' * 60)

for ablation in ['A', 'B', 'C', 'D']:
    print(f'\n--- Ablation {ablation} ---')
    for k in [8, 32, 120]:
        summary_path = f'{base}/{ablation}/adapted_k{k}/evaluation/eval_summary.json'
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            s = data.get('summary', {})
            print(f'  K={k}: ATT={s.get(\"ATT\",{}).get(\"mean\",\"?\")} '
                  f'AQL={s.get(\"AQL\",{}).get(\"mean\",\"?\")} '
                  f'AWT={s.get(\"AWT\",{}).get(\"mean\",\"?\")}')
        else:
            print(f'  K={k}: NOT FOUND')
" 2>&1 | tee -a "${BASE_OUTPUT}/pipeline.log"
