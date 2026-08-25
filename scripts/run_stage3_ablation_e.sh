#!/bin/bash
# Stage 3 Ablation E: Topology-Conditioned Adapter + Meta-Training
# E1: SimExpert+Pareto (D) + topology adapter
# E2: Joint Jinan+Hangzhou pre-training + few-shot adapt
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source /home/sysu/anaconda3/etc/profile.d/conda.sh
conda activate hnr5090
export PYTHON="$(which python3)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUTPUT="records/pipeline_runs/stage3_ablation_e_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT}"

SOURCE_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt"
STAGE1_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
LLM_PATH="/home/data/model/qwen-vl-8B"
TRAIN_GPUS="0,1"
LABEL_BUDGETS="8 32 120"
SEEDS="2024 42 123"
SIM_HORIZON=30

# Topology config: both Jinan (3x4) and Hangzhou (4x4)
TOPO_CONFIG='{"Jinan-3_4":{"rows":3,"cols":4},"Hangzhou-4_4":{"rows":4,"cols":4}}'

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Reuse existing SimExpert relabeled data ----
HZ_RELABELED="records/pipeline_runs/stage3_ablation_20260810_204128/relabeled_simexpert.jsonl"
HZ_COLLECTED="records/pipeline_runs/stage3_ablation_20260810_204128/collected_states.jsonl"

if [ ! -f "$HZ_RELABELED" ]; then
    log "ERROR: Hangzhou relabeled data not found at $HZ_RELABELED"
    log "Run Jinan→Hangzhou experiment first."
    exit 1
fi

cat > "${BASE_OUTPUT}/experiment_meta.json" << EOF
{"timestamp":"$(date -Iseconds)","type":"ablation_e","topo_config":$TOPO_CONFIG}
EOF

# ---- E1: Topology Adapter Only (Jinan→Hangzhou, same as D but +topo) ----
run_adaptation_topo() {
    local ABL="$1" DATA="$2" MODE="$3" RANK="$4" ALPHA="$5" METRIC="$6" INIT_CKPT="$7"
    for K in ${LABEL_BUDGETS}; do
        local K_DIR="${BASE_OUTPUT}/${ABL}/adapted_k${K}"
        if [ -f "${K_DIR}/evaluation/eval_summary.json" ]; then continue; fi
        mkdir -p "${K_DIR}"
        log "${ABL} K=${K}"

        $PYTHON -c "
import json,random; random.seed(2024)
with open('${DATA}') as f: all_recs=[json.loads(l) for l in f if l.strip()]
tg={}
for r in all_recs: tg.setdefault(r.get('time_s',0),[]).append(r)
times=sorted(tg); sel=sorted(random.sample(times,min(${K},len(times))))
recs=[r for t in sel for r in tg[t]]
with open('${K_DIR}/train_data.jsonl','w') as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'K=${K}: {len(sel)} snaps -> {len(recs)} recs')
" >> "${BASE_OUTPUT}/pipeline.log" 2>&1

        local WS=$(echo "${TRAIN_GPUS}" | tr ',' '\n' | wc -l)
        local INIT_FLAG=""
        [ -n "$INIT_CKPT" ] && INIT_FLAG="--init_policy_ckpt ${INIT_CKPT}"

        CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --nproc_per_node=${WS} \
            projection_train/train_projection_bc.py \
            --data_path "${K_DIR}/train_data.jsonl" --llm_path "${LLM_PATH}" \
            --align_ckpt "${STAGE1_CKPT}" ${INIT_FLAG} \
            --output_dir "${K_DIR}/checkpoints" \
            --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
            --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
            --prompt_mode queue_wait_short --batch_size 2 --grad_accum_steps 8 \
            --epochs 5 --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine \
            --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy stratified \
            --num_workers 2 --seed 2024 --bf16 --log_interval 10 --save_every 1 \
            --best_metric "${METRIC}" --best_mode min --ddp_timeout_seconds 1800 \
            --target_mode "${MODE}" --soft_target_alpha "${ALPHA}" --soft_label_temperature 0.35 \
            --ranking_loss_weight "${RANK}" --cost_att_weight 1.0 --cost_aql_weight 1.0 \
            --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
            --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0.0 \
            --enable_topology_adapter --topology_roadnet_configs "${TOPO_CONFIG}" \
            >> "${BASE_OUTPUT}/pipeline.log" 2>&1

        local EVAL_DIR="${K_DIR}/evaluation"; mkdir -p "${EVAL_DIR}"
        for seed in ${SEEDS}; do
            [ -f "${EVAL_DIR}/result_seed${seed}.json" ] && continue
            $PYTHON projection_train/run_projection_online.py \
                --memo "stage3_e_${ABL}_k${K}" --projection_ckpt "${K_DIR}/checkpoints/best.pt" \
                --llm_path "${LLM_PATH}" --dataset hangzhou --traffic_file anon_4_4_hangzhou_real_5816.json \
                --run_counts 3600 --num_rounds 1 --seed "${seed}" \
                --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
                --periodicity 16 --input_dim 8 --hidden_dim 64 \
                --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
                --inference_batch_size 64 --bf16 --result_json "${EVAL_DIR}/result_seed${seed}.json" \
                >> "${BASE_OUTPUT}/pipeline.log" 2>&1
        done
        $PYTHON -c "
import json,os,numpy as np; from scipy import stats
mk={'ATT':'test_avg_travel_time_over','AQL':'test_avg_queue_len_over','AWT':'test_avg_waiting_time_over'}
av={m:[] for m in mk}
for s in [2024,42,123]:
    p=os.path.join('${EVAL_DIR}',f'result_seed{s}.json')
    if os.path.exists(p):
        with open(p) as f: rr=json.load(f).get('results',json.load(f))
        for lb,k in mk.items(): av[lb].append(float(rr.get(k,0)))
sm={}
for m,vs in av.items():
    a=np.array(vs,dtype=np.float64); n=len(a); mv=float(np.mean(a))
    sv=float(np.std(a,ddof=1)) if n>1 else 0.0
    ch=stats.t.ppf(0.975,n-1)*sv/np.sqrt(n) if n>1 else 0.0
    sm[m]={'mean':round(mv,2),'std':round(sv,2),'n':n,'ci95_low':round(mv-ch,2),'ci95_high':round(mv+ch,2)}
with open(os.path.join('${EVAL_DIR}','eval_summary.json'),'w') as f:
    json.dump({'ablation':'${ABL}','K':${K},'summary':sm,'raw_values':{m:[round(v,2) for v in vs] for m,vs in av.items()}},f,indent=2)
print('${ABL} K=${K} eval:',json.dumps(sm,indent=2))
" >> "${BASE_OUTPUT}/pipeline.log" 2>&1
    done
}

# ---- E1: Topology adapter, Jinan→Hangzhou, SimExpert+Pareto data ----
log "=== E1: Topology Adapter (Jinan→Hangzhou, Pareto data) ==="
run_adaptation_topo "E1" "$HZ_RELABELED" "mixed" "0.05" "0.3" "val_top1_cost_regret" "$SOURCE_CKPT"

# ---- E2: Joint Meta-Training ----
# Need Jinan SimExpert data too. Collect if not available.
JN_COLLECTED="${BASE_OUTPUT}/jinan_collected_states.jsonl"
JN_RELABELED="${BASE_OUTPUT}/jinan_relabeled_simexpert.jsonl"

log "=== E2 Prep: Collect Jinan states ==="
if [ ! -f "$JN_COLLECTED" ]; then
    $PYTHON projection_train/run_projection_online.py \
        --memo "stage3_e2_jn_collect" --projection_ckpt "${SOURCE_CKPT}" \
        --llm_path "${LLM_PATH}" --dataset jinan --traffic_file anon_3_4_jinan_real.json \
        --run_counts 3600 --num_rounds 1 --seed 2024 \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
        --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 \
        --collect_states_output "${JN_COLLECTED}" \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
fi
log "Jinan states: $(wc -l < ${JN_COLLECTED}) records"

if [ ! -f "$JN_RELABELED" ]; then
    log "SimExpert relabeling Jinan states..."
    $PYTHON scripts/relabel_simexpert.py \
        --input "${JN_COLLECTED}" --output "${JN_RELABELED}" \
        --roadnet data/Jinan/3_4/roadnet_3_4.json \
        --traffic_file data/Jinan/3_4/anon_3_4_jinan_real.json \
        --dataset jinan --work_dir "${BASE_OUTPUT}/simexpert_jn_work" \
        --sim_horizon ${SIM_HORIZON} --cost_att_weight 1.0 --cost_aql_weight 1.0 \
        --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
fi

# Create joint dataset
JOINT_DATA="${BASE_OUTPUT}/joint_train.jsonl"
log "Building joint Jinan+Hangzhou dataset..."
if [ ! -f "$JOINT_DATA" ]; then
    cat "$HZ_RELABELED" "$JN_RELABELED" > "$JOINT_DATA"
    log "Joint dataset: $(wc -l < ${JOINT_DATA}) records"
fi

# Phase M1: Joint pre-training
log "=== E2 Phase M1: Joint Pre-training ==="
JOINT_CKPT_DIR="${BASE_OUTPUT}/E2_joint/checkpoints"
mkdir -p "${JOINT_CKPT_DIR}"

if [ ! -f "${JOINT_CKPT_DIR}/best.pt" ]; then
    WS=$(echo "${TRAIN_GPUS}" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --nproc_per_node=${WS} \
        projection_train/train_projection_bc.py \
        --data_path "${JOINT_DATA}" --llm_path "${LLM_PATH}" \
        --align_ckpt "${STAGE1_CKPT}" --init_policy_ckpt "${SOURCE_CKPT}" \
        --output_dir "${JOINT_CKPT_DIR}" \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --prompt_mode queue_wait_short --batch_size 2 --grad_accum_steps 8 \
        --epochs 5 --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine \
        --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy stratified \
        --num_workers 2 --seed 2024 --bf16 --log_interval 10 --save_every 1 \
        --best_metric val_top1_cost_regret --best_mode min --ddp_timeout_seconds 1800 \
        --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 \
        --ranking_loss_weight 0.05 --cost_att_weight 1.0 --cost_aql_weight 1.0 \
        --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
        --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0.0 \
        --enable_topology_adapter --topology_roadnet_configs "${TOPO_CONFIG}" \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
fi
log "Joint pre-training complete"

# Phase M2: Few-shot adapt from joint checkpoint → Hangzhou
log "=== E2 Phase M2: Few-Shot Adapt (Joint → Hangzhou) ==="
JOINT_BEST="${JOINT_CKPT_DIR}/best.pt"
run_adaptation_topo "E2" "$HZ_RELABELED" "mixed" "0.05" "0.3" "val_top1_cost_regret" "$JOINT_BEST"

# ---- Summary ----
log "=== Ablation E Complete ==="
log "Results: ${BASE_OUTPUT}"
$PYTHON -c "
import json,os
base='${BASE_OUTPUT}'
for ab in ['E1','E2']:
    print(f'\n--- {ab} ---')
    for k in [8,32,120]:
        p=f'{base}/{ab}/adapted_k{k}/evaluation/eval_summary.json'
        if os.path.exists(p):
            with open(p) as f: d=json.load(f)
            s=d.get('summary',{})
            print(f'  K={k}: ATT={s.get(\"ATT\",{}).get(\"mean\",\"?\")} AQL={s.get(\"AQL\",{}).get(\"mean\",\"?\")} AWT={s.get(\"AWT\",{}).get(\"mean\",\"?\")}')
        else:
            print(f'  K={k}: PENDING')
" | tee -a "${BASE_OUTPUT}/pipeline.log"
