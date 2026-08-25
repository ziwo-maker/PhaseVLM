#!/bin/bash
# Stage 3 Reverse: Hangzhou → Jinan Cross-Network Adaptation
# Runs ablations A-D with source=hangzhou, target=jinan
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source /home/sysu/anaconda3/etc/profile.d/conda.sh
conda activate hnr5090
export PYTHON="$(which python3)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUTPUT="records/pipeline_runs/stage3_reverse_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT}"

# Hangzhou source checkpoint
SOURCE_CKPT="records/pipeline_runs/exp_hangzhou_real/stage2_queue_wait_rule_gap2_p60_ml412_qw_20260530_115914/checkpoints/best.pt"
STAGE1_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"  # reuse Stage 1 align
LLM_PATH="/home/data/model/qwen-vl-8B"
TRAIN_GPUS="0,1"
COLLECTION_FLOW="anon_3_4_jinan_real.json"
TEST_FLOW="anon_3_4_jinan_real_2000.json"
LABEL_BUDGETS="8 32 120"
SEEDS="2024 42 123"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cat > "${BASE_OUTPUT}/experiment_meta.json" << EOF
{"timestamp":"$(date -Iseconds)","direction":"hangzhou_to_jinan","source_ckpt":"$SOURCE_CKPT","stage1_ckpt":"$STAGE1_CKPT"}
EOF

# ---- Phase 2: Zero-shot (Hangzhou ckpt → Jinan) ----
log "Phase 2: Zero-Shot Hangzhou→Jinan"
ZERO_SHOT_DIR="${BASE_OUTPUT}/zero_shot"
mkdir -p "${ZERO_SHOT_DIR}"
for seed in ${SEEDS}; do
    RESULT_JSON="${ZERO_SHOT_DIR}/result_seed${seed}.json"
    if [ -f "$RESULT_JSON" ]; then continue; fi
    log "Zero-shot seed=${seed}"
    $PYTHON projection_train/run_projection_online.py \
        --memo "stage3_reverse_zs" --projection_ckpt "${SOURCE_CKPT}" \
        --llm_path "${LLM_PATH}" --dataset jinan --traffic_file "${TEST_FLOW}" \
        --run_counts 3600 --num_rounds 1 --seed "${seed}" \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
        --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 --result_json "${RESULT_JSON}" \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
done
log "Phase 2 complete"

# ---- Phase 3: Collect Jinan states ----
log "Phase 3: Collect Jinan States"
COLLECTED="${BASE_OUTPUT}/collected_states.jsonl"
if [ ! -f "$COLLECTED" ]; then
    $PYTHON projection_train/run_projection_online.py \
        --memo "stage3_reverse_collect" --projection_ckpt "${SOURCE_CKPT}" \
        --llm_path "${LLM_PATH}" --dataset jinan --traffic_file "${COLLECTION_FLOW}" \
        --run_counts 3600 --num_rounds 1 --seed 2024 \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
        --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 \
        --collect_states_output "${COLLECTED}" \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
fi
log "Phase 3: $(wc -l < ${COLLECTED}) records collected"

# ---- Phase 4a: RuleOnly relabeling (A) ----
log "Phase 4a: RuleOnly"
RELABELED_A="${BASE_OUTPUT}/A/relabeled_states.jsonl"
if [ ! -f "$RELABELED_A" ]; then
    mkdir -p "${BASE_OUTPUT}/A"
    $PYTHON -c "
import json; from projection_train.action_space import ACTION_LIST
with open('${COLLECTED}') as fin, open('${RELABELED_A}','w') as fout:
    for l in fin:
        if not l.strip(): continue
        r=json.loads(l); pq=r.get('phase_queue',{}); pw=r.get('phase_wait',{})
        best,best_s=ACTION_LIST[0],-1.0
        for a in ACTION_LIST:
            s=float(pq.get(a,0)or 0)+2.0*float(pw.get(a,0)or 0)
            if s>best_s: best_s=s; best=a
        r['expert_action']=best; r['relabel_method']='ruleonly_awt2.0'
        r['candidate_action_metrics']={}; r['candidate_action_waits']={}
        r['expert_action_margin']=0.0; r['base_action']=r.get('policy_action',ACTION_LIST[0])
        fout.write(json.dumps(r,ensure_ascii=False)+'\n')
print('RuleOnly done')
" >> "${BASE_OUTPUT}/pipeline.log" 2>&1
fi

# ---- Phase 4b: SimExpert relabeling (B/C/D) ----
log "Phase 4b: SimExpert"
RELABELED_SE="${BASE_OUTPUT}/relabeled_simexpert.jsonl"
if [ ! -f "$RELABELED_SE" ]; then
    $PYTHON scripts/relabel_simexpert.py \
        --input "${COLLECTED}" --output "${RELABELED_SE}" \
        --roadnet data/Jinan/3_4/roadnet_3_4.json \
        --traffic_file data/Jinan/3_4/anon_3_4_jinan_real.json \
        --dataset jinan --work_dir "${BASE_OUTPUT}/simexpert_work" \
        --sim_horizon 30 --cost_att_weight 1.0 --cost_aql_weight 1.0 \
        --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
        >> "${BASE_OUTPUT}/pipeline.log" 2>&1
fi

# ---- Phase 5+6: Adapt + Eval (A-D) ----
run_adaptation() {
    local ABL="$1" DATA="$2" MODE="$3" RANK="$4" ALPHA="$5" METRIC="$6"
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
        CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --nproc_per_node=${WS} \
            projection_train/train_projection_bc.py \
            --data_path "${K_DIR}/train_data.jsonl" --llm_path "${LLM_PATH}" \
            --align_ckpt "${STAGE1_CKPT}" --init_policy_ckpt "${SOURCE_CKPT}" \
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
            >> "${BASE_OUTPUT}/pipeline.log" 2>&1

        local EVAL_DIR="${K_DIR}/evaluation"; mkdir -p "${EVAL_DIR}"
        for seed in ${SEEDS}; do
            [ -f "${EVAL_DIR}/result_seed${seed}.json" ] && continue
            $PYTHON projection_train/run_projection_online.py \
                --memo "stage3_rev_${ABL}_k${K}" --projection_ckpt "${K_DIR}/checkpoints/best.pt" \
                --llm_path "${LLM_PATH}" --dataset jinan --traffic_file "${TEST_FLOW}" \
                --run_counts 3600 --num_rounds 1 --seed "${seed}" \
                --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
                --periodicity 16 --input_dim 8 --hidden_dim 64 \
                --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
                --inference_batch_size 64 --bf16 --result_json "${EVAL_DIR}/result_seed${seed}.json" \
                >> "${BASE_OUTPUT}/pipeline.log" 2>&1
        done
        $PYTHON -c "
import json,os,numpy as np; from scipy import stats
r={}; mk={'ATT':'test_avg_travel_time_over','AQL':'test_avg_queue_len_over','AWT':'test_avg_waiting_time_over'}
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
print('${ABL} K=${K} eval done:',json.dumps(sm,indent=2))
" >> "${BASE_OUTPUT}/pipeline.log" 2>&1
    done
}

log "=== Starting Ablation Matrix (Reverse) ==="
run_adaptation "A" "${RELABELED_A}" "hard" "0.0" "0.0" "val_acc"
run_adaptation "B" "${RELABELED_SE}" "hard" "0.0" "0.0" "val_acc"
run_adaptation "C" "${RELABELED_SE}" "soft" "0.0" "0.3" "val_top1_cost_regret"
run_adaptation "D" "${RELABELED_SE}" "mixed" "0.05" "0.3" "val_top1_cost_regret"

log "=== Reverse Experiment Complete ==="
log "Results: ${BASE_OUTPUT}"
