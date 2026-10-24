#!/usr/bin/env bash
# Same-flow multi-parameter policy-responsive counterfactual matrix for Hangzhou-real epoch 6.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="$REPO_ROOT/records/pipeline_runs/hangzhou_real_simexpert_h30_att1_aql1_awt1_spill0p2_stage12_ml512_ga16_20260908_004432"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_6.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$SOURCE_RUN/stage1/best.pt}"
COLLECTED="${COLLECTED:-$REPO_ROOT/records/counterfactual_validation/hangzhou_real_epoch6_counterfactual_h30_4gpu_20260908_102500/collect/decision_states.jsonl}"
H30_LABEL="${H30_LABEL:-$REPO_ROOT/records/counterfactual_validation/hangzhou_real_epoch6_counterfactual_h30_4gpu_20260908_102500/relabel/policy_responsive_h30.jsonl}"
RUN_ID="${RUN_ID:-hangzhou_real_epoch6_counterfactual_matrix_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
EPOCHS="${EPOCHS:-5}"
FLOW="anon_4_4_hangzhou_real.json"
FLOW_PATH="$REPO_ROOT/data/Hangzhou/4_4/$FLOW"
ROADNET="$REPO_ROOT/data/Hangzhou/4_4/roadnet_4_4.json"

require_file() { [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 1; }; }
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$RUN_ROOT/launcher.log"; }
stage() { printf '%s\n' "$1" > "$RUN_ROOT/status/stage.txt"; log "stage=$1"; }

notify() {
  "$PYTHON_BIN" - "$1" <<'PY' || true
import json, sys, urllib.request
from pathlib import Path
try:
    cfg = json.loads(Path('/root/.claude/feishu.json').read_text(encoding='utf-8'))
    if cfg.get('mode') != 'push' or not cfg.get('webhook_url'): raise SystemExit(0)
    body = json.dumps({'msg_type':'text','content':{'text':sys.argv[1][:3000]}}, ensure_ascii=False).encode()
    urllib.request.urlopen(urllib.request.Request(cfg['webhook_url'], data=body, headers={'Content-Type':'application/json'}), timeout=10).read(1024)
except Exception: pass
PY
}

monitor() {
  while [[ ! -f "$RUN_ROOT/status/completed" && ! -f "$RUN_ROOT/status/failed" ]]; do
    sleep 2400
    [[ -f "$RUN_ROOT/status/completed" || -f "$RUN_ROOT/status/failed" ]] && break
    notify "Hangzhou-real epoch6 counterfactual matrix: stage=$(cat "$RUN_ROOT/status/stage.txt" 2>/dev/null || echo unknown), online_results=$(rg --files "$RUN_ROOT/online" -g result.json 2>/dev/null | wc -l)/60, run=$RUN_ID"
  done
}

on_exit() { local rc=$?; (( rc == 0 )) || { printf 'exit_code=%s\n' "$rc" > "$RUN_ROOT/status/failed"; notify "Hangzhou-real epoch6 counterfactual matrix failed at $(cat "$RUN_ROOT/status/stage.txt" 2>/dev/null || echo unknown); run=$RUN_ID"; }; [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null || true; }
trap on_exit EXIT

split_relabel() {
  local arm="$1" horizon="$2" label="$3"
  local split="$RUN_ROOT/relabel/$arm"
  mkdir -p "$split"
  "$PYTHON_BIN" - "$COLLECTED" "$split" <<'PY'
import json, sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
groups={}
for row in rows: groups.setdefault(row['snapshot_path'],[]).append(row)
snapshots=sorted(groups)
if len(rows)!=1920 or len(snapshots)!=120: raise SystemExit(f'bad collection rows={len(rows)} snapshots={len(snapshots)}')
root=Path(sys.argv[2])
for shard in range(4):
    with (root/f'input_{shard}.jsonl').open('w',encoding='utf-8') as out:
        for snapshot in snapshots[shard::4]:
            for row in groups[snapshot]: out.write(json.dumps(row,ensure_ascii=False)+'\n')
PY
  local pids=() shard gpu rc=0
  for shard in 0 1 2 3; do gpu="${GPUS[$shard]}"; timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py --input "$split/input_${shard}.jsonl" --output "$split/label_${shard}.jsonl" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" --dataset hangzhou --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --work_dir "$split/runtime_${shard}" --sim_horizon "$horizon" --min_action_time 30 --traffic_series_interval 2 --cost_scope global --neighbor-mode responsive --inference_batch_size 32 --bf16 --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --cost_att_weight 1 --cost_aql_weight 1 --cost_awt_weight 1 --cost_spillback_weight 0.2 > "$RUN_ROOT/logs/relabel_${arm}_shard${shard}.log" 2>&1 & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done; (( rc == 0 )) || return "$rc"
  "$PYTHON_BIN" - "$split" "$label" <<'PY'
import json, sys
from pathlib import Path
root,out=Path(sys.argv[1]),Path(sys.argv[2]); rows=[]
for shard in range(4): rows += [json.loads(x) for x in (root/f'label_{shard}.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
if len(rows)!=1920: raise SystemExit(f'expected 1920 labels, got {len(rows)}')
rows.sort(key=lambda x:(x['snapshot_path'],x['intersection_id']))
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
PY
}

reweight_h120() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
from projection_train.action_space import ACTION_LIST
from projection_train.cost_ranking import _normalized_metric_costs
source,out=Path(sys.argv[1]),Path(sys.argv[2]); rows=[]; weights={'att':1.0,'aql':1.0,'awt':2.0,'spillback':0.2}
for line in source.read_text(encoding='utf-8').splitlines():
    row=json.loads(line); costs=_normalized_metric_costs(row['candidate_action_metrics'],weights); feasible=row.get('pareto_feasible',{})
    choices=[a for a in ACTION_LIST if a in costs and feasible.get(a,True)]
    base=row.get('base_action'); best=min(choices,key=lambda a:(costs[a],0 if a==base else 1)); values=sorted(costs[a] for a in choices)
    row['candidate_action_waits']=costs; row['expert_action']=best; row['expert_action_margin']=values[1]-values[0] if len(values)>1 else 0.0
    base_cost=costs.get(base,0.0); row['candidate_advantages']={a:base_cost-v for a,v in costs.items()}; row['cost_weights']=weights
    for action, value in row['candidate_action_metrics'].items():
        value['score']=costs.get(action); value['weight_att']=1.0; value['weight_aql']=1.0; value['weight_awt']=2.0; value['weight_spillback']=0.2
    rows.append(row)
if len(rows)!=1920: raise SystemExit(f'expected 1920 labels, got {len(rows)}')
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
PY
}

train_arm() {
  local arm="$1" label="$2" out="$RUN_ROOT/training/$arm"; mkdir -p "$out"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 projection_train/train_projection_bc.py --data_path "$label" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$out" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --batch_size 1 --grad_accum_steps 8 --epochs "$EPOCHS" --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight 0.5 --cost_att_weight 1 --cost_aql_weight 1 --cost_awt_weight 2 --cost_spillback_weight 0.2 --best_metric val_top1_cost_regret --best_mode min --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 > "$RUN_ROOT/logs/train_${arm}.log" 2>&1
}

online() {
  local arm="$1" epoch="$2" seed="$3" gpu="$4" out="$RUN_ROOT/online/$arm/epoch_$epoch/seed_$seed"; mkdir -p "$out"
  timeout --foreground 3600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py --memo "$RUN_ID/$arm/epoch_$epoch/seed_$seed" --projection_ckpt "$RUN_ROOT/training/$arm/epoch_$epoch.pt" --llm_path "$MODEL_PATH" --dataset hangzhou --traffic_file "$FLOW" --run_counts 3600 --num_rounds 1 --seed "$seed" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 --result_json "$out/result.json" > "$RUN_ROOT/logs/online_${arm}_e${epoch}_s${seed}.log" 2>&1
}

main() {
  cd "$REPO_ROOT"; require_file "$PYTHON_BIN"; require_file "$TORCHRUN"; require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$COLLECTED"; require_file "$H30_LABEL"; require_file "$FLOW_PATH"; require_file "$ROADNET"; [[ -d "$MODEL_PATH" ]] || exit 1
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"; read -r -a SEEDS <<< "$SEEDS_RAW"; (( ${#GPUS[@]} == 4 )) || exit 1
  mkdir -p "$RUN_ROOT"/{logs,status,relabel,training,online}; export PYTHONPATH="$REPO_ROOT" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  SOURCE_SHA256="$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')"
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" <<PY
import json,sys
json.dump({'schema':'hangzhou_real_epoch6_counterfactual_matrix_v1','source_checkpoint':'$SOURCE_CKPT','source_sha256':'$SOURCE_SHA256','flow':'$FLOW','scope':'same_flow','collection':'$COLLECTED','gpus':'$GPU_LIST','epochs':$EPOCHS,'seeds':'$SEEDS_RAW','arms':{'h30_balanced':{'horizon':30,'weights':[1,1,1,.2]},'h60_balanced':{'horizon':60,'weights':[1,1,1,.2]},'h120_balanced':{'horizon':120,'weights':[1,1,1,.2]},'h120_awt2':{'horizon':120,'weights':[1,1,2,.2],'label_source':'reweighted h120 candidate metrics'}},'training':{'ranking_loss_weight':.5,'loss_weights':[1,1,2,.2]}},open(sys.argv[1],'w'),indent=2)
PY
  monitor & MONITOR_PID=$!; notify "Hangzhou-real epoch6 multi-parameter counterfactual matrix started: H30/H60/H120 balanced plus H120 AWTx2; four-GPU; run=$RUN_ID"
  local h60="$RUN_ROOT/relabel/h60_balanced.jsonl" h120="$RUN_ROOT/relabel/h120_balanced.jsonl" h120w="$RUN_ROOT/relabel/h120_awt2.jsonl"
  stage relabel_h60; split_relabel h60_balanced 60 "$h60"
  stage relabel_h120; split_relabel h120_balanced 120 "$h120"
  stage reweight_h120_awt2; reweight_h120 "$h120" "$h120w"
  local arm label
  for arm in h30_balanced h60_balanced h120_balanced h120_awt2; do case "$arm" in h30_balanced) label="$H30_LABEL";; h60_balanced) label="$h60";; h120_balanced) label="$h120";; *) label="$h120w";; esac; stage "train_$arm"; train_arm "$arm" "$label"; require_file "$RUN_ROOT/training/$arm/epoch_$EPOCHS.pt"; done
  stage online; local jobs=() arm epoch seed i j pids=()
  for arm in h30_balanced h60_balanced h120_balanced h120_awt2; do for ((epoch=1;epoch<=EPOCHS;epoch++)); do for seed in "${SEEDS[@]}"; do jobs+=("$arm:$epoch:$seed"); done; done; done
  for ((i=0;i<${#jobs[@]};i+=4)); do pids=(); for ((j=0;j<4 && i+j<${#jobs[@]};j++)); do IFS=: read -r arm epoch seed <<< "${jobs[$((i+j))]}"; online "$arm" "$epoch" "$seed" "${GPUS[$j]}" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done; done
  stage aggregate
  "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]); metrics={'AWT':'test_avg_waiting_time_over','ATT':'test_avg_travel_time_over','AQL':'test_avg_queue_len_over','queued':'test_queuing_vehicle_num_over','reward':'test_reward_over','p95_wait':'test_vehicle_wait_p95_over','max_wait':'test_vehicle_wait_max_over'}; groups={}
for path in root.glob('online/*/epoch_*/seed_*/result.json'):
    d=json.loads(path.read_text(encoding='utf-8')).get('results',{}); arm=path.parts[-4]; epoch=int(path.parts[-3].split('_')[1]); groups.setdefault((arm,epoch),[]).append({k:float(d[v]) for k,v in metrics.items()})
rows=[{'arm':a,'epoch':e,'n':len(x),**{k:statistics.fmean(y[k] for y in x) for k in metrics}} for (a,e),x in sorted(groups.items())]
if len(rows)!=20 or any(x['n']!=3 for x in rows): raise SystemExit(f'incomplete online matrix: groups={len(rows)}')
best=min(rows,key=lambda x:(x['AWT'],x['ATT'],x['AQL'])); (root/'online'/'aggregate.json').write_text(json.dumps({'by_checkpoint':rows,'best_adapted_by_awt':best},indent=2)+'\n',encoding='utf-8'); print(json.dumps(best))
PY
  touch "$RUN_ROOT/status/completed"; notify "Hangzhou-real epoch6 multi-parameter counterfactual matrix completed; run=$RUN_ID"; log "completed=$RUN_ROOT"
}
main "$@"
