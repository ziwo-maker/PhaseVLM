#!/usr/bin/env bash
# Temporary full same-flow counterfactual matrix for the user-designated
# Hangzhou-5816 max_length=512 source policy.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO_ROOT/online_test_best_weights/anon_4_4_hangzhou_real_5816/hangzhou5816_maxlen512_epoch15_3seed_userdesignated.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$REPO_ROOT/records/pipeline_runs/hangzhou5816_ref_stage1_stage2_4gpu_ml512_20260905_102007/stage1/best.pt}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-4806f3d56f2c7409925c0d067151b04ce06bbd4e0fb6efaa1b93f10dc5be9397}"
RUN_ID="${RUN_ID:-tmp_hangzhou5816_full_counterfactual_matrix_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EPOCHS="${EPOCHS:-5}"
SNAPSHOT_RETRIES="${SNAPSHOT_RETRIES:-5}"
FLOW="anon_4_4_hangzhou_real_5816.json"
FLOW_PATH="$REPO_ROOT/data/Hangzhou/4_4/$FLOW"
ROADNET="$REPO_ROOT/data/Hangzhou/4_4/roadnet_4_4.json"

require_file() { [[ -s "$1" ]] || { printf 'Missing or empty file: %s\n' "$1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { printf 'Missing directory: %s\n' "$1" >&2; exit 1; }; }
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$RUN_ROOT/logs/launcher.log"; }
set_stage() { printf '%s\n' "$1" > "$RUN_ROOT/status/stage.txt"; log "stage=$1"; }

notify_feishu() {
  local message="$1"
  "$PYTHON_BIN" - "$message" <<'PY' || true
import json, sys, urllib.request
from pathlib import Path
try:
    cfg = json.loads(Path('/root/.claude/feishu.json').read_text())
    if cfg.get('mode') != 'push' or not cfg.get('webhook_url'):
        raise SystemExit(0)
    body = {'msg_type': 'text', 'content': {'text': sys.argv[1][:4000]}}
    req = urllib.request.Request(cfg['webhook_url'], data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=10) as response:
        response.read(4096)
except Exception:
    pass
PY
}

monitor_progress() {
  while [[ ! -f "$RUN_ROOT/status/completed" && ! -f "$RUN_ROOT/status/failed" ]]; do
    sleep 2400
    [[ -f "$RUN_ROOT/status/completed" || -f "$RUN_ROOT/status/failed" ]] && break
    local stage completed
    stage="$(cat "$RUN_ROOT/status/stage.txt" 2>/dev/null || printf unknown)"
    completed="$(find "$RUN_ROOT/online" -path '*/result.json' -type f -printf . 2>/dev/null | wc -c)"
    notify_feishu "Hangzhou5816 counterfactual matrix still running: stage=${stage}; completed_online_results=${completed}/60; run=${RUN_ID}."
  done
}

on_exit() {
  local code=$?
  if [[ "$code" -ne 0 ]]; then
    printf 'failed exit_code=%s\n' "$code" > "$RUN_ROOT/status/failed"
    notify_feishu "Hangzhou5816 counterfactual matrix failed: stage=$(cat "$RUN_ROOT/status/stage.txt" 2>/dev/null || printf unknown), exit=${code}; run=${RUN_ID}. Inspect $RUN_ROOT/logs."
  fi
  [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null || true
}
trap on_exit EXIT

validate_collection() {
  "$PYTHON_BIN" - "$1" "$RUN_COUNTS" <<'PY'
import json, sys
from collections import Counter
path, run_counts = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
by_time = Counter(float(r.get('time_s', -1)) for r in rows)
snapshots = {}
for r in rows:
    snapshots.setdefault(str(r.get('snapshot_path', '')), float(r.get('time_s', -1)))
expected = list(range(0, run_counts, 30))
missing_times = [t for t in expected if by_time[t] != 16]
bad = []
for snapshot, time_s in snapshots.items():
    try:
        with open(snapshot, encoding='utf-8') as handle:
            json.load(handle)
    except Exception as exc:
        bad.append((time_s, type(exc).__name__))
print(f'rows={len(rows)} snapshots={len(snapshots)} expected_snapshots={len(expected)} bad_snapshots={len(bad)}')
if missing_times:
    print('missing_or_incomplete_times=', missing_times[:20])
if bad:
    print('bad_snapshot_examples=', bad[:10])
if len(rows) != len(expected) * 16 or len(snapshots) != len(expected) or missing_times or bad:
    raise SystemExit('collection integrity gate failed')
PY
}

split_and_relabel() {
  local horizon="$1" label="$2"
  local split_root="$RUN_ROOT/relabel/h${horizon}"
  mkdir -p "$split_root"
  "$PYTHON_BIN" - "$COLLECTED" "$split_root" 4 <<'PY'
import json, os, sys
source, out_dir, shard_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(x) for x in open(source, encoding='utf-8') if x.strip()]
groups = {}
for row in rows:
    groups.setdefault(str(row['snapshot_path']), []).append(row)
snapshots = sorted(groups)
if len(snapshots) != 120:
    raise SystemExit(f'expected 120 validated snapshots, got {len(snapshots)}')
for index in range(shard_count):
    part = snapshots[index::shard_count]
    target = os.path.join(out_dir, f'input_shard{index}.jsonl')
    with open(target, 'w', encoding='utf-8') as handle:
        for snapshot in part:
            for row in groups[snapshot]:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'shard={index} snapshots={len(part)} rows={sum(len(groups[x]) for x in part)}')
PY
  local -a pids=()
  for shard in 0 1 2 3; do
    local gpu="${GPUS[$shard]}"
    local shard_input="$split_root/input_shard${shard}.jsonl"
    local shard_output="$split_root/label_shard${shard}.jsonl"
    timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
      --input "$shard_input" --output "$shard_output" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" \
      --dataset hangzhou --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" \
      --work_dir "$split_root/runtime_shard${shard}" --sim_horizon "$horizon" --min_action_time 30 --traffic_series_interval 2 \
      --cost_scope global --neighbor-mode responsive --inference_batch_size 32 --bf16 \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
      > "$RUN_ROOT/logs/relabel_h${horizon}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { log "H${horizon} relabel failure"; return 1; }
  "$PYTHON_BIN" - "$split_root" "$label" <<'PY'
import json, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
rows=[]; coverage=[]
for shard in range(4):
    rows.extend(json.loads(x) for x in (root / f'label_shard{shard}.jsonl').read_text(encoding='utf-8').splitlines() if x)
    coverage.append(json.loads((root / f'label_shard{shard}_coverage.json').read_text(encoding='utf-8')))
if len(rows) != 1920:
    raise SystemExit(f'expected 1920 labels, got {len(rows)}')
if any(int(x.get('skipped_snapshots', 0)) for x in coverage):
    raise SystemExit('relabel integrity gate failed: skipped snapshot')
output.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
summary = {key: sum(int(x.get(key, 0)) for x in coverage) for key in ('total_records','total_snapshots','selected_snapshots','processed_snapshots','skipped_snapshots','skipped_records','candidate_rollouts')}
(output.parent / f'{output.stem}_coverage.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, sort_keys=True))
PY
}

train_arm() {
  local arm="$1" label="$2" ranking="$3"
  local output="$RUN_ROOT/training/$arm"
  mkdir -p "$output"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 projection_train/train_projection_bc.py \
    --data_path "$label" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$output" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --batch_size 1 --grad_accum_steps 8 --epochs "$EPOCHS" --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --min_action_margin 0.02 --num_workers 4 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight "$ranking" \
    --cost_att_weight 1.0 --cost_aql_weight 1.0 --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
    --base_policy_kl_weight 0.02 --base_policy_kl_mode expert_match --best_metric val_top1_cost_regret --best_mode min \
    --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
    > "$RUN_ROOT/logs/train_${arm}.log" 2>&1
  require_file "$output/epoch_${EPOCHS}.pt"
}

run_online() {
  local arm="$1" epoch="$2" seed="$3" gpu="$4"
  local output="$RUN_ROOT/online/$arm/epoch_${epoch}/seed_${seed}"
  mkdir -p "$output"
  timeout --foreground 3600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/$arm/epoch_${epoch}/seed_${seed}" --projection_ckpt "$RUN_ROOT/training/$arm/epoch_${epoch}.pt" --llm_path "$MODEL_PATH" \
    --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 --result_json "$output/result.json" \
    > "$RUN_ROOT/logs/online_${arm}_epoch${epoch}_seed${seed}.log" 2>&1
}

aggregate() {
  "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]); metrics={'AWT':'test_avg_waiting_time_over','ATT':'test_avg_travel_time_over','AQL':'test_avg_queue_len_over','queued':'test_queuing_vehicle_num_over','p95_wait':'test_vehicle_wait_p95_over','max_wait':'test_vehicle_wait_max_over'}
groups={}
for path in root.glob('online/*/epoch_*/seed_*/result.json'):
    payload=json.loads(path.read_text(encoding='utf-8')); values=payload.get('results',payload); arm=path.parts[-4]; epoch=int(path.parts[-3].split('_')[1]); seed=int(path.parts[-2].split('_')[1])
    groups.setdefault((arm,epoch),[]).append({'seed':seed, **{k:float(values[v]) for k,v in metrics.items()}, 'path':str(path)})
rows=[]
for (arm,epoch), values in sorted(groups.items()): rows.append({'arm':arm,'epoch':epoch,'n':len(values),'per_seed':values, **{k:statistics.fmean(x[k] for x in values) for k in metrics}})
if len(rows) != 20 or any(x['n'] != 3 for x in rows): raise SystemExit('online completeness gate failed')
best=min(rows,key=lambda x:(x['AWT'],x['ATT'],x['AQL']))
(root/'online'/'aggregate.json').write_text(json.dumps({'by_checkpoint':rows,'best_by_awt':best},indent=2)+'\n',encoding='utf-8')
print(json.dumps(best,sort_keys=True))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$TORCHRUN"; require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$FLOW_PATH"; require_file "$ROADNET"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"; read -r -a SEEDS <<< "$SEEDS_RAW"
  (( ${#GPUS[@]} == 4 )) || { printf 'GPU_LIST must contain exactly four GPUs\n' >&2; exit 1; }
  [[ "$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA256" ]] || { printf 'source SHA256 mismatch\n' >&2; exit 1; }
  mkdir -p "$RUN_ROOT"/{logs,status,collect,relabel,training,online}
  export PYTHONPATH="$REPO_ROOT" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CITYFLOW_SNAPSHOT_DUMP_RETRIES="$SNAPSHOT_RETRIES"
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" <<PY
import json, sys
json.dump({'schema':'tmp_hangzhou5816_full_counterfactual_matrix_v1','flow':'$FLOW','scope':'same_flow','source_checkpoint':'$SOURCE_CKPT','source_sha256':'$EXPECTED_SOURCE_SHA256','gpus':'$GPU_LIST','seeds':'$SEEDS_RAW','run_counts':$RUN_COUNTS,'epochs':$EPOCHS,'snapshot_retries':$SNAPSHOT_RETRIES,'arms':{'responsive_h30_r005':{'horizon':30,'ranking_loss_weight':0.05},'responsive_h120_r000':{'horizon':120,'ranking_loss_weight':0.0},'responsive_h120_r005':{'horizon':120,'ranking_loss_weight':0.05},'responsive_h120_r015':{'horizon':120,'ranking_loss_weight':0.15}},'training_cost_weights':{'att':1.0,'aql':1.0,'awt':1.0,'spillback':0.2},'min_action_margin':0.02},open(sys.argv[1],'w'),indent=2)
PY
  monitor_progress & MONITOR_PID=$!
  notify_feishu "Hangzhou5816 full counterfactual matrix started: H30 r=.05; H120 r=0/.05/.15; 4-GPU DDP; run=${RUN_ID}."
  set_stage collect
  COLLECTED="$RUN_ROOT/collect/decision_states.jsonl"
  timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/collect" --projection_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed 2024 \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --collect_states_output "$COLLECTED" --result_json "$RUN_ROOT/collect/result.json" \
    > "$RUN_ROOT/logs/collect.log" 2>&1
  validate_collection "$COLLECTED" | tee "$RUN_ROOT/status/collection_validation.txt"
  set_stage relabel_h30
  H30_LABEL="$RUN_ROOT/relabel/responsive_h30.jsonl"; split_and_relabel 30 "$H30_LABEL"
  set_stage relabel_h120
  H120_LABEL="$RUN_ROOT/relabel/responsive_h120.jsonl"; split_and_relabel 120 "$H120_LABEL"
  set_stage train_h30_r005
  train_arm responsive_h30_r005 "$H30_LABEL" 0.05
  set_stage train_h120_r000
  train_arm responsive_h120_r000 "$H120_LABEL" 0.0
  set_stage train_h120_r005
  train_arm responsive_h120_r005 "$H120_LABEL" 0.05
  set_stage train_h120_r015
  train_arm responsive_h120_r015 "$H120_LABEL" 0.15
  set_stage online_evaluation
  local -a jobs=() pids=()
  local arm epoch seed i j
  for arm in responsive_h30_r005 responsive_h120_r000 responsive_h120_r005 responsive_h120_r015; do for ((epoch=1; epoch<=EPOCHS; epoch++)); do require_file "$RUN_ROOT/training/$arm/epoch_${epoch}.pt"; for seed in "${SEEDS[@]}"; do jobs+=("$arm:$epoch:$seed"); done; done; done
  for ((i=0; i<${#jobs[@]}; i+=4)); do
    pids=()
    for ((j=0; j<4 && i+j<${#jobs[@]}; j++)); do IFS=: read -r arm epoch seed <<< "${jobs[$((i+j))]}"; run_online "$arm" "$epoch" "$seed" "${GPUS[$j]}" & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
  set_stage aggregate
  aggregate | tee "$RUN_ROOT/status/best_by_awt.json"
  touch "$RUN_ROOT/status/completed"
  notify_feishu "Hangzhou5816 full counterfactual matrix completed: $(cat "$RUN_ROOT/status/best_by_awt.json"); run=${RUN_ID}."
  log "completed: $RUN_ROOT"
}

main "$@"
