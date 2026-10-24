#!/usr/bin/env bash
# Resume the failed Jinan-real epoch-12 counterfactual matrix with four GPUs.
# H30/H60 labels are integrity-checked and reused; H120 is recomputed in four
# independent snapshot shards. Long-running relabel/training commands have no
# arbitrary wall-clock timeout.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan_real_simexpert_h60_att1_aql1_awt1_spill0_stage12_ml512_ga16_20260908_001305}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_12.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$SOURCE_RUN/stage1/best.pt}"
BASE_RUN="${BASE_RUN:-$REPO_ROOT/records/counterfactual_validation/jinan_real_epoch12_counterfactual_matrix_ml512_20260908_231300}"
RUN_ID="${RUN_ID:-jinan_real_epoch12_counterfactual_matrix_retry4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-2024}"

FLOW="anon_3_4_jinan_real.json"
FLOW_PATH="$REPO_ROOT/data/Jinan/3_4/$FLOW"
ROADNET="$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json"
GRID_ROWS=3
GRID_COLS=4
SNAPSHOT_COUNT=120
INTERSECTION_COUNT=12
EXPECTED_ROWS=$((SNAPSHOT_COUNT * INTERSECTION_COUNT))
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
RELABEL_DIR="$RUN_ROOT/relabel"
TRAIN_DIR="$RUN_ROOT/training"
COLLECTED="$BASE_RUN/collect/decision_states.jsonl"
H30_LABEL="$BASE_RUN/relabel/h30_balanced.jsonl"
H60_LABEL="$BASE_RUN/relabel/h60_balanced.jsonl"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/launcher.log"; }
require_file() { [[ -s "$1" ]] || { log "missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "missing directory: $1" >&2; exit 1; }; }
set_stage() { printf '%s\n' "$1" > "$STATUS_DIR/stage.txt"; log "stage=$1"; }

notify_feishu() {
  "$PYTHON_BIN" - "$1" <<'PY' || true
import json
import sys
import urllib.request
from pathlib import Path
try:
    config = json.loads((Path.home() / '.claude' / 'feishu.json').read_text(encoding='utf-8'))
    mode = str(config.get('mode', config.get('notify_mode', ''))).lower()
    webhook = str(config.get('webhook_url', config.get('webhook', '')))
    if mode not in {'push', 'on', 'true', '1'} or not webhook:
        raise SystemExit(0)
    body = json.dumps({'msg_type': 'text', 'content': {'text': sys.argv[1][:3500]}}, ensure_ascii=False).encode()
    request = urllib.request.Request(webhook, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read(1024)
except Exception:
    pass
PY
}

monitor_progress() {
  while [[ ! -f "$STATUS_DIR/completed" && ! -f "$STATUS_DIR/failed" ]]; do
    sleep 2400
    [[ -f "$STATUS_DIR/completed" || -f "$STATUS_DIR/failed" ]] && break
    local current_stage completed_shards trained
    current_stage="$(<"$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown)"
    completed_shards="$(rg --files "$RELABEL_DIR/h120_shards" -g 'label_*.jsonl' 2>/dev/null | wc -l)"
    trained="$(rg --files "$TRAIN_DIR" -g best.pt 2>/dev/null | wc -l)"
    notify_feishu "Jinan-real epoch12 four-GPU recovery: stage=${current_stage}; h120_shards=${completed_shards}/4; trained_arms=${trained}/4; run=${RUN_ID}."
  done
}

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'exit_code=%s at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS_DIR/failed"
    notify_feishu "Jinan-real epoch12 four-GPU recovery failed: stage=$(<"$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown); exit=${rc}; run=${RUN_ID}."
  fi
  [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap on_exit EXIT

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$SOURCE_CKPT" "$ALIGN_CKPT" "$BASE_RUN" "$GPU_LIST" "$EPOCHS" "$SEED" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
out, source, align, base_run, gpus, epochs, seed = sys.argv[1:]
digest = hashlib.sha256()
with open(source, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
payload = {
    'schema': 'jinan_real_epoch12_counterfactual_matrix_retry4gpu_v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'flow': 'anon_3_4_jinan_real.json',
    'scope': 'same_flow',
    'source_checkpoint': source,
    'source_checkpoint_sha256': digest.hexdigest(),
    'alignment_checkpoint': align,
    'reused_artifacts_from_failed_run': base_run,
    'reused_labels': {'h30_balanced': f'{base_run}/relabel/h30_balanced.jsonl', 'h60_balanced': f'{base_run}/relabel/h60_balanced.jsonl'},
    'gpus': [int(item) for item in gpus.split(',')],
    'architecture': {'seq_len': 16, 'image_size': 224, 'periodicity': 16, 'input_dim': 8, 'hidden_dim': 256, 'prefix_tokens': 32, 'prefix_mode': 'prefix', 'max_length': 512, 'prompt_mode': 'queue_wait_short'},
    'relabel_protocol': {'neighbor_mode': 'responsive', 'cost_scope': 'global', 'h120_seconds': 120, 'h120_shards': 4, 'snapshots_per_shard': 30, 'per_stage_wall_clock_timeout': None},
    'arms': {
        'h30_balanced': {'horizon_seconds': 30, 'weights': [1, 1, 1, 0], 'labels': 'reused'},
        'h60_balanced': {'horizon_seconds': 60, 'weights': [1, 1, 1, 0], 'labels': 'reused'},
        'h120_balanced': {'horizon_seconds': 120, 'weights': [1, 1, 1, 0], 'labels': 'four_gpu_recomputed'},
        'h120_awt2': {'horizon_seconds': 120, 'weights': [1, 1, 2, 0], 'labels': 'reweighted_h120_candidate_metrics'},
    },
    'training': {'epochs': int(epochs), 'batch_size': 1, 'grad_accum_steps': 16, 'ranking_loss_weight': 0.5, 'per_stage_wall_clock_timeout': None, 'online_evaluation': False, 'seed': int(seed)},
}
Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
}

validate_existing_label() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
path, arm = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
coverage = json.loads(path.with_name(path.stem + '_coverage.json').read_text(encoding='utf-8'))
keys = {(row.get('snapshot_path'), row.get('intersection_id')) for row in rows}
if len(rows) != 1440 or len(keys) != 1440 or coverage.get('processed_snapshots') != 120 or coverage.get('skipped_snapshots') != 0:
    raise SystemExit(f'{arm} integrity failed: rows={len(rows)}, keys={len(keys)}, coverage={coverage}')
print(f'{arm}: reused 1440 complete labels')
PY
}

split_h120_inputs() {
  "$PYTHON_BIN" - "$COLLECTED" "$RELABEL_DIR/h120_shards" <<'PY'
import json
import sys
from pathlib import Path
source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
groups = {}
for row in rows:
    groups.setdefault(str(row['snapshot_path']), []).append(row)
snapshots = sorted(groups)
if len(rows) != 1440 or len(snapshots) != 120 or any(len(groups[item]) != 12 for item in snapshots):
    raise SystemExit(f'bad collection: rows={len(rows)} snapshots={len(snapshots)}')
out_dir.mkdir(parents=True, exist_ok=True)
for shard in range(4):
    chosen = snapshots[shard::4]
    if len(chosen) != 30:
        raise SystemExit(f'shard {shard} snapshots={len(chosen)}')
    with (out_dir / f'input_{shard}.jsonl').open('w', encoding='utf-8') as handle:
        for snapshot in chosen:
            for row in groups[snapshot]:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'shard={shard} snapshots={len(chosen)} rows={sum(len(groups[item]) for item in chosen)}')
PY
}

relabel_h120_shard() {
  local shard="$1" gpu="$2" root="$RELABEL_DIR/h120_shards"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
    --input "$root/input_${shard}.jsonl" --output "$root/label_${shard}.jsonl" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" \
    --dataset jinan --num_rows "$GRID_ROWS" --num_cols "$GRID_COLS" --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" \
    --work_dir "$root/runtime_${shard}" --sim_horizon 120 --min_action_time 30 --traffic_series_interval 2 \
    --cost_scope global --neighbor-mode responsive --inference_batch_size 32 --bf16 \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --cost_att_weight 1 --cost_aql_weight 1 --cost_awt_weight 1 --cost_spillback_weight 0 \
    > "$LOG_DIR/relabel_h120_shard${shard}.log" 2>&1
}

merge_h120() {
  "$PYTHON_BIN" - "$RELABEL_DIR/h120_shards" "$RELABEL_DIR/h120_balanced.jsonl" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
coverage = Counter()
for shard in range(4):
    shard_rows = [json.loads(line) for line in (root / f'label_{shard}.jsonl').read_text(encoding='utf-8').splitlines() if line]
    shard_coverage = json.loads((root / f'label_{shard}_coverage.json').read_text(encoding='utf-8'))
    if len(shard_rows) != 360 or shard_coverage.get('processed_snapshots') != 30 or shard_coverage.get('skipped_snapshots') != 0:
        raise SystemExit(f'shard {shard} incomplete: rows={len(shard_rows)}, coverage={shard_coverage}')
    rows.extend(shard_rows)
    coverage.update(shard_coverage)
keys = {(str(row.get('snapshot_path')), str(row.get('intersection_id'))) for row in rows}
if len(rows) != 1440 or len(keys) != 1440:
    raise SystemExit(f'merged h120 incomplete: rows={len(rows)} unique={len(keys)}')
rows.sort(key=lambda row: (float(row.get('time_s', 0)), str(row.get('intersection_id', ''))))
output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
summary = {key: int(coverage[key]) for key in ('total_records', 'total_snapshots', 'selected_snapshots', 'processed_snapshots', 'skipped_records', 'skipped_snapshots', 'candidate_rollouts')}
summary.update({'sim_horizon': 120, 'neighbor_mode': 'responsive', 'cost_scope': 'global', 'shards': 4})
output.with_name(output.stem + '_coverage.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, sort_keys=True))
PY
}

reweight_h120_awt2() {
  "$PYTHON_BIN" - "$RELABEL_DIR/h120_balanced.jsonl" "$RELABEL_DIR/h120_awt2.jsonl" <<'PY'
import json
import sys
from pathlib import Path
from projection_train.action_space import ACTION_LIST
from projection_train.cost_ranking import _normalized_metric_costs
source, output = Path(sys.argv[1]), Path(sys.argv[2])
weights = {'att': 1.0, 'aql': 1.0, 'awt': 2.0, 'spillback': 0.0}
rows = []
for line in source.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    costs = _normalized_metric_costs(row['candidate_action_metrics'], weights)
    feasible = row.get('pareto_feasible', {})
    choices = [action for action in ACTION_LIST if action in costs and feasible.get(action, True)] or [action for action in ACTION_LIST if action in costs]
    base = row.get('base_action')
    row['expert_action'] = min(choices, key=lambda action: (costs[action], 0 if action == base else 1))
    values = sorted(costs[action] for action in choices)
    row['expert_action_margin'] = values[1] - values[0] if len(values) > 1 else 0.0
    row['candidate_action_waits'] = costs
    row['candidate_advantages'] = {action: costs.get(base, 0.0) - value for action, value in costs.items()}
    row['cost_weights'] = weights
    for action, metrics in row['candidate_action_metrics'].items():
        metrics.update({'score': costs.get(action), 'weight_att': 1.0, 'weight_aql': 1.0, 'weight_awt': 2.0, 'weight_spillback': 0.0})
    rows.append(row)
if len(rows) != 1440:
    raise SystemExit(f'expected 1440 h120_awt2 rows, got {len(rows)}')
output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
PY
}

validate_label() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
path, arm = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
keys = {(str(row.get('snapshot_path')), str(row.get('intersection_id'))) for row in rows}
actions = Counter(str(row.get('expert_action', '')) for row in rows)
if len(rows) != 1440 or len(keys) != 1440 or '' in actions or not actions:
    raise SystemExit(f'{arm} integrity failed: rows={len(rows)}, unique={len(keys)}, actions={dict(actions)}')
print(json.dumps({'arm': arm, 'records': len(rows), 'actions': dict(actions)}, sort_keys=True))
PY
}

train_arm() {
  local arm="$1" label="$2" awt="$3" gpu="$4"
  local output="$TRAIN_DIR/$arm"
  mkdir -p "$output"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/train_projection_bc.py \
    --data_path "$label" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$output" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --action_eval_chunk_size 1 --batch_size 1 --grad_accum_steps 16 --epochs "$EPOCHS" --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 12 --num_workers 4 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 --ranking_loss_weight 0.5 --ranking_min_cost_gap 0.02 \
    --cost_att_weight 1 --cost_aql_weight 1 --cost_awt_weight "$awt" --cost_spillback_weight 0 \
    --base_policy_kl_weight 0.02 --base_policy_kl_mode expert_match --best_metric val_top1_cost_regret --best_mode min \
    --log_interval 20 --save_every 1 --seed "$SEED" --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
    > "$LOG_DIR/train_${arm}.log" 2>&1
  require_file "$output/best.pt"
  require_file "$output/epoch_${EPOCHS}.pt"
}

aggregate() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$H30_LABEL" "$H60_LABEL" "$EPOCHS" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
root, h30, h60, epochs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
labels = {'h30_balanced': h30, 'h60_balanced': h60, 'h120_balanced': root/'relabel/h120_balanced.jsonl', 'h120_awt2': root/'relabel/h120_awt2.jsonl'}
summary = {'schema': 'counterfactual_training_summary_v1', 'arms': {}}
for arm, label in labels.items():
    rows = [json.loads(line) for line in label.read_text(encoding='utf-8').splitlines() if line]
    metrics = [json.loads(line) for line in (root/'training'/arm/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
    best = min(metrics, key=lambda row: float(row.get('val_top1_cost_regret', float('inf'))))
    summary['arms'][arm] = {'label_path': str(label), 'label_records': len(rows), 'expert_action_distribution': dict(sorted(Counter(row['expert_action'] for row in rows).items())), 'best_offline_epoch': best.get('epoch'), 'best_val_top1_cost_regret': best.get('val_top1_cost_regret'), 'best_checkpoint': str(root/'training'/arm/'best.pt'), 'final_checkpoint': str(root/'training'/arm/f'epoch_{epochs}.pt')}
(root/'training_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$COLLECTED"; require_file "$H30_LABEL"; require_file "$H60_LABEL"; require_file "$FLOW_PATH"; require_file "$ROADNET"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  (( ${#GPUS[@]} == 4 )) || { printf 'GPU_LIST must contain exactly four GPU IDs\n' >&2; exit 1; }
  [[ ! -e "$RUN_ROOT" ]] || { printf 'RUN_ROOT already exists: %s\n' "$RUN_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RELABEL_DIR/h120_shards" "$TRAIN_DIR"
  export PYTHONPATH="$REPO_ROOT" OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CITYFLOW_SNAPSHOT_DUMP_RETRIES=5
  write_manifest
  validate_existing_label "$H30_LABEL" h30_balanced | tee "$STATUS_DIR/reused_h30_validation.txt"
  validate_existing_label "$H60_LABEL" h60_balanced | tee "$STATUS_DIR/reused_h60_validation.txt"
  monitor_progress & MONITOR_PID=$!
  notify_feishu "Jinan-real epoch12 four-GPU recovery started: four H120 snapshot shards without a wall-clock limit, then four concurrent Stage-2 arms; run=${RUN_ID}."

  set_stage split_h120
  split_h120_inputs | tee "$STATUS_DIR/h120_shards.txt"
  set_stage relabel_h120_four_gpu
  local -a shard_pids=()
  for shard in 0 1 2 3; do
    relabel_h120_shard "$shard" "${GPUS[$shard]}" &
    shard_pids+=("$!")
  done
  rc=0
  for pid in "${shard_pids[@]}"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { log 'at least one H120 shard failed'; exit 1; }
  set_stage merge_h120
  merge_h120 | tee "$STATUS_DIR/h120_merge.json"
  validate_label "$RELABEL_DIR/h120_balanced.jsonl" h120_balanced | tee "$STATUS_DIR/h120_validation.json"

  set_stage reweight_h120_awt2
  reweight_h120_awt2
  validate_label "$RELABEL_DIR/h120_awt2.jsonl" h120_awt2 | tee "$STATUS_DIR/h120_awt2_validation.json"

  set_stage train_four_arms
  train_arm h30_balanced "$H30_LABEL" 1 "${GPUS[0]}" & P_T30=$!
  train_arm h60_balanced "$H60_LABEL" 1 "${GPUS[1]}" & P_T60=$!
  train_arm h120_balanced "$RELABEL_DIR/h120_balanced.jsonl" 1 "${GPUS[2]}" & P_T120=$!
  train_arm h120_awt2 "$RELABEL_DIR/h120_awt2.jsonl" 2 "${GPUS[3]}" & P_T120W=$!
  rc=0
  for pid in "$P_T30" "$P_T60" "$P_T120" "$P_T120W"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { log 'at least one training arm failed'; exit 1; }
  set_stage aggregate
  aggregate | tee "$STATUS_DIR/training_summary.log"
  touch "$STATUS_DIR/completed"
  notify_feishu "Jinan-real epoch12 four-GPU recovery completed: summary=${RUN_ROOT}/training_summary.json; run=${RUN_ID}."
  log "completed: $RUN_ROOT"
}

main "$@"
