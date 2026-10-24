#!/usr/bin/env bash
# Four independent same-flow Stage-2 counterfactual fine-tuning arms.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan_real_simexpert_h60_att1_aql1_awt1_spill0_stage12_ml512_ga16_20260908_001305}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_12.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$SOURCE_RUN/stage1/best.pt}"
LABEL_ROOT_BASE="${LABEL_ROOT_BASE:-$REPO_ROOT/records/counterfactual_validation/jinan_real_epoch12_counterfactual_matrix_ml512_20260908_231300/relabel}"
LABEL_ROOT_H120="${LABEL_ROOT_H120:-$REPO_ROOT/records/counterfactual_validation/jinan_real_epoch12_counterfactual_matrix_retry4gpu_20260909_081500/relabel}"
H30_LABEL="${H30_LABEL:-$LABEL_ROOT_BASE/h30_balanced.jsonl}"
H60_LABEL="${H60_LABEL:-$LABEL_ROOT_BASE/h60_balanced.jsonl}"
H120_LABEL="${H120_LABEL:-$LABEL_ROOT_H120/h120_balanced.jsonl}"
H120_AWT2_LABEL="${H120_AWT2_LABEL:-$LABEL_ROOT_H120/h120_awt2.jsonl}"
RUN_ID="${RUN_ID:-jinan_real_epoch12_counterfactual_train4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-2024}"

LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
TRAIN_DIR="$RUN_ROOT/training"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/launcher.log"; }
require_file() { [[ -s "$1" ]] || { log "missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "missing directory: $1" >&2; exit 1; }; }
set_stage() { printf '%s\n' "$1" > "$STATUS_DIR/stage.txt"; log "stage=$1"; }

# This run is expected to exceed an hour. Notification failures are non-fatal.
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
    payload = json.dumps({'msg_type': 'text', 'content': {'text': sys.argv[1][:3500]}}, ensure_ascii=False).encode()
    request = urllib.request.Request(webhook, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
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
    local current_stage completed
    current_stage="$(<"$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown)"
    completed="$(rg --files "$TRAIN_DIR" -g best.pt 2>/dev/null | wc -l)"
    notify_feishu "Jinan-real epoch12 four-GPU Stage-2 training: stage=${current_stage}; completed_arms=${completed}/4; run=${RUN_ID}."
  done
}

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'exit_code=%s at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS_DIR/failed"
    notify_feishu "Jinan-real epoch12 four-GPU Stage-2 training failed: stage=$(<"$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown); exit=${rc}; run=${RUN_ID}."
  fi
  [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap on_exit EXIT

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
if len(rows) != 1440 or len(keys) != 1440 or not actions or '' in actions:
    raise SystemExit(f'{arm} invalid labels: rows={len(rows)} unique={len(keys)} actions={dict(actions)}')
print(json.dumps({'arm': arm, 'records': len(rows), 'actions': dict(actions)}, sort_keys=True))
PY
}

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$SOURCE_CKPT" "$ALIGN_CKPT" "$GPU_LIST" "$EPOCHS" "$SEED" "$H30_LABEL" "$H60_LABEL" "$H120_LABEL" "$H120_AWT2_LABEL" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
out, source, align, gpus, epochs, seed, h30, h60, h120, h120w = sys.argv[1:]
digest = hashlib.sha256()
with open(source, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
payload = {
    'schema': 'jinan_real_epoch12_counterfactual_train4gpu_v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'flow': 'anon_3_4_jinan_real.json',
    'scope': 'same_flow',
    'source_checkpoint': source,
    'source_checkpoint_sha256': digest.hexdigest(),
    'alignment_checkpoint': align,
    'gpus': [int(item) for item in gpus.split(',')],
    'architecture': {'seq_len': 16, 'image_size': 224, 'periodicity': 16, 'input_dim': 8, 'hidden_dim': 256, 'prefix_tokens': 32, 'prefix_mode': 'prefix', 'max_length': 512, 'prompt_mode': 'queue_wait_short'},
    'training': {'epochs': int(epochs), 'batch_size': 1, 'grad_accum_steps': 16, 'lr': 1e-5, 'ranking_loss_weight': 0.5, 'ranking_min_cost_gap': 0.02, 'per_stage_wall_clock_timeout': None, 'seed': int(seed), 'online_evaluation': False},
    'arms': {
        'h30_balanced': {'labels': h30, 'cost_weights': [1, 1, 1, 0]},
        'h60_balanced': {'labels': h60, 'cost_weights': [1, 1, 1, 0]},
        'h120_balanced': {'labels': h120, 'cost_weights': [1, 1, 1, 0]},
        'h120_awt2': {'labels': h120w, 'cost_weights': [1, 1, 2, 0]},
    },
}
Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
}

train_arm() {
  local arm="$1"
  local label="$2"
  local awt="$3"
  local gpu="$4"
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
  "$PYTHON_BIN" - "$RUN_ROOT" "$EPOCHS" <<'PY'
import json
import sys
from pathlib import Path
root, epochs = Path(sys.argv[1]), int(sys.argv[2])
arms = ('h30_balanced', 'h60_balanced', 'h120_balanced', 'h120_awt2')
summary = {'schema': 'counterfactual_training_summary_v1', 'arms': {}}
for arm in arms:
    metrics = [json.loads(line) for line in (root/'training'/arm/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
    best = min(metrics, key=lambda row: float(row.get('val_top1_cost_regret', float('inf'))))
    summary['arms'][arm] = {'best_offline_epoch': best.get('epoch'), 'best_val_top1_cost_regret': best.get('val_top1_cost_regret'), 'best_checkpoint': str(root/'training'/arm/'best.pt'), 'final_checkpoint': str(root/'training'/arm/f'epoch_{epochs}.pt'), 'epochs_recorded': len(metrics)}
(root/'training_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$H30_LABEL"; require_file "$H60_LABEL"; require_file "$H120_LABEL"; require_file "$H120_AWT2_LABEL"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  (( ${#GPUS[@]} == 4 )) || { printf 'GPU_LIST must contain exactly four GPU IDs\n' >&2; exit 1; }
  [[ ! -e "$RUN_ROOT" ]] || { printf 'RUN_ROOT already exists: %s\n' "$RUN_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_DIR" "$STATUS_DIR" "$TRAIN_DIR"
  export PYTHONPATH="$REPO_ROOT" OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  write_manifest
  validate_label "$H30_LABEL" h30_balanced | tee "$STATUS_DIR/h30_label_validation.json"
  validate_label "$H60_LABEL" h60_balanced | tee "$STATUS_DIR/h60_label_validation.json"
  validate_label "$H120_LABEL" h120_balanced | tee "$STATUS_DIR/h120_label_validation.json"
  validate_label "$H120_AWT2_LABEL" h120_awt2 | tee "$STATUS_DIR/h120_awt2_label_validation.json"
  monitor_progress & MONITOR_PID=$!
  notify_feishu "Jinan-real epoch12 four-GPU Stage-2 training started: H30, H60, H120 balanced and H120 AWTx2; run=${RUN_ID}."
  set_stage train_four_arms
  train_arm h30_balanced "$H30_LABEL" 1 "${GPUS[0]}" & P_H30=$!
  train_arm h60_balanced "$H60_LABEL" 1 "${GPUS[1]}" & P_H60=$!
  train_arm h120_balanced "$H120_LABEL" 1 "${GPUS[2]}" & P_H120=$!
  train_arm h120_awt2 "$H120_AWT2_LABEL" 2 "${GPUS[3]}" & P_H120W=$!
  rc=0
  for pid in "$P_H30" "$P_H60" "$P_H120" "$P_H120W"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { log 'at least one training arm failed'; exit 1; }
  set_stage aggregate
  aggregate | tee "$STATUS_DIR/training_summary.log"
  touch "$STATUS_DIR/completed"
  notify_feishu "Jinan-real epoch12 four-GPU Stage-2 training completed: summary=${RUN_ROOT}/training_summary.json; run=${RUN_ID}."
  log "completed: $RUN_ROOT"
}

main "$@"
