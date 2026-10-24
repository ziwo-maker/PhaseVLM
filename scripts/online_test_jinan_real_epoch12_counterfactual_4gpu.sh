#!/usr/bin/env bash
# Same-flow, multi-epoch online screening for Jinan-real epoch-12 counterfactual arms.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan_real_simexpert_h60_att1_aql1_awt1_spill0_stage12_ml512_ga16_20260908_001305}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_12.pt}"
TRAIN_RUN="${TRAIN_RUN:-$REPO_ROOT/records/counterfactual_validation/jinan_real_epoch12_counterfactual_train4gpu_20260909_103000}"
RUN_ID="${RUN_ID:-jinan_real_epoch12_counterfactual_online4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/online_tests/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024}"
RUN_COUNTS="${RUN_COUNTS:-3600}"

FLOW="anon_3_4_jinan_real.json"
ARCHIVE_ROOT="$REPO_ROOT/online_test_best_weights/anon_3_4_jinan_real"
ARCHIVE_INDEX="$REPO_ROOT/online_test_best_weights/README.md"
CURRENT_ARCHIVED_AWT="${CURRENT_ARCHIVED_AWT:-32.327135}"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/launcher.log"; }
require_file() { [[ -s "$1" ]] || { log "missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "missing directory: $1" >&2; exit 1; }; }
set_stage() { printf '%s\n' "$1" > "$STATUS_DIR/stage.txt"; log "stage=$1"; }

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'exit_code=%s at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS_DIR/failed"
  fi
}
trap on_exit EXIT

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$SOURCE_CKPT" "$TRAIN_RUN" "$GPU_LIST" "$SEEDS_RAW" "$RUN_COUNTS" "$CURRENT_ARCHIVED_AWT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
out, source, training, gpus, seeds, counts, archived_awt = sys.argv[1:]
digest = hashlib.sha256()
with open(source, 'rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
payload = {
    'schema': 'jinan_real_epoch12_counterfactual_online4gpu_v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'flow': 'anon_3_4_jinan_real.json',
    'scope': 'same_flow',
    'run_counts': int(counts),
    'seeds': [int(item) for item in seeds.split()],
    'gpus': [int(item) for item in gpus.split(',')],
    'source_checkpoint': source,
    'source_checkpoint_sha256': digest.hexdigest(),
    'training_run': training,
    'architecture': {'seq_len': 16, 'image_size': 224, 'periodicity': 16, 'input_dim': 8, 'hidden_dim': 256, 'prefix_tokens': 32, 'prefix_mode': 'prefix', 'max_length': 512, 'prompt_mode': 'queue_wait_short'},
    'phase_overrides': {'max_phase_override_gap': 0, 'max_phase_override_left_wait_protect': 0, 'max_phase_override_max_wait_protect': 0, 'max_phase_override_target_queue_floor': 0, 'max_phase_override_total_queue_floor': 0},
    'selection': {'primary_metric': 'test_avg_waiting_time_over', 'tie_breakers': ['test_avg_travel_time_over', 'test_avg_queue_len_over'], 'previous_archived_awt': float(archived_awt)},
    'checkpoints': {'source_epoch12': [12], 'h30_balanced': [2, 3, 5], 'h60_balanced': [1, 2, 5], 'h120_balanced': [1, 2, 5], 'h120_awt2': [1, 2, 5]},
}
Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
}

run_job() {
  local arm="$1" epoch="$2" checkpoint="$3" seed="$4" gpu="$5"
  local output="$RESULT_DIR/$arm/epoch_${epoch}/seed_${seed}"
  local result="$output/result.json"
  mkdir -p "$output"
  [[ ! -s "$result" ]] || { log "skip existing arm=$arm epoch=$epoch seed=$seed"; return 0; }
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/$arm/epoch_${epoch}/seed_${seed}" --projection_ckpt "$checkpoint" --llm_path "$MODEL_PATH" \
    --dataset jinan --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 32 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 \
    --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
    --result_json "$result" > "$LOG_DIR/${arm}_epoch${epoch}_seed${seed}.log" 2>&1
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$REPO_ROOT/projection_train/run_projection_online.py"; require_file "$SOURCE_CKPT"; require_dir "$MODEL_PATH"; require_dir "$TRAIN_RUN"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  read -r -a SEEDS <<< "$SEEDS_RAW"
  (( ${#GPUS[@]} == 4 )) || { printf 'GPU_LIST must contain exactly four GPU IDs\n' >&2; exit 1; }
  (( ${#SEEDS[@]} >= 1 )) || { printf 'SEEDS must not be empty\n' >&2; exit 1; }
  [[ ! -e "$RUN_ROOT" ]] || { printf 'RUN_ROOT already exists: %s\n' "$RUN_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR"
  export PYTHONPATH="$REPO_ROOT" OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CITYFLOW_SNAPSHOT_DUMP_RETRIES=5
  write_manifest

  declare -a jobs=()
  local seed arm epoch checkpoint
  for seed in "${SEEDS[@]}"; do jobs+=("source_epoch12|12|$SOURCE_CKPT|$seed"); done
  for arm in h30_balanced h60_balanced h120_balanced h120_awt2; do
    case "$arm" in
      h30_balanced) epochs=(2 3 5);;
      h60_balanced|h120_balanced|h120_awt2) epochs=(1 2 5);;
    esac
    for epoch in "${epochs[@]}"; do
      checkpoint="$TRAIN_RUN/training/$arm/epoch_${epoch}.pt"
      require_file "$checkpoint"
      for seed in "${SEEDS[@]}"; do jobs+=("$arm|$epoch|$checkpoint|$seed"); done
    done
  done
  printf 'planned_jobs=%s\n' "${#jobs[@]}" > "$STATUS_DIR/progress.txt"
  set_stage online_test
  local index offset rc=0 parsed_arm parsed_epoch parsed_ckpt parsed_seed
  for ((index=0; index<${#jobs[@]}; index+=${#GPUS[@]})); do
    local -a pids=()
    for ((offset=0; offset<${#GPUS[@]} && index+offset<${#jobs[@]}; offset++)); do
      IFS='|' read -r parsed_arm parsed_epoch parsed_ckpt parsed_seed <<< "${jobs[$((index+offset))]}"
      run_job "$parsed_arm" "$parsed_epoch" "$parsed_ckpt" "$parsed_seed" "${GPUS[$offset]}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    (( rc == 0 )) || { log 'at least one online job failed'; exit 1; }
    printf 'completed_jobs=%s\n' "$((index + ${#pids[@]}))" > "$STATUS_DIR/progress.txt"
  done
  set_stage aggregate
  "$PYTHON_BIN" - "$RUN_ROOT" "$REPO_ROOT" "$CURRENT_ARCHIVED_AWT" "$ARCHIVE_ROOT" "$ARCHIVE_INDEX" "$RUN_ID" <<'PY'
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
repo = Path(sys.argv[2])
previous_awt = float(sys.argv[3])
archive_root = Path(sys.argv[4])
archive_index = Path(sys.argv[5])
run_id = sys.argv[6]
metrics = {
    'AWT': 'test_avg_waiting_time_over', 'ATT': 'test_avg_travel_time_over', 'AQL': 'test_avg_queue_len_over',
    'queued': 'test_queuing_vehicle_num_over', 'reward': 'test_reward_over', 'p95_wait': 'test_vehicle_wait_p95_over', 'max_wait': 'test_vehicle_wait_max_over',
}
rows = []
for path in sorted(root.glob('results/*/epoch_*/seed_*/result.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    result = payload.get('results', payload)
    if isinstance(result, list): result = result[-1]
    rows.append({'arm': path.parts[-4], 'epoch': int(path.parts[-3].split('_')[1]), 'seed': int(path.parts[-2].split('_')[1]), 'checkpoint': str((payload.get('args') or {}).get('projection_ckpt', payload.get('projection_ckpt', ''))), 'result_path': str(path), **{name: float(result[key]) for name, key in metrics.items()}})
groups = {}
for row in rows: groups.setdefault((row['arm'], row['epoch']), []).append(row)
summary = []
for (arm, epoch), items in sorted(groups.items()):
    summary.append({'arm': arm, 'epoch': epoch, 'n': len(items), 'checkpoint': items[0]['checkpoint'], **{name: sum(item[name] for item in items) / len(items) for name in metrics}, **{f'{name}_std': (sum((item[name] - sum(value[name] for value in items) / len(items)) ** 2 for item in items) / len(items)) ** .5 if len(items) > 1 else 0.0 for name in metrics}})
expected = 13 * len({item['seed'] for item in rows})
if len(rows) != expected:
    raise SystemExit(f'incomplete online matrix: expected {expected} rows, got {len(rows)}')
adapted = [row for row in summary if row['arm'] != 'source_epoch12']
best = min(adapted, key=lambda row: (row['AWT'], row['ATT'], row['AQL']))
payload = {'schema': 'online_checkpoint_screening_v1', 'created_at': datetime.now(timezone.utc).isoformat(), 'per_seed': rows, 'by_checkpoint': summary, 'source_epoch12': next(row for row in summary if row['arm'] == 'source_epoch12'), 'best_adapted_by_awt': best, 'previous_archived_awt': previous_awt}
archive = None
if best['AWT'] < previous_awt:
    source = Path(best['checkpoint'])
    if not source.is_file():
        raise SystemExit(f'cannot archive missing checkpoint: {source}')
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f'jinan_real_epoch12_counterfactual_{best["arm"]}_epoch{best["epoch"]}_seed{rows[0]["seed"]}.pt'
    shutil.copy2(source, target)
    digest = hashlib.sha256()
    with target.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    digest = digest.hexdigest()
    command = f'CUDA_VISIBLE_DEVICES=<gpu> {repo}/projection_train/run_projection_online.py --projection_ckpt {source} --llm_path /mnt/data/Qwen3-VL-8B-Instruct --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --seed {rows[0]["seed"]} --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0'
    entry = f'''\n### {datetime.now().date()} anon_3_4_jinan_real epoch12_counterfactual_{best['arm']}_epoch{best['epoch']}\n\n- Flow: `anon_3_4_jinan_real.json`\n- Selection: same-flow online checkpoint screening across H30 epochs 2/3/5, H60 epochs 1/2/5, H120 epochs 1/2/5, and H120-AWTx2 epochs 1/2/5; AWT primary with ATT/AQL tie-breakers.\n- Metric: `test_avg_waiting_time_over` (lower is better).\n- Previous archived AWT: `{previous_awt:.6f}`.\n- New AWT: `{best['AWT']:.6f}`; ATT `{best['ATT']:.6f}`; AQL `{best['AQL']:.6f}`; queued `{best['queued']:.0f}`; reward `{best['reward']:.6f}`; P95 `{best['p95_wait']:.2f}`; max wait `{best['max_wait']:.2f}`.\n- Source checkpoint: `{source}`\n- Archived weight: `{target.relative_to(repo)}`\n- Checkpoint SHA256: `{digest}`\n- Scope: same-flow, run_counts=3600, seed(s) `{sorted({item['seed'] for item in rows})}`, Qwen3-VL-8B-Instruct, max_length=512, snapshots and phase overrides disabled. This is an online-screening result; no multi-seed claim is made when only one seed is configured.\n- Result artifacts: `{root / 'aggregate.json'}` and `{root / 'results' / best['arm'] / f"epoch_{best['epoch']}"}`\n- Command: `{command}`\n'''
    with archive_index.open('a', encoding='utf-8') as handle: handle.write(entry)
    archive = {'archived_weight': str(target), 'sha256': digest, 'previous_awt': previous_awt, 'new_awt': best['AWT']}
payload['archive'] = archive
(root / 'aggregate.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'best_adapted_by_awt': best, 'archive': archive}, ensure_ascii=False))
PY
  touch "$STATUS_DIR/completed"
  log "completed: $RUN_ROOT"
}

main "$@"
