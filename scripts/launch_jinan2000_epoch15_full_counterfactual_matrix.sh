#!/usr/bin/env bash
# Same-flow policy-responsive SimExpert counterfactual matrix from Stage-2 epoch 15.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan2000_simexpert_best_ml512_bs1ga16_20260906_110909}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_15.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$SOURCE_RUN/stage1/best.pt}"
RUN_ID="${RUN_ID:-jinan2000_epoch15_full_counterfactual_matrix_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EVAL_SEEDS_RAW="${EVAL_SEEDS:-2024 42 123}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
SNAPSHOT_RETRIES="${SNAPSHOT_RETRIES:-5}"

FLOW="anon_3_4_jinan_real_2000.json"
FLOW_PATH="$REPO_ROOT/data/Jinan/3_4/$FLOW"
ROADNET="$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json"
GRID_ROWS=3
GRID_COLS=4
INTERSECTION_COUNT=$((GRID_ROWS * GRID_COLS))
SNAPSHOT_COUNT=$((RUN_COUNTS / 30))
COLLECTED="$RUN_ROOT/collect/decision_states.jsonl"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
RELABELED_DIR="$RUN_ROOT/relabel"
TRAIN_DIR="$RUN_ROOT/training"
ONLINE_DIR="$RUN_ROOT/online"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/launcher.log"; }
require_file() { [[ -s "$1" ]] || { log "Missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "Missing directory: $1" >&2; exit 1; }; }
set_stage() { printf '%s\n' "$1" > "$STATUS_DIR/stage.txt"; log "stage=$1"; }

notify_feishu() {
  local message="$1"
  "$PYTHON_BIN" - "$message" <<'PY' || true
import json
import sys
import urllib.request
from pathlib import Path

try:
    config_path = Path.home() / ".claude" / "feishu.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode = str(config.get("mode", config.get("notify_mode", ""))).lower()
    webhook = str(config.get("webhook_url", config.get("webhook", "")))
    if mode not in {"push", "on", "true", "1"} or not webhook:
        raise SystemExit(0)
    payload = json.dumps({"msg_type": "text", "content": {"text": sys.argv[1][:4000]}}, ensure_ascii=False).encode()
    request = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read(4096)
except Exception:
    pass
PY
}

monitor_progress() {
  while [[ ! -f "$STATUS_DIR/completed" && ! -f "$STATUS_DIR/failed" ]]; do
    sleep 2400
    [[ -f "$STATUS_DIR/completed" || -f "$STATUS_DIR/failed" ]] && break
    local stage completed
    stage="$(cat "$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown)"
    completed="$(find "$ONLINE_DIR" -path '*/result.json' -type f -printf . 2>/dev/null | wc -c)"
    notify_feishu "Jinan2000 epoch15 counterfactual matrix progress: stage=${stage}; completed_online_results=${completed}/63; run=${RUN_ID}."
  done
}

on_exit() {
  local code=$?
  if [[ "$code" -ne 0 ]]; then
    printf 'failed exit_code=%s at=%s\n' "$code" "$(date --iso-8601=seconds)" > "$STATUS_DIR/failed"
    notify_feishu "Jinan2000 epoch15 counterfactual matrix failed: stage=$(cat "$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown), exit=${code}; run=${RUN_ID}."
  fi
  [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap on_exit EXIT

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$SOURCE_CKPT" "$ALIGN_CKPT" "$MODEL_PATH" "$RUN_COUNTS" "$FINETUNE_EPOCHS" "$GPU_LIST" "$EVAL_SEEDS_RAW" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, source_ckpt, align_ckpt, model_path, run_counts, epochs, gpus, seeds = sys.argv[1:]
def sha256(value):
    digest = hashlib.sha256()
    with open(value, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "schema": "jinan2000_epoch15_full_counterfactual_matrix_v1",
    "flow": "anon_3_4_jinan_real_2000.json",
    "evaluation_scope": "same-flow",
    "source_checkpoint": source_ckpt,
    "source_checkpoint_sha256": sha256(source_ckpt),
    "align_checkpoint": align_ckpt,
    "model_path": model_path,
    "gpus": gpus,
    "collection": {"run_counts": int(run_counts), "seed": 2024, "snapshots": int(run_counts) // 30, "intersections_per_snapshot": 12},
    "architecture": {"seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "hidden_dim": 64, "prefix_tokens": 8, "prefix_mode": "prefix", "max_length": 512},
    "teacher_cost_weights": {"att": 1.0, "aql": 1.0, "awt": 2.0, "spillback": 0.2},
    "arms": {
        "responsive_h30_r005": {"horizon": 30, "ranking_loss_weight": 0.05},
        "responsive_h120_r000": {"horizon": 120, "ranking_loss_weight": 0.0},
        "responsive_h120_r005": {"horizon": 120, "ranking_loss_weight": 0.05},
        "responsive_h120_r015": {"horizon": 120, "ranking_loss_weight": 0.15},
    },
    "finetune": {"epochs": int(epochs), "batch_size_per_rank": 1, "grad_accum_steps": 16, "world_size": 4, "lr": 1e-5},
    "online": {"run_counts": int(run_counts), "seeds": [int(x) for x in seeds.split()], "source_baseline_jobs": 3, "adapted_jobs": 60},
}
Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

validate_collection() {
  "$PYTHON_BIN" - "$COLLECTED" "$SNAPSHOT_COUNT" "$INTERSECTION_COUNT" <<'PY'
import json
import sys
from collections import Counter

path, snapshot_count, intersection_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
by_time = Counter(float(row.get("time_s", -1)) for row in rows)
snapshots = {}
for row in rows:
    snapshots.setdefault(str(row.get("snapshot_path", "")), float(row.get("time_s", -1)))
expected_times = list(range(0, snapshot_count * 30, 30))
missing = [time_s for time_s in expected_times if by_time[time_s] != intersection_count]
bad = []
for snapshot, time_s in snapshots.items():
    try:
        with open(snapshot, encoding="utf-8") as handle:
            json.load(handle)
    except Exception as exc:
        bad.append((time_s, type(exc).__name__))
print(f"rows={len(rows)} snapshots={len(snapshots)} expected_snapshots={snapshot_count} bad_snapshots={len(bad)}")
if len(rows) != snapshot_count * intersection_count or len(snapshots) != snapshot_count or missing or bad:
    print("incomplete_times=", missing[:20])
    print("bad_snapshots=", bad[:10])
    raise SystemExit("collection integrity gate failed")
PY
}

split_and_relabel() {
  local horizon="$1"
  local label="$2"
  local split_root="$RELABELED_DIR/h${horizon}"
  mkdir -p "$split_root"
  "$PYTHON_BIN" - "$COLLECTED" "$split_root" "$SNAPSHOT_COUNT" <<'PY'
import json
import os
import sys

source, out_dir, expected_snapshots = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
groups = {}
for row in rows:
    groups.setdefault(str(row["snapshot_path"]), []).append(row)
snapshots = sorted(groups)
if len(snapshots) != expected_snapshots:
    raise SystemExit(f"expected {expected_snapshots} snapshots, got {len(snapshots)}")
for shard in range(4):
    selected = snapshots[shard::4]
    target = os.path.join(out_dir, f"input_shard{shard}.jsonl")
    with open(target, "w", encoding="utf-8") as handle:
        for snapshot in selected:
            for row in groups[snapshot]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"shard={shard} snapshots={len(selected)} rows={sum(len(groups[x]) for x in selected)}")
PY
  local -a pids=()
  local shard gpu
  for shard in 0 1 2 3; do
    gpu="${GPUS[$shard]}"
    timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
      --input "$split_root/input_shard${shard}.jsonl" --output "$split_root/label_shard${shard}.jsonl" \
      --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" --dataset jinan --num_rows "$GRID_ROWS" --num_cols "$GRID_COLS" \
      --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --work_dir "$split_root/runtime_shard${shard}" \
      --sim_horizon "$horizon" --min_action_time 30 --traffic_series_interval 2 --cost_scope global --neighbor-mode responsive \
      --inference_batch_size 32 --bf16 --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
      --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
      --cost_att_weight 1.0 --cost_aql_weight 1.0 --cost_awt_weight 2.0 --cost_spillback_weight 0.2 \
      > "$LOG_DIR/relabel_h${horizon}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { log "H${horizon} relabel failure"; return 1; }
  "$PYTHON_BIN" - "$split_root" "$label" "$SNAPSHOT_COUNT" "$INTERSECTION_COUNT" <<'PY'
import json
import sys
from pathlib import Path

root, output, snapshots, intersections = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = []
coverage = []
for shard in range(4):
    label_path = root / f"label_shard{shard}.jsonl"
    coverage_path = root / f"label_shard{shard}_coverage.json"
    rows.extend(json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines() if line)
    coverage.append(json.loads(coverage_path.read_text(encoding="utf-8")))
expected_rows = snapshots * intersections
if len(rows) != expected_rows:
    raise SystemExit(f"expected {expected_rows} labels, got {len(rows)}")
if any(int(item.get("skipped_snapshots", 0)) for item in coverage):
    raise SystemExit("relabel integrity gate failed: skipped snapshots")
output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
summary = {key: sum(int(item.get(key, 0)) for item in coverage) for key in ("total_records", "total_snapshots", "selected_snapshots", "processed_snapshots", "skipped_snapshots", "skipped_records", "candidate_rollouts")}
(output.parent / f"{output.stem}_coverage.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
}

train_arm() {
  local arm="$1"
  local labels="$2"
  local ranking_weight="$3"
  local output="$TRAIN_DIR/$arm"
  mkdir -p "$output"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$TORCHRUN_BIN" --standalone --nproc_per_node=4 projection_train/train_projection_bc.py \
    --data_path "$labels" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$output" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --prompt_mode queue_wait_short --action_eval_chunk_size 1 --batch_size 1 --grad_accum_steps 16 --epochs "$FINETUNE_EPOCHS" \
    --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 12 --min_action_margin 0.02 --num_workers 4 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 --ranking_loss_weight "$ranking_weight" \
    --cost_att_weight 1.0 --cost_aql_weight 1.0 --cost_awt_weight 2.0 --cost_spillback_weight 0.2 \
    --base_policy_kl_weight 0.02 --base_policy_kl_mode expert_match --best_metric val_top1_cost_regret --best_mode min \
    --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
    > "$LOG_DIR/train_${arm}.log" 2>&1
  require_file "$output/epoch_${FINETUNE_EPOCHS}.pt"
}

run_online() {
  local arm="$1"
  local checkpoint="$2"
  local epoch="$3"
  local seed="$4"
  local gpu="$5"
  local output="$ONLINE_DIR/$arm/epoch_${epoch}/seed_${seed}"
  mkdir -p "$output"
  timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/$arm/epoch_${epoch}/seed_${seed}" --projection_ckpt "$checkpoint" --llm_path "$MODEL_PATH" \
    --dataset jinan --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 \
    --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --result_json "$output/result.json" > "$LOG_DIR/online_${arm}_epoch${epoch}_seed${seed}.log" 2>&1
}

aggregate() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$FINETUNE_EPOCHS" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root, finetune_epochs = Path(sys.argv[1]), int(sys.argv[2])
metrics = {"AWT": "test_avg_waiting_time_over", "ATT": "test_avg_travel_time_over", "AQL": "test_avg_queue_len_over", "queued": "test_queuing_vehicle_num_over", "reward": "test_reward_over", "p95_wait": "test_vehicle_wait_p95_over", "max_wait": "test_vehicle_wait_max_over"}
groups = {}
for path in root.glob("online/*/epoch_*/seed_*/result.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("results") or payload
    if isinstance(result, list):
        result = result[-1]
    arm = path.parts[-4]
    epoch = int(path.parts[-3].split("_")[1])
    seed = int(path.parts[-2].split("_")[1])
    groups.setdefault((arm, epoch), []).append({"seed": seed, "path": str(path), **{name: float(result[key]) for name, key in metrics.items()}})
rows = []
for (arm, epoch), values in sorted(groups.items()):
    rows.append({"arm": arm, "epoch": epoch, "n": len(values), "per_seed": values, **{name: statistics.fmean(value[name] for value in values) for name in metrics}, **{f"{name}_std": statistics.pstdev(value[name] for value in values) if len(values) > 1 else 0.0 for name in metrics}})
expected_groups = 1 + 4 * finetune_epochs
if len(rows) != expected_groups or any(row["n"] != 3 for row in rows):
    raise SystemExit(f"online completeness gate failed: expected {expected_groups} three-seed groups, got {len(rows)}")
source = next(row for row in rows if row["arm"] == "source_epoch15")
adapted = [row for row in rows if row["arm"] != "source_epoch15"]
best_by_awt = min(adapted, key=lambda row: (row["AWT"], row["ATT"], row["AQL"]))
summary = {"source_baseline": source, "by_checkpoint": rows, "best_adapted_by_awt": best_by_awt}
(root / "online" / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(best_by_awt, sort_keys=True))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$TORCHRUN_BIN"; require_file "$REPO_ROOT/projection_train/run_projection_online.py"; require_file "$REPO_ROOT/projection_train/train_projection_bc.py"; require_file "$REPO_ROOT/scripts/relabel_policy_responsive.py"
  require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$FLOW_PATH"; require_file "$ROADNET"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_RAW"
  (( ${#GPUS[@]} == 4 )) || { log "GPU_LIST must contain exactly four GPUs" >&2; exit 1; }
  (( ${#EVAL_SEEDS[@]} == 3 )) || { log "EVAL_SEEDS must contain exactly three seeds" >&2; exit 1; }
  [[ ! -e "$RUN_ROOT" ]] || { log "Run root already exists: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RUN_ROOT/collect" "$RELABELED_DIR" "$TRAIN_DIR" "$ONLINE_DIR"
  write_manifest
  export PYTHONPATH="$REPO_ROOT" OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CITYFLOW_SNAPSHOT_DUMP_RETRIES="$SNAPSHOT_RETRIES"
  monitor_progress & MONITOR_PID=$!
  notify_feishu "Jinan2000 epoch15 full counterfactual matrix started: H30 r=.05; H120 r=0/.05/.15; four-GPU DDP; run=${RUN_ID}."

  set_stage collect
  timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/collect" --projection_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --dataset jinan --traffic_file "$FLOW" \
    --run_counts "$RUN_COUNTS" --num_rounds 1 --seed 2024 --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 \
    --bf16 --deterministic --collect_states_output "$COLLECTED" --result_json "$RUN_ROOT/collect/result.json" > "$LOG_DIR/collect.log" 2>&1
  validate_collection | tee "$STATUS_DIR/collection_validation.txt"

  set_stage relabel_h30
  H30_LABEL="$RELABELED_DIR/responsive_h30.jsonl"
  split_and_relabel 30 "$H30_LABEL"
  set_stage relabel_h120
  H120_LABEL="$RELABELED_DIR/responsive_h120.jsonl"
  split_and_relabel 120 "$H120_LABEL"

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
  local arm epoch seed index worker checkpoint
  for seed in "${EVAL_SEEDS[@]}"; do jobs+=("source_epoch15:$SOURCE_CKPT:15:$seed"); done
  for arm in responsive_h30_r005 responsive_h120_r000 responsive_h120_r005 responsive_h120_r015; do
    for ((epoch=1; epoch<=FINETUNE_EPOCHS; epoch++)); do
      checkpoint="$TRAIN_DIR/$arm/epoch_${epoch}.pt"
      require_file "$checkpoint"
      for seed in "${EVAL_SEEDS[@]}"; do jobs+=("$arm:$checkpoint:$epoch:$seed"); done
    done
  done
  for ((index=0; index<${#jobs[@]}; index+=4)); do
    pids=()
    for ((worker=0; worker<4 && index+worker<${#jobs[@]}; worker++)); do
      IFS=: read -r arm checkpoint epoch seed <<< "${jobs[$((index+worker))]}"
      run_online "$arm" "$checkpoint" "$epoch" "$seed" "${GPUS[$worker]}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
  set_stage aggregate
  aggregate | tee "$STATUS_DIR/best_adapted_by_awt.json"
  touch "$STATUS_DIR/completed"
  notify_feishu "Jinan2000 epoch15 full counterfactual matrix completed: $(cat "$STATUS_DIR/best_adapted_by_awt.json"); run=${RUN_ID}."
  log "completed: $RUN_ROOT"
}

main "$@"
