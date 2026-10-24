#!/usr/bin/env bash
# Same-flow multi-parameter policy-responsive counterfactual training from Jinan-real epoch 12.
# This workflow deliberately stops after label generation and Stage-2 fine-tuning.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan_real_simexpert_h60_att1_aql1_awt1_spill0_stage12_ml512_ga16_20260908_001305}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_12.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$SOURCE_RUN/stage1/best.pt}"
RUN_ID="${RUN_ID:-jinan_real_epoch12_counterfactual_matrix_ml512_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-1,2,3}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-2024}"

FLOW="anon_3_4_jinan_real.json"
FLOW_PATH="$REPO_ROOT/data/Jinan/3_4/$FLOW"
ROADNET="$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json"
GRID_ROWS=3
GRID_COLS=4
INTERSECTION_COUNT=$((GRID_ROWS * GRID_COLS))
SNAPSHOT_COUNT=$((RUN_COUNTS / 30))
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
COLLECTED="$RUN_ROOT/collect/decision_states.jsonl"
RELABEL_DIR="$RUN_ROOT/relabel"
TRAIN_DIR="$RUN_ROOT/training"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/launcher.log"; }
require_file() { [[ -s "$1" ]] || { log "missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "missing directory: $1" >&2; exit 1; }; }
set_stage() { printf '%s\n' "$1" > "$STATUS_DIR/stage.txt"; log "stage=$1"; }

# Feishu is best-effort and only used because this full matrix is expected to exceed one hour.
notify_feishu() {
  "$PYTHON_BIN" - "$1" <<'PY' || true
import json
import sys
import urllib.request
from pathlib import Path

try:
    config = json.loads((Path.home() / ".claude" / "feishu.json").read_text(encoding="utf-8"))
    mode = str(config.get("mode", config.get("notify_mode", ""))).lower()
    webhook = str(config.get("webhook_url", config.get("webhook", "")))
    if mode not in {"push", "on", "true", "1"} or not webhook:
        raise SystemExit(0)
    payload = json.dumps({"msg_type": "text", "content": {"text": sys.argv[1][:3500]}}, ensure_ascii=False).encode()
    request = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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
    local current_stage trained
    current_stage="$(<"$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown)"
    trained="$(rg --files "$TRAIN_DIR" -g best.pt 2>/dev/null | wc -l)"
    notify_feishu "Jinan-real epoch12 counterfactual matrix progress: stage=${current_stage}; trained_arms=${trained}/4; run=${RUN_ID}."
  done
}

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'exit_code=%s at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS_DIR/failed"
    notify_feishu "Jinan-real epoch12 counterfactual matrix failed: stage=$(<"$STATUS_DIR/stage.txt" 2>/dev/null || printf unknown); exit=${rc}; run=${RUN_ID}."
  fi
  [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap on_exit EXIT

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$SOURCE_CKPT" "$ALIGN_CKPT" "$MODEL_PATH" "$GPU_LIST" "$RUN_COUNTS" "$EPOCHS" "$SEED" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, source, align, model, gpus, counts, epochs, seed = sys.argv[1:]
digest = hashlib.sha256()
with open(source, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
payload = {
    "schema": "jinan_real_epoch12_counterfactual_matrix_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "flow": "anon_3_4_jinan_real.json",
    "scope": "same_flow",
    "source_checkpoint": source,
    "source_checkpoint_sha256": digest.hexdigest(),
    "alignment_checkpoint": align,
    "model_path": model,
    "gpus": [int(item) for item in gpus.split(",")],
    "collection": {"run_counts": int(counts), "seed": int(seed), "snapshots": int(counts) // 30, "intersections_per_snapshot": 12},
    "architecture": {"seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "hidden_dim": 256, "prefix_tokens": 32, "prefix_mode": "prefix", "max_length": 512, "prompt_mode": "queue_wait_short"},
    "relabel_protocol": {"counterfactual_controller": "responsive", "cost_scope": "global", "min_action_time": 30, "traffic_series_interval": 2},
    "arms": {
        "h30_balanced": {"horizon_seconds": 30, "cost_weights": {"att": 1.0, "aql": 1.0, "awt": 1.0, "spillback": 0.0}},
        "h60_balanced": {"horizon_seconds": 60, "cost_weights": {"att": 1.0, "aql": 1.0, "awt": 1.0, "spillback": 0.0}},
        "h120_balanced": {"horizon_seconds": 120, "cost_weights": {"att": 1.0, "aql": 1.0, "awt": 1.0, "spillback": 0.0}},
        "h120_awt2": {"horizon_seconds": 120, "cost_weights": {"att": 1.0, "aql": 1.0, "awt": 2.0, "spillback": 0.0}, "label_derivation": "reweighted from h120_balanced per-action counterfactual metrics"},
    },
    "training": {"epochs": int(epochs), "batch_size": 1, "grad_accum_steps": 16, "lr": 1e-5, "ranking_loss_weight": 0.5, "online_evaluation": False},
}
Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

validate_collection() {
  "$PYTHON_BIN" - "$COLLECTED" "$SNAPSHOT_COUNT" "$INTERSECTION_COUNT" <<'PY'
import json
import sys
from collections import Counter

source, snapshots_expected, intersections = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
groups = {}
for row in rows:
    snapshot = str(row.get("snapshot_path", ""))
    if not snapshot:
        raise SystemExit("collection integrity gate failed: empty snapshot path")
    groups.setdefault(snapshot, []).append(row)
bad = []
for snapshot, sample in groups.items():
    try:
        with open(snapshot, encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        bad.append((snapshot, type(exc).__name__))
counts = Counter(float(row.get("time_s", -1)) for row in rows)
expected_times = list(range(0, snapshots_expected * 30, 30))
missing = [item for item in expected_times if counts[item] != intersections]
print(f"records={len(rows)} snapshots={len(groups)} expected_snapshots={snapshots_expected} malformed_snapshots={len(bad)}")
if len(rows) != snapshots_expected * intersections or len(groups) != snapshots_expected or bad or missing:
    print(f"missing_or_bad_times={missing[:20]} bad_snapshots={bad[:5]}")
    raise SystemExit("collection integrity gate failed")
PY
}

relabel_arm() {
  local arm="$1" horizon="$2" att="$3" aql="$4" awt="$5" spillback="$6" gpu="$7"
  local output="$RELABEL_DIR/${arm}.jsonl"
  local runtime="$RELABEL_DIR/runtime_${arm}"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
    --input "$COLLECTED" --output "$output" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" \
    --dataset jinan --num_rows "$GRID_ROWS" --num_cols "$GRID_COLS" --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" \
    --work_dir "$runtime" --sim_horizon "$horizon" --min_action_time 30 --traffic_series_interval 2 \
    --cost_scope global --neighbor-mode responsive --inference_batch_size 32 --bf16 \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --cost_att_weight "$att" --cost_aql_weight "$aql" --cost_awt_weight "$awt" --cost_spillback_weight "$spillback" \
    > "$LOG_DIR/relabel_${arm}.log" 2>&1
  validate_label "$output" "$arm"
}

validate_label() {
  "$PYTHON_BIN" - "$1" "$2" "$SNAPSHOT_COUNT" "$INTERSECTION_COUNT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path, arm, snapshots, intersections = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
actions = Counter(str(row.get("expert_action", "")) for row in rows)
keys = {(str(row.get("snapshot_path", "")), str(row.get("intersection_id", ""))) for row in rows}
expected = snapshots * intersections
coverage_path = Path(path.replace(".jsonl", "_coverage.json"))
coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else None
if len(rows) != expected or len(keys) != expected or not actions or "" in actions or (coverage and int(coverage.get("skipped_snapshots", 0)) != 0):
    raise SystemExit(f"{arm}: expected {expected} unique labels, rows={len(rows)} unique={len(keys)} actions={dict(actions)}")
print(json.dumps({"arm": arm, "records": len(rows), "actions": actions, "coverage": coverage}, sort_keys=True))
PY
}

reweight_h120_awt2() {
  local source="$RELABEL_DIR/h120_balanced.jsonl" output="$RELABEL_DIR/h120_awt2.jsonl"
  "$PYTHON_BIN" - "$source" "$output" <<'PY'
import json
import sys
from pathlib import Path
from projection_train.action_space import ACTION_LIST
from projection_train.cost_ranking import _normalized_metric_costs

source, output = Path(sys.argv[1]), Path(sys.argv[2])
weights = {"att": 1.0, "aql": 1.0, "awt": 2.0, "spillback": 0.0}
rows = []
for line in source.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    costs = _normalized_metric_costs(row["candidate_action_metrics"], weights)
    feasible = row.get("pareto_feasible", {})
    choices = [action for action in ACTION_LIST if action in costs and feasible.get(action, True)]
    if not choices:
        choices = [action for action in ACTION_LIST if action in costs]
    base = row.get("base_action")
    expert = min(choices, key=lambda action: (costs[action], 0 if action == base else 1))
    values = sorted(costs[action] for action in choices)
    row["candidate_action_waits"] = costs
    row["expert_action"] = expert
    row["expert_action_margin"] = values[1] - values[0] if len(values) > 1 else 0.0
    base_cost = costs.get(base, 0.0)
    row["candidate_advantages"] = {action: base_cost - value for action, value in costs.items()}
    row["cost_weights"] = weights
    for action, metrics in row["candidate_action_metrics"].items():
        metrics["score"] = costs.get(action)
        metrics["weight_att"] = weights["att"]
        metrics["weight_aql"] = weights["aql"]
        metrics["weight_awt"] = weights["awt"]
        metrics["weight_spillback"] = weights["spillback"]
    rows.append(row)
output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
PY
  validate_label "$output" h120_awt2
}

train_arm() {
  local arm="$1" att="$2" aql="$3" awt="$4" spillback="$5" gpu="$6"
  local label="$RELABEL_DIR/${arm}.jsonl" output="$TRAIN_DIR/$arm"
  mkdir -p "$output"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/train_projection_bc.py \
    --data_path "$label" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$output" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --action_eval_chunk_size 1 --batch_size 1 --grad_accum_steps 16 --epochs "$EPOCHS" --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 12 --num_workers 4 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 --ranking_loss_weight 0.5 --ranking_min_cost_gap 0.02 \
    --cost_att_weight "$att" --cost_aql_weight "$aql" --cost_awt_weight "$awt" --cost_spillback_weight "$spillback" \
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
from collections import Counter
from pathlib import Path

root, epochs = Path(sys.argv[1]), int(sys.argv[2])
arms = ("h30_balanced", "h60_balanced", "h120_balanced", "h120_awt2")
summary = {"schema": "counterfactual_training_summary_v1", "arms": {}}
for arm in arms:
    label = root / "relabel" / f"{arm}.jsonl"
    rows = [json.loads(line) for line in label.read_text(encoding="utf-8").splitlines() if line]
    metrics_path = root / "training" / arm / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    best = min(metrics, key=lambda item: float(item.get("val_top1_cost_regret", float("inf")))) if metrics else {}
    summary["arms"][arm] = {
        "label_records": len(rows),
        "expert_action_distribution": dict(sorted(Counter(str(row["expert_action"]) for row in rows).items())),
        "best_offline_epoch": best.get("epoch"),
        "best_val_top1_cost_regret": best.get("val_top1_cost_regret"),
        "best_checkpoint": str(root / "training" / arm / "best.pt"),
        "final_checkpoint": str(root / "training" / arm / f"epoch_{epochs}.pt"),
    }
(root / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$REPO_ROOT/projection_train/run_projection_online.py"; require_file "$REPO_ROOT/projection_train/train_projection_bc.py"; require_file "$REPO_ROOT/scripts/relabel_policy_responsive.py"
  require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$FLOW_PATH"; require_file "$ROADNET"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  (( ${#GPUS[@]} == 3 )) || { printf 'GPU_LIST must contain exactly three GPUs, got %s\n' "$GPU_LIST" >&2; exit 1; }
  (( RUN_COUNTS % 30 == 0 )) || { printf 'RUN_COUNTS must be divisible by 30\n' >&2; exit 1; }
  [[ ! -e "$RUN_ROOT" ]] || { printf 'RUN_ROOT already exists: %s\n' "$RUN_ROOT" >&2; exit 1; }
  mkdir -p "$LOG_DIR" "$STATUS_DIR" "$RUN_ROOT/collect" "$RELABEL_DIR" "$TRAIN_DIR"
  export PYTHONPATH="$REPO_ROOT" OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CITYFLOW_SNAPSHOT_DUMP_RETRIES=5
  write_manifest
  monitor_progress & MONITOR_PID=$!
  notify_feishu "Jinan-real epoch12 counterfactual matrix started: responsive H30/H60/H120 balanced plus H120 AWTx2; same-flow; GPUs ${GPU_LIST}; run=${RUN_ID}."

  set_stage collect
  timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/collect" --projection_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --dataset jinan --traffic_file "$FLOW" \
    --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$SEED" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 \
    --bf16 --deterministic --collect_states_output "$COLLECTED" --result_json "$RUN_ROOT/collect/result.json" > "$LOG_DIR/collect.log" 2>&1
  validate_collection | tee "$STATUS_DIR/collection_validation.txt"

  set_stage relabel_matrix
  relabel_arm h30_balanced 30 1 1 1 0 "${GPUS[0]}" & P_H30=$!
  relabel_arm h60_balanced 60 1 1 1 0 "${GPUS[1]}" & P_H60=$!
  relabel_arm h120_balanced 120 1 1 1 0 "${GPUS[2]}" & P_H120=$!
  rc=0
  for pid in "$P_H30" "$P_H60" "$P_H120"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { log 'at least one relabel arm failed'; exit 1; }

  set_stage reweight_h120_awt2
  reweight_h120_awt2

  set_stage train_matrix_primary
  train_arm h30_balanced 1 1 1 0 "${GPUS[0]}" & P_T30=$!
  train_arm h60_balanced 1 1 1 0 "${GPUS[1]}" & P_T60=$!
  train_arm h120_balanced 1 1 1 0 "${GPUS[2]}" & P_T120=$!
  rc=0
  for pid in "$P_T30" "$P_T60" "$P_T120"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { log 'at least one primary training arm failed'; exit 1; }

  set_stage train_h120_awt2
  train_arm h120_awt2 1 1 2 0 "${GPUS[0]}"

  set_stage aggregate
  aggregate | tee "$STATUS_DIR/training_summary.log"
  touch "$STATUS_DIR/completed"
  notify_feishu "Jinan-real epoch12 counterfactual matrix completed: run=${RUN_ID}; summary=${RUN_ROOT}/training_summary.json."
  log "completed: $RUN_ROOT"
}

main "$@"
