#!/usr/bin/env bash
# Same-flow policy-responsive counterfactual validation for the designated
# Hangzhou-real Stage-2 epoch 6 checkpoint.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/hangzhou_real_simexpert_h30_att1_aql1_awt1_spill0p2_stage12_ml512_ga16_20260908_004432}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/stage2/epoch_6.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$SOURCE_RUN/stage1/best.pt}"
RUN_ID="${RUN_ID:-hangzhou_real_epoch6_counterfactual_h30_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
FLOW="anon_4_4_hangzhou_real.json"
FLOW_PATH="$REPO_ROOT/data/Hangzhou/4_4/$FLOW"
ROADNET="$REPO_ROOT/data/Hangzhou/4_4/roadnet_4_4.json"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EPOCHS="${EPOCHS:-5}"

require_file() { [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

write_manifest() {
  RUN_ROOT="$RUN_ROOT" RUN_ID="$RUN_ID" SOURCE_RUN="$SOURCE_RUN" SOURCE_CKPT="$SOURCE_CKPT" ALIGN_CKPT="$ALIGN_CKPT" SOURCE_SHA256="$SOURCE_SHA256" FLOW="$FLOW" RUN_COUNTS="$RUN_COUNTS" GPU_LIST="$GPU_LIST" SEEDS_RAW="$SEEDS_RAW" EPOCHS="$EPOCHS" "$PYTHON_BIN" - <<'PY'
import json, os
from datetime import datetime, timezone
payload = {
    "schema": "hangzhou_real_epoch6_policy_responsive_counterfactual_h30_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_run": os.environ["SOURCE_RUN"], "source_checkpoint": os.environ["SOURCE_CKPT"],
    "source_checkpoint_sha256": os.environ["SOURCE_SHA256"], "align_checkpoint": os.environ["ALIGN_CKPT"],
    "flow": os.environ["FLOW"], "scope": "same_flow",
    "architecture": {"seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "hidden_dim": 256, "prefix_tokens": 32, "prefix_mode": "prefix", "max_length": 512},
    "collection": {"seed": 2024, "run_counts": int(os.environ["RUN_COUNTS"]), "snapshots": True},
    "counterfactual": {"teacher": "policy_responsive", "horizon_seconds": 30, "neighbor_mode": "responsive", "cost_scope": "global"},
    "training": {"gpus": os.environ["GPU_LIST"], "batch_size_per_rank": 1, "grad_accum_steps": 8, "epochs": int(os.environ["EPOCHS"]), "init_policy_checkpoint": os.environ["SOURCE_CKPT"]},
    "online_evaluation": {"seeds": [int(x) for x in os.environ["SEEDS_RAW"].split()], "run_counts": int(os.environ["RUN_COUNTS"]), "all_saved_epochs": True},
}
with open(os.path.join(os.environ["RUN_ROOT"], "manifest.json"), "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
PY
}

validate_collected() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
snapshots = sorted({str(row.get("snapshot_path", "")) for row in rows if row.get("snapshot_path")})
bad = []
for snapshot in snapshots:
    try:
        with open(snapshot, encoding="utf-8") as f: json.load(f)
    except (OSError, json.JSONDecodeError): bad.append(snapshot)
if bad:
    invalid = set(bad)
    with open(path + ".unfiltered", "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    rows = [row for row in rows if row.get("snapshot_path") not in invalid]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    snapshots = [item for item in snapshots if item not in invalid]
print(f"collected_records={len(rows)} valid_snapshots={len(snapshots)} filtered_snapshots={len(bad)}")
if len(rows) < 32 or len(snapshots) < 2: raise SystemExit("insufficient valid counterfactual states")
PY
}

relabel_parallel() {
  local collected="$1" labels="$2" shard_dir="$RUN_ROOT/relabel/shards"
  mkdir -p "$shard_dir"
  "$PYTHON_BIN" - "$collected" "$shard_dir" <<'PY'
import json, sys
from pathlib import Path
rows = [line for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) < 4: raise SystemExit("not enough records to shard")
root = Path(sys.argv[2])
handles = [(root / f"input_{i}.jsonl").open("w", encoding="utf-8") for i in range(4)]
try:
    for index, row in enumerate(rows): handles[index % 4].write(row + "\n")
finally:
    for handle in handles: handle.close()
PY
  local shard gpu pid rc=0 pids=()
  for shard in 0 1 2 3; do
    gpu="${GPUS[$shard]}"
    timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
      --input "$shard_dir/input_${shard}.jsonl" --output "$shard_dir/labels_${shard}.jsonl" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" --dataset hangzhou --num_rows 4 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --work_dir "$RUN_ROOT/runtime_h30_shard${shard}" --sim_horizon 30 --min_action_time 30 --traffic_series_interval 2 --cost_scope global --neighbor-mode responsive --inference_batch_size 32 --bf16 --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 > "$RUN_ROOT/logs/relabel_h30_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || return "$rc"
  "$PYTHON_BIN" - "$labels" "$shard_dir" <<'PY'
import json, sys
from pathlib import Path
out, root = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for path in sorted(root.glob("labels_*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
if not rows: raise SystemExit("parallel relabeling produced no records")
rows.sort(key=lambda row: (str(row.get("snapshot_path", "")), str(row.get("intersection_id", ""))))
with out.open("w", encoding="utf-8") as handle:
    for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"merged_label_records={len(rows)}")
PY
}

run_online() {
  local epoch="$1" seed="$2" gpu="$3" output="$RUN_ROOT/online/epoch_${epoch}/seed_${seed}"
  mkdir -p "$output"
  timeout --foreground 3600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/epoch_${epoch}/seed_${seed}" --projection_ckpt "$RUN_ROOT/training/epoch_${epoch}.pt" --llm_path "$MODEL_PATH" \
    --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --result_json "$output/result.json" > "$RUN_ROOT/logs/online_epoch${epoch}_seed${seed}.log" 2>&1
}

aggregate_online() {
  RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY'
import json, os, statistics
from pathlib import Path
root = Path(os.environ["RUN_ROOT"]); keys = {"AWT":"test_avg_waiting_time_over", "ATT":"test_avg_travel_time_over", "AQL":"test_avg_queue_len_over", "queued":"test_queuing_vehicle_num_over"}; groups = {}
for path in sorted(root.glob("online/epoch_*/seed_*/result.json")):
    data = json.loads(path.read_text(encoding="utf-8")); data = data.get("results", data)
    epoch, seed = int(path.parts[-3].split("_")[1]), int(path.parts[-2].split("_")[1])
    groups.setdefault(epoch, []).append({"seed": seed, **{key: float(data[value]) for key, value in keys.items()}})
rows = [{"epoch": epoch, "n": len(items), "per_seed": items, **{key: statistics.fmean(item[key] for item in items) for key in keys}} for epoch, items in sorted(groups.items())]
summary = {"by_epoch": rows, "best_by_awt": min(rows, key=lambda row: (row["AWT"], row["ATT"], row["AQL"])) if rows else None}
(root / "online" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
}

main() {
  cd "$REPO_ROOT"; require_file "$PYTHON_BIN"; require_file "$TORCHRUN"; require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$FLOW_PATH"; require_file "$ROADNET"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"; read -r -a SEEDS <<< "$SEEDS_RAW"; (( ${#GPUS[@]} == 4 )) || { echo "GPU_LIST must contain exactly four GPUs" >&2; exit 1; }
  SOURCE_SHA256="$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')"; mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/collect" "$RUN_ROOT/relabel" "$RUN_ROOT/training" "$RUN_ROOT/online"; write_manifest
  export PYTHONPATH="$REPO_ROOT" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  local collected="$RUN_ROOT/collect/decision_states.jsonl" labels="$RUN_ROOT/relabel/policy_responsive_h30.jsonl"
  log "collecting same-flow decision states with replay snapshots"
  timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/collect" --projection_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed 2024 \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --collect_states_output "$collected" --result_json "$RUN_ROOT/collect/result.json" > "$RUN_ROOT/logs/collect.log" 2>&1
  validate_collected "$collected"
  log "relabeling policy-responsive H30 counterfactual branches on four GPUs"
  relabel_parallel "$collected" "$labels"
  require_file "$labels"; log "four-GPU counterfactual Stage-2 adaptation"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 projection_train/train_projection_bc.py \
    --data_path "$labels" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$RUN_ROOT/training" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --batch_size 1 --grad_accum_steps 8 --epochs "$EPOCHS" --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight 0.5 --cost_awt_weight 2.0 --cost_aql_weight 1.0 --best_metric val_top1_cost_regret --best_mode min --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 > "$RUN_ROOT/logs/train.log" 2>&1
  require_file "$RUN_ROOT/training/best.pt"; local tasks=() epoch seed index j pids=()
  for ((epoch=1; epoch<=EPOCHS; epoch++)); do require_file "$RUN_ROOT/training/epoch_${epoch}.pt"; for seed in "${SEEDS[@]}"; do tasks+=("$epoch:$seed"); done; done
  log "starting three-seed online evaluation for every saved epoch"
  for ((index=0; index<${#tasks[@]}; index+=${#GPUS[@]})); do pids=(); for ((j=0; j<${#GPUS[@]} && index+j<${#tasks[@]}; j++)); do IFS=: read -r epoch seed <<< "${tasks[$((index+j))]}"; run_online "$epoch" "$seed" "${GPUS[$j]}" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done; done
  aggregate_online; log "completed: $RUN_ROOT"
}
main "$@"
