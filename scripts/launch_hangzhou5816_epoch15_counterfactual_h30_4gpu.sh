#!/usr/bin/env bash
# Same-flow responsive counterfactual adaptation starting from the user-designated
# Hangzhou_5816 epoch-15 archive. It collects target states, relabels H=30
# branches, trains on four GPUs, then evaluates every saved epoch on three seeds.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO_ROOT/online_test_best_weights/anon_4_4_hangzhou_real_5816/hangzhou5816_maxlen512_epoch15_3seed_userdesignated.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$REPO_ROOT/records/pipeline_runs/hangzhou5816_ref_stage1_stage2_4gpu_ml512_20260905_102007/stage1/best.pt}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-4806f3d56f2c7409925c0d067151b04ce06bbd4e0fb6efaa1b93f10dc5be9397}"
RUN_ID="${RUN_ID:-hangzhou5816_epoch15_counterfactual_h30_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
FLOW="anon_4_4_hangzhou_real_5816.json"
FLOW_PATH="$REPO_ROOT/data/Hangzhou/4_4/$FLOW"
ROADNET="$REPO_ROOT/data/Hangzhou/4_4/roadnet_4_4.json"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EPOCHS="${EPOCHS:-5}"

require_file() { [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

write_manifest() {
  RUN_ROOT="$RUN_ROOT" RUN_ID="$RUN_ID" SOURCE_CKPT="$SOURCE_CKPT" ALIGN_CKPT="$ALIGN_CKPT" SOURCE_SHA256="$SOURCE_SHA256" \
  FLOW="$FLOW" RUN_COUNTS="$RUN_COUNTS" GPU_LIST="$GPU_LIST" SEEDS_RAW="$SEEDS_RAW" EPOCHS="$EPOCHS" "$PYTHON_BIN" - <<'PY'
import json, os
from datetime import datetime, timezone
payload = {
    "schema": "hangzhou5816_epoch15_policy_responsive_counterfactual_h30_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "flow": os.environ["FLOW"], "scope": "same_flow",
    "source_checkpoint": os.environ["SOURCE_CKPT"],
    "source_checkpoint_sha256": os.environ["SOURCE_SHA256"],
    "selection_scope": "user-designated current global best; see online_test_best_weights README audit note",
    "align_checkpoint": os.environ["ALIGN_CKPT"],
    "architecture": {"seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "hidden_dim": 256, "prefix_tokens": 32, "prefix_mode": "prefix", "max_length": 512},
    "collection": {"run_counts": int(os.environ["RUN_COUNTS"]), "seed": 2024, "snapshots": True},
    "counterfactual": {"teacher": "policy_responsive", "horizon_seconds": 30, "neighbor_mode": "responsive", "cost_scope": "global"},
    "training": {"gpus": os.environ["GPU_LIST"], "batch_size_per_rank": 1, "grad_accum_steps": 8, "epochs": int(os.environ["EPOCHS"]), "init_policy_checkpoint": os.environ["SOURCE_CKPT"]},
    "online_evaluation": {"seeds": [int(x) for x in os.environ["SEEDS_RAW"].split()], "run_counts": int(os.environ["RUN_COUNTS"]), "all_saved_epochs": True},
}
with open(os.path.join(os.environ["RUN_ROOT"], "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

validate_collected() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
snapshots = sorted({str(row.get("snapshot_path", "")) for row in rows if row.get("snapshot_path")})
bad = []
for snap in snapshots:
    try:
        with open(snap, encoding="utf-8") as f: json.load(f)
    except (OSError, json.JSONDecodeError): bad.append(snap)
if bad:
    badset = set(bad)
    original_rows = rows
    rows = [row for row in rows if row.get("snapshot_path") not in badset]
    with open(path + ".unfiltered", "w", encoding="utf-8") as f:
        for row in original_rows: f.write(json.dumps(row) + "\n")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    snapshots = [s for s in snapshots if s not in badset]
print(f"collected_records={len(rows)} valid_snapshots={len(snapshots)} filtered_snapshots={len(bad)}")
if len(rows) < 32 or len(snapshots) < 2: raise SystemExit("insufficient valid counterfactual states")
PY
}

run_online() {
  local checkpoint="$1" seed="$2" gpu="$3"
  local output="$RUN_ROOT/online/epoch_${checkpoint}/seed_${seed}"
  mkdir -p "$output"
  timeout --foreground 3600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/epoch_${checkpoint}/seed_${seed}" --projection_ckpt "$RUN_ROOT/training/epoch_${checkpoint}.pt" --llm_path "$MODEL_PATH" \
    --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 --result_json "$output/result.json" > "$RUN_ROOT/logs/online_epoch${checkpoint}_seed${seed}.log" 2>&1
}

aggregate_online() {
  RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY'
import json, statistics, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
keys = {"AWT":"test_avg_waiting_time_over", "ATT":"test_avg_travel_time_over", "AQL":"test_avg_queue_len_over", "queued":"test_queuing_vehicle_num_over"}
groups = {}
for path in sorted(root.glob("online/epoch_*/seed_*/result.json")):
    data = json.loads(path.read_text()).get("results", json.loads(path.read_text()))
    epoch, seed = int(path.parts[-3].split("_")[1]), int(path.parts[-2].split("_")[1])
    groups.setdefault(epoch, []).append({"seed":seed, **{k:float(data[v]) for k,v in keys.items()}})
out = {"by_epoch": []}
for epoch, rows in sorted(groups.items()):
    out["by_epoch"].append({"epoch":epoch, "n":len(rows), "per_seed":rows, **{k:statistics.fmean(x[k] for x in rows) for k in keys}})
out["best_by_awt"] = min(out["by_epoch"], key=lambda x:(x["AWT"],x["ATT"],x["AQL"])) if out["by_epoch"] else None
(root / "online" / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$TORCHRUN"; require_file "$SOURCE_CKPT"; require_file "$ALIGN_CKPT"; require_file "$FLOW_PATH"; require_file "$ROADNET"; require_dir "$MODEL_PATH"
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  read -r -a SEEDS <<< "$SEEDS_RAW"
  (( ${#GPUS[@]} == 4 )) || { echo "GPU_LIST must contain exactly four GPUs" >&2; exit 1; }
  SOURCE_SHA256="$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')"
  [[ "$SOURCE_SHA256" == "$EXPECTED_SOURCE_SHA256" ]] || { echo "source checkpoint SHA256 mismatch: $SOURCE_SHA256" >&2; exit 1; }
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/collect" "$RUN_ROOT/training" "$RUN_ROOT/online"
  write_manifest
  export PYTHONPATH="$REPO_ROOT" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  local collected="$RUN_ROOT/collect/decision_states.jsonl" labels="$RUN_ROOT/relabel/policy_responsive_h30.jsonl"
  if [[ ! -s "$collected" ]]; then
    log "collecting same-flow decision states with replay snapshots"
    timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
      --memo "$RUN_ID/collect" --projection_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed 2024 \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
      --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic \
      --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
      --collect_states_output "$collected" --result_json "$RUN_ROOT/collect/result.json" > "$RUN_ROOT/logs/collect.log" 2>&1
  fi
  validate_collected "$collected"
  if [[ ! -s "$labels" ]]; then
    mkdir -p "$RUN_ROOT/relabel"
    log "relabeling policy-responsive H30 counterfactual branches"
    timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
      --input "$collected" --output "$labels" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" --dataset hangzhou --num_rows 4 --num_cols 4 \
      --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" --work_dir "$RUN_ROOT/runtime_h30" --sim_horizon 30 --min_action_time 30 --traffic_series_interval 2 \
      --cost_scope global --neighbor-mode responsive --inference_batch_size 32 --bf16 --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
      > "$RUN_ROOT/logs/relabel_h30.log" 2>&1
  fi
  require_file "$labels"
  log "four-GPU counterfactual Stage 2 adaptation"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=4 projection_train/train_projection_bc.py \
    --data_path "$labels" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" --output_dir "$RUN_ROOT/training" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --batch_size 1 --grad_accum_steps 8 --epochs "$EPOCHS" --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --ranking_loss_weight 0.5 --cost_awt_weight 2.0 --cost_aql_weight 1.0 --best_metric val_top1_cost_regret --best_mode min --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
    > "$RUN_ROOT/logs/train.log" 2>&1
  require_file "$RUN_ROOT/training/best.pt"

  log "starting multi-GPU three-seed online evaluation for all saved epochs"
  local tasks=() e s i pids=()
  for ((e=1; e<=EPOCHS; e++)); do require_file "$RUN_ROOT/training/epoch_${e}.pt"; for s in "${SEEDS[@]}"; do tasks+=("$e:$s"); done; done
  for ((i=0; i<${#tasks[@]}; i+=${#GPUS[@]})); do
    pids=()
    for ((j=0; j<${#GPUS[@]} && i+j<${#tasks[@]}; j++)); do IFS=: read -r e s <<< "${tasks[$((i+j))]}"; run_online "$e" "$s" "${GPUS[$j]}" & pids+=("$!"); done
    for p in "${pids[@]}"; do wait "$p"; done
  done
  aggregate_online
  log "completed: $RUN_ROOT"
}

main "$@"
