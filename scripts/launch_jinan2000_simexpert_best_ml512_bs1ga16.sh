#!/usr/bin/env bash
# Train a same-flow Jinan-2000 policy from the selected SimExpert dataset.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_ID="${RUN_ID:-jinan2000_simexpert_best_ml512_bs1ga16_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/$RUN_ID}"

SOURCE_ROOT="${SOURCE_ROOT:-/mnt/data/LLMSTSCrecords/records/sim_expert_validation_search/anon_3_4_jinan_real_2000/best}"
TRAIN_DATA="$SOURCE_ROOT/align_train.cleaned.jsonl"
VAL_DATA="$SOURCE_ROOT/align_val.cleaned.jsonl"
STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2"
LOG_DIR="$RUN_ROOT/logs"
STATUS_FILE="$RUN_ROOT/status.txt"
MANIFEST_PATH="$RUN_ROOT/manifest.json"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_file() { [[ -s "$1" ]] || { log "Required file is missing or empty: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "Required directory is missing: $1" >&2; exit 1; }; }

record_failure() {
  local code=$?
  if [[ "$code" -ne 0 ]]; then
    printf 'failed exit_code=%s at=%s\n' "$code" "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  fi
}
trap record_failure EXIT

write_manifest() {
  "$PYTHON_BIN" - "$MANIFEST_PATH" "$RUN_ID" "$RUN_ROOT" "$MODEL_PATH" "$GPU_LIST" "$SOURCE_ROOT" "$TRAIN_DATA" "$VAL_DATA" <<'PY'
import json
import sys
from datetime import datetime, timezone

(
    path, run_id, run_root, model_path, gpu_list, source_root, train_data, val_data
) = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "run_root": run_root,
    "model_path": model_path,
    "gpus": gpu_list,
    "source": {
        "search_root": source_root,
        "traffic_file": "anon_3_4_jinan_real_2000.json",
        "selection_objective": "weighted_minmax_full_run_metrics",
        "expert_weights": {"att": 1.0, "aql": 1.0, "awt": 2.0, "spillback": 0.2},
        "selected_full_run_metrics": {"ATT": 265.2165, "AQL": 103.1667, "AWT": 36.0619},
        "train_data": train_data,
        "val_data": val_data,
    },
    "architecture": {
        "seq_len": 16, "image_size": 224, "periodicity": 16,
        "input_dim": 8, "hidden_dim": 64, "prefix_tokens": 8,
        "prefix_mode": "prefix", "max_length": 512,
    },
    "stage1": {
        "world_size": 4, "batch_size_per_rank": 1, "grad_accum_steps": 4,
        "effective_global_batch": 16, "epochs": 15, "seed": 2024,
    },
    "stage2": {
        "world_size": 4, "batch_size_per_rank": 1, "grad_accum_steps": 16,
        "effective_global_batch": 64, "epochs": 15, "seed": 2024,
        "target_mode": "mixed", "best_metric": "val_top1_cost_regret",
    },
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN_BIN"
  require_file "$REPO_ROOT/train_align.py"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  require_dir "$MODEL_PATH"
  require_file "$TRAIN_DATA"
  require_file "$VAL_DATA"
  if [[ -e "$RUN_ROOT" ]]; then
    log "Refusing to reuse existing run directory: $RUN_ROOT" >&2
    exit 1
  fi

  mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$LOG_DIR"
  write_manifest
  printf 'running stage=preflight at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  export PYTHONPATH="$REPO_ROOT"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_ASYNC_ERROR_HANDLING=1

  "$PYTHON_BIN" - "$TRAIN_DATA" "$VAL_DATA" "$MODEL_PATH" > "$LOG_DIR/preflight.json" <<'PY'
import json
import sys
from pathlib import Path

train_path, val_path, model_path = map(Path, sys.argv[1:])
for path in (train_path, val_path):
    count = 0
    candidate_record = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            record = json.loads(line)
            if record.get("expert_action") and record.get("candidate_action_metrics"):
                candidate_record = record
                break
    if candidate_record is None:
        raise SystemExit(f"{path} has no labeled candidate-metric record")
    print(json.dumps({
        "path": str(path), "rows_scanned_until_valid": count,
        "expert_action": candidate_record["expert_action"],
        "candidate_actions": sorted(candidate_record["candidate_action_metrics"]),
    }))
print(json.dumps({"model_config_present": (model_path / "config.json").is_file()}))
PY

  log "Stage 1 start: four-GPU DDP, max_length=512"
  printf 'running stage=stage1 at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_align.py \
    --data_path "$TRAIN_DATA" \
    --model_path "$MODEL_PATH" --output_dir "$STAGE1_DIR" --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 --temperature 0.07 \
    --batch_size 1 --grad_accum_steps 4 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 20 --save_every_epochs 1 --seed 2024 --bf16 \
    > "$LOG_DIR/stage1.log" 2>&1
  require_file "$STAGE1_DIR/best.pt"

  log "Stage 2 start: four-GPU DDP, max_length=512, per-rank batch=1, grad_accum=16"
  printf 'running stage=stage2 at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" projection_train/train_projection_bc.py \
    --train_data_path "$TRAIN_DATA" --val_data_path "$VAL_DATA" \
    --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_DIR/best.pt" --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 --prompt_mode queue_wait_short --action_eval_chunk_size 1 \
    --batch_size 1 --grad_accum_steps 16 --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 --num_workers 4 --seed 2024 --bf16 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 \
    --ranking_loss_weight 0.15 --base_policy_kl_weight 0.02 \
    --cost_att_weight 1.0 --cost_aql_weight 1.0 --cost_awt_weight 2.0 --cost_spillback_weight 0.2 \
    --best_metric val_top1_cost_regret --best_mode min --save_every 1 --checkpoint_slim --log_interval 20 \
    --ddp_timeout_seconds 1800 \
    > "$LOG_DIR/stage2.log" 2>&1
  require_file "$STAGE2_DIR/best.pt"
  printf 'completed at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  log "Stage 1 and Stage 2 completed"
}

main "$@"
