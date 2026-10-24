#!/usr/bin/env bash
# jinan_real_2500 held-out — fresh Stage 1 -> Stage 2 -> multi-seed online tests.
#
# Source training data excludes jinan_real_2500, then final online evaluation is
# run on anon_3_4_jinan_real_2500.json with multiple seeds and two checkpoints
# from the trained Stage 2 model: best.pt and epoch_15.pt.
#
# This is a fresh run wrapper with a unique RUN_ROOT by default.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
RUN_ID="${RUN_ID:-jinan2500_stage1_stage2_online_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/$RUN_ID}"
TRAIN_GPUS_RAW="${TRAIN_GPUS:-0,1,2,3}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
EVAL_GPUS_RAW="${EVAL_GPUS:-0 1 2}"
EVAL_SEEDS_RAW="${EVAL_SEEDS:-2024 42 123}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
FEISHU_SCRIPT="${FEISHU_SCRIPT:-$REPO_ROOT/scripts/feishu_push.py}"

read -r -a EVAL_GPUS <<< "$EVAL_GPUS_RAW"
read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_RAW"
read -r -a TRAIN_GPU_LIST <<< "${TRAIN_GPUS_RAW//,/ }"

STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2"
ONLINE_DIR="$RUN_ROOT/online_eval"
LOG_DIR="$RUN_ROOT/logs"
MANIFEST_JSON="$RUN_ROOT/manifest.json"
SUMMARY_JSON="$ONLINE_DIR/summary.json"
CURRENT_PHASE="$RUN_ROOT/current_phase.txt"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"

DATA1="$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DATA2="$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
TRAFFIC_FILE="anon_3_4_jinan_real_2500.json"

timestamp() { date --iso-8601=seconds; }

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_file() {
  [[ -s "$1" ]] || { log "Missing required file: $1"; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "Missing required directory: $1"; exit 1; }
}

notify() {
  [[ -s "$FEISHU_SCRIPT" ]] || return 0
  timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" >/dev/null 2>&1 || true
}

validate_gpus() {
  local rows gpu used
  mapfile -t rows < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  for gpu in "${TRAIN_GPU_LIST[@]}" "${EVAL_GPUS[@]}"; do
    used="$(printf '%s\n' "${rows[@]}" | awk -F, -v target="$gpu" '$1 + 0 == target {gsub(/ /, "", $2); print $2}')"
    [[ "$used" =~ ^[0-9]+$ ]] || { log "Could not read GPU $gpu memory usage"; exit 1; }
    (( used < 1000 )) || { log "GPU $gpu is not idle: ${used} MiB used"; exit 1; }
  done
}

write_manifest() {
  "$PYTHON_BIN" - "$MANIFEST_JSON" "$RUN_ID" "$MODEL_PATH" "$RUN_COUNTS" "$TRAIN_GPUS_RAW" "$EVAL_GPUS_RAW" "${EVAL_SEEDS[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

out, run_id, model_path, run_counts_raw, train_gpus, eval_gpus_raw, *seed_raw = sys.argv[1:]
payload = {
    "schema": "jinan2500_stage1_stage2_online_full_v1",
    "run_id": run_id,
    "created_at": datetime.now().astimezone().isoformat(),
    "model_path": model_path,
    "run_counts": int(run_counts_raw),
    "stage1_data": [
        "/home/code/LLMSTSC/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
        "/home/code/LLMSTSC/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
    ],
    "stage2_data": [
        "/home/code/LLMSTSC/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
        "/home/code/LLMSTSC/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
    ],
    "train_gpus": train_gpus,
    "eval_gpus": [int(x) for x in eval_gpus_raw.split()],
    "eval_seeds": [int(x) for x in seed_raw],
    "prompt_mode": "queue_wait_short",
    "max_length": 64,
    "traffic_file": "anon_3_4_jinan_real_2500.json",
    "online_checkpoints": ["best.pt", "epoch_15.pt"],
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_stage1() {
  log "Stage 1: alignment"
  mkdir -p "$STAGE1_DIR" "$LOG_DIR"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_RAW" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" train_align.py \
    --data_path "$DATA1" "$DATA2" \
    --model_path "$MODEL_PATH" \
    --output_dir "$STAGE1_DIR" \
    --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix \
    --temperature 0.07 \
    --batch_size 2 --grad_accum_steps 2 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 \
    2>&1 | tee "$LOG_DIR/stage1.log"
  require_file "$STAGE1_DIR/best.pt"
}

run_stage2() {
  log "Stage 2: BC policy"
  mkdir -p "$STAGE2_DIR" "$LOG_DIR"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_RAW" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
    --data_path "$DATA1" "$DATA2" \
    --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_DIR/best.pt" \
    --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 64 \
    --prompt_mode queue_wait_short \
    --batch_size 2 --grad_accum_steps 8 --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 \
    --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --best_metric val_top1_cost_regret --best_mode min --best_min_delta 0.0 \
    --log_interval 20 --save_every 1 --seed 2024 --bf16 \
    --checkpoint_slim --ddp_timeout_seconds 1800 \
    2>&1 | tee "$LOG_DIR/stage2.log"
  require_file "$STAGE2_DIR/best.pt"
  require_file "$STAGE2_DIR/epoch_15.pt"
}

run_online_checkpoint() {
  local tag="$1"
  local ckpt="$2"
  local gpu seed seed_dir
  log "Online tests for $tag"
  for seed in "${EVAL_SEEDS[@]}"; do
    seed_dir="$ONLINE_DIR/$tag/seed_${seed}"
    mkdir -p "$seed_dir"
  done

  for idx in "${!EVAL_SEEDS[@]}"; do
    seed="${EVAL_SEEDS[$idx]}"
    gpu="${EVAL_GPUS[$idx]}"
    (
      mkdir -p "$ONLINE_DIR/$tag/seed_${seed}"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
        --memo "jinan2500_stage1_stage2/${tag}/seed_${seed}" \
        --projection_ckpt "$ckpt" \
        --llm_path "$MODEL_PATH" \
        --dataset jinan \
        --traffic_file "$TRAFFIC_FILE" \
        --run_counts "$RUN_COUNTS" \
        --num_rounds 1 \
        --seed "$seed" \
        --prompt_mode queue_wait_short \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
        --prefix_tokens 32 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 \
        --result_json "$ONLINE_DIR/$tag/seed_${seed}/result.json" \
        2>&1 | tee "$LOG_DIR/online_${tag}_seed_${seed}.log"
    ) &
  done

  local rc=0 pid
  for pid in $(jobs -p); do
    wait "$pid" || rc=1
  done
  return "$rc"
}

write_summary() {
  "$PYTHON_BIN" - "$ONLINE_DIR" "$SUMMARY_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

online_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
summary = {"results": []}
for result_path in sorted(online_dir.glob("*/*/result.json")):
    try:
        payload = json.load(open(result_path, encoding="utf-8"))
        args = payload.get("args") or {}
        res = payload.get("results") or {}
        summary["results"].append({
            "tag": result_path.parts[-3],
            "seed": args.get("seed"),
            "path": str(result_path),
            "metrics": res,
        })
    except Exception as exc:
        summary["results"].append({"path": str(result_path), "error": str(exc)})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

main() {
  cd "$REPO_ROOT"
  mkdir -p "$RUN_ROOT" "$LOG_DIR" "$ONLINE_DIR"
  printf 'state=running\nstarted_at=%s\nrun_id=%s\nrun_root=%s\n' \
    "$(timestamp)" "$RUN_ID" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
  printf 'phase=preflight\n' > "$CURRENT_PHASE"

  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN"
  require_dir "$MODEL_PATH"
  require_file "$REPO_ROOT/train_align.py"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  require_file "$DATA1"
  require_file "$DATA2"
  require_file "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"
  validate_gpus
  write_manifest
  notify "Jinan 2500 stage1+2+online launched: run_root=$RUN_ROOT; seeds=${EVAL_SEEDS[*]}; checkpoints=best,epoch15"

  printf 'phase=stage1\n' > "$CURRENT_PHASE"
  run_stage1

  printf 'phase=stage2\n' > "$CURRENT_PHASE"
  run_stage2

  printf 'phase=online_best\n' > "$CURRENT_PHASE"
  run_online_checkpoint "best" "$STAGE2_DIR/best.pt"

  printf 'phase=online_epoch15\n' > "$CURRENT_PHASE"
  run_online_checkpoint "epoch15" "$STAGE2_DIR/epoch_15.pt"

  write_summary
  printf 'state=completed\ncompleted_at=%s\nsummary_json=%s\n' \
    "$(timestamp)" "$SUMMARY_JSON" > "$CAMPAIGN_STATUS"
  notify "Jinan 2500 stage1+2+online completed: run_root=$RUN_ROOT; summary=$SUMMARY_JSON"
}

trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then printf "state=failed\nfailed_at=%s\nexit_status=%s\n" "$(timestamp)" "$rc" > "$CAMPAIGN_STATUS"; notify "Jinan 2500 stage1+2+online failed: exit=$rc; run_root=$RUN_ROOT"; fi' EXIT

main "$@"
