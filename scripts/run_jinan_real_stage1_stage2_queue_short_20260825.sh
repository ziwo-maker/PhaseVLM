#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
REPO_ROOT="/home/code/LLMSTSC"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/home/data/model/qwen-vl-8B}"
DATA_PATH="${DATA_PATH:-data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_3_4_jinan_real.json}"
STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"
ONLINE_DIR="$RUN_ROOT/online_eval"
LOG_DIR="$RUN_ROOT/logs"

notify_feishu() {
  local title="$1"
  local body="$2"
  local template="${3:-blue}"
  local config="${FEISHU_CONFIG:-/root/.claude/feishu.json}"
  local parsed
  local mode
  local webhook

  [[ -r "$config" ]] || return 0
  parsed="$("$PYTHON" - "$config" <<'PY' 2>/dev/null || true
import json, sys
path = sys.argv[1]
try:
    cfg = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(0)
mode = str(cfg.get("mode", cfg.get("notify_mode", "on")) or "on").lower()
webhook = str(cfg.get("webhook_url", cfg.get("webhook", "")) or "")
if mode in {"off", "none", "disabled", "false", "0"} or not webhook:
    raise SystemExit(0)
print(mode)
print(webhook)
PY
)"
  [[ -n "$parsed" ]] || return 0
  mode="$(printf '%s\n' "$parsed" | sed -n '1p')"
  webhook="$(printf '%s\n' "$parsed" | sed -n '2p')"
  [[ -n "$mode" && -n "$webhook" ]] || return 0

  "$PYTHON" - "$webhook" "$title" "$body" "$template" <<'PY' | \
    curl --silent --show-error --connect-timeout 5 --max-time 10 \
      --output /dev/null --request POST "$webhook" \
      --header 'Content-Type: application/json' --data @- || true
import json, sys
_, webhook, title, body, template = sys.argv
payload = {
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": [{"tag": "markdown", "content": body}],
    },
}
print(json.dumps(payload, ensure_ascii=False))
PY
}

notify_completion() {
  local status="$?"
  local completed_at
  local stage1_state="missing"
  local stage2_state="missing"
  local online_state="missing"
  local template="red"

  completed_at="$(date --iso-8601=seconds)"
  [[ -s "$STAGE1_DIR/best.pt" ]] && stage1_state="checkpoint_ready"
  [[ -s "$STAGE2_DIR/best.pt" ]] && stage2_state="checkpoint_ready"
  [[ -s "$ONLINE_DIR/summary.json" ]] && online_state="summary_ready"
  [[ "$status" -eq 0 ]] && template="green"

  mkdir -p "$LOG_DIR" 2>/dev/null || true
  {
    printf 'exit_status=%s\n' "$status"
    printf 'completed_at=%s\n' "$completed_at"
    printf 'stage1=%s\n' "$stage1_state"
    printf 'stage2=%s\n' "$stage2_state"
    printf 'online=%s\n' "$online_state"
    printf 'run_root=%s\n' "$RUN_ROOT"
  } > "$LOG_DIR/completion.status" 2>/dev/null || true

  notify_feishu \
    "Jinan Real Queue-Short Experiment Finished" \
    "Exit status: $status\nStage 1: $stage1_state\nStage 2: $stage2_state\nOnline eval: $online_state\nRun: $RUN_ROOT" \
    "$template"

  trap - EXIT
  exit "$status"
}

trap notify_completion EXIT

for required_path in "$PYTHON" "$TORCHRUN" "$MODEL_PATH" "$REPO_ROOT/$DATA_PATH" "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done

if [[ -e "$RUN_ROOT" ]]; then
  existing_non_launcher="$(
    find "$RUN_ROOT" -mindepth 1 \
      ! -path "$RUN_ROOT/logs" \
      ! -path "$RUN_ROOT/logs/launcher.log" \
      -print -quit 2>/dev/null
  )"
  if [[ -n "$existing_non_launcher" ]]; then
    echo "RUN_ROOT already contains experiment artifacts: $RUN_ROOT" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR" "$STAGE1_DIR" "$STAGE2_DIR" "$ONLINE_DIR"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

{
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repo_root=%s\n' "$REPO_ROOT"
  printf 'python=%s\n' "$PYTHON"
  printf 'torchrun=%s\n' "$TORCHRUN"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'data_path=%s\n' "$DATA_PATH"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'stage1_epochs=15\n'
  printf 'stage2_epochs=15\n'
  printf 'online_eval_seeds=2024,42,123\n'
} | tee "$LOG_DIR/run_config.txt"

"$TORCHRUN" --standalone --nproc_per_node=3 train_align.py \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --output_dir "$STAGE1_DIR" \
  --decision_only \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix \
  --temperature 0.07 \
  --batch_size 4 --grad_accum_steps 2 \
  --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --epochs 15 --num_workers 4 --val_ratio 0.1 \
  --eval_max_samples 2000 --log_every_steps 50 --save_every_epochs 5 \
  --seed 2024 --bf16 \
  2>&1 | tee "$LOG_DIR/stage1.log"

ALIGN_CKPT="$STAGE1_DIR/best.pt"
if [[ ! -s "$ALIGN_CKPT" ]]; then
  echo "Stage 1 completed without a valid checkpoint: $ALIGN_CKPT" >&2
  exit 1
fi

"$TORCHRUN" --standalone --nproc_per_node=3 projection_train/train_projection_bc.py \
  --data_path "$DATA_PATH" \
  --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" \
  --output_dir "$STAGE2_DIR" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode queue_wait_short \
  --batch_size 4 --grad_accum_steps 4 \
  --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --val_ratio 0.1 --split_strategy blocked --num_workers 4 \
  --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
  --best_metric val_macro_f1 --best_mode auto --best_min_delta 0.0 \
  --log_interval 20 --save_every 1 --seed 2024 --bf16 \
  --ddp_timeout_seconds 1800 \
  2>&1 | tee "$LOG_DIR/stage2.log"

STAGE2_CKPT="$STAGE2_DIR/best.pt"
if [[ ! -s "$STAGE2_CKPT" ]]; then
  echo "Stage 2 completed without a valid checkpoint: $STAGE2_CKPT" >&2
  exit 1
fi

run_online_eval() {
  local seed="$1"
  local gpu="$2"
  local seed_dir="$ONLINE_DIR/seed_${seed}"
  mkdir -p "$seed_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "jinan_real_queue_short_stage1_$(basename "$RUN_ROOT")_seed${seed}" \
      --projection_ckpt "$STAGE2_CKPT" \
      --llm_path "$MODEL_PATH" \
      --dataset jinan \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts 3600 \
      --seed "$seed" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --bf16 \
      --result_json "$seed_dir/result.json" \
      2>&1 | tee "$LOG_DIR/online_seed_${seed}.log"
  )
}

run_online_eval 2024 0 &
pid_2024="$!"
run_online_eval 42 1 &
pid_42="$!"
run_online_eval 123 2 &
pid_123="$!"

eval_status=0
wait "$pid_2024" || eval_status=1
wait "$pid_42" || eval_status=1
wait "$pid_123" || eval_status=1
if [[ "$eval_status" -ne 0 ]]; then
  echo "At least one online evaluation failed; see $LOG_DIR/online_seed_*.log" >&2
  exit "$eval_status"
fi

"$PYTHON" - "$ONLINE_DIR" "$ONLINE_DIR/summary.json" <<'PY'
import json, pathlib, sys
online_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
summary = {"results": []}
for result_path in sorted(online_dir.glob("seed_*/result.json")):
    try:
        payload = json.load(open(result_path, encoding="utf-8"))
    except Exception as exc:
        summary["results"].append({"path": str(result_path), "error": str(exc)})
        continue
    args = payload.get("args") or {}
    results = payload.get("results") or {}
    if isinstance(results, list) and results:
        metrics = results[-1]
    else:
        metrics = results
    summary["results"].append({
        "seed": args.get("seed"),
        "path": str(result_path),
        "work_dir": payload.get("work_dir", ""),
        "metrics": metrics,
    })
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/run_config.txt"
