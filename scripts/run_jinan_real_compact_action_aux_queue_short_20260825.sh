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
STAGE1_DATA="${STAGE1_DATA:-data/processed/jinan_real_stage1_compact_highmargin_m05_queue_wait_rule_gap2_p60_20260825.jsonl}"
STAGE2_DATA="${STAGE2_DATA:-data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_3_4_jinan_real.json}"
ACTION_AUX_WEIGHTS="${ACTION_AUX_WEIGHTS:-0.0 0.3}"
LOG_DIR="$RUN_ROOT/logs"

notify_feishu() {
  local title="$1"
  local body="$2"
  local template="${3:-blue}"
  local config
  local parsed
  local webhook

  for config in "${FEISHU_CONFIG:-}" "/root/.claude/feishu.json" "$HOME/.codex/feishu.json"; do
    [[ -n "$config" && -r "$config" ]] && break
  done
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
print(webhook)
PY
)"
  webhook="$(printf '%s\n' "$parsed" | sed -n '1p')"
  [[ -n "$webhook" ]] || return 0

  "$PYTHON" - "$title" "$body" "$template" <<'PY' | \
    curl --silent --show-error --connect-timeout 5 --max-time 10 \
      --output /dev/null --request POST "$webhook" \
      --header 'Content-Type: application/json' --data @- || true
import json, sys
_, title, body, template = sys.argv
print(json.dumps({
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": [{"tag": "markdown", "content": body}],
    },
}, ensure_ascii=False))
PY
}

notify_completion() {
  local status="$?"
  local completed_at
  local template="red"
  completed_at="$(date --iso-8601=seconds)"
  [[ "$status" -eq 0 ]] && template="green"
  mkdir -p "$LOG_DIR" 2>/dev/null || true
  {
    printf 'exit_status=%s\n' "$status"
    printf 'completed_at=%s\n' "$completed_at"
    printf 'run_root=%s\n' "$RUN_ROOT"
    for arm_dir in "$RUN_ROOT"/compact_*; do
      [[ -d "$arm_dir" ]] || continue
      printf '%s_stage1=%s\n' "$(basename "$arm_dir")" "$([[ -s "$arm_dir/stage1/best.pt" ]] && echo checkpoint_ready || echo missing)"
      printf '%s_stage2=%s\n' "$(basename "$arm_dir")" "$([[ -s "$arm_dir/stage2/checkpoints/best.pt" ]] && echo checkpoint_ready || echo missing)"
      printf '%s_online=%s\n' "$(basename "$arm_dir")" "$([[ -s "$arm_dir/online_eval/summary.json" ]] && echo summary_ready || echo missing)"
    done
  } > "$LOG_DIR/completion.status" 2>/dev/null || true
  notify_feishu \
    "Jinan Real Compact Action-Aux Experiment Finished" \
    "Exit status: $status\nRun: $RUN_ROOT\nSee: $LOG_DIR/completion.status" \
    "$template"
  trap - EXIT
  exit "$status"
}

trap notify_completion EXIT

for required_path in "$PYTHON" "$TORCHRUN" "$MODEL_PATH" "$REPO_ROOT/$STAGE1_DATA" "$REPO_ROOT/$STAGE2_DATA" "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"; do
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

mkdir -p "$LOG_DIR"
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
  printf 'stage1_data=%s\n' "$STAGE1_DATA"
  printf 'stage2_data=%s\n' "$STAGE2_DATA"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'capacity=prefix_tokens8_hidden_dim64\n'
  printf 'action_aux_weights=%s\n' "$ACTION_AUX_WEIGHTS"
} | tee "$LOG_DIR/run_config.txt"

run_online_eval() {
  local arm_name="$1"
  local stage2_ckpt="$2"
  local online_dir="$3"
  local seed="$4"
  local gpu="$5"
  local seed_dir="$online_dir/seed_${seed}"
  mkdir -p "$seed_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "jinan_real_${arm_name}_$(basename "$RUN_ROOT")_seed${seed}" \
      --projection_ckpt "$stage2_ckpt" \
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
      2>&1 | tee "$LOG_DIR/${arm_name}_online_seed_${seed}.log"
  )
}

summarize_online() {
  local online_dir="$1"
  "$PYTHON" - "$online_dir" "$online_dir/summary.json" <<'PY'
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
    metrics = results[-1] if isinstance(results, list) and results else results
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
}

run_arm() {
  local arm_name="$1"
  local action_aux_weight="$2"
  local arm_root="$RUN_ROOT/$arm_name"
  local stage1_dir="$arm_root/stage1"
  local stage2_dir="$arm_root/stage2/checkpoints"
  local online_dir="$arm_root/online_eval"

  mkdir -p "$stage1_dir" "$stage2_dir" "$online_dir"
  printf '\n===== arm=%s action_aux_weight=%s started_at=%s =====\n' "$arm_name" "$action_aux_weight" "$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/run_config.txt"

  "$TORCHRUN" --standalone --nproc_per_node=3 train_align.py \
    --data_path "$STAGE1_DATA" \
    --model_path "$MODEL_PATH" \
    --output_dir "$stage1_dir" \
    --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix \
    --temperature 0.07 \
    --action_aux_weight "$action_aux_weight" \
    --batch_size 4 --grad_accum_steps 2 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 20 --num_workers 4 --val_ratio 0.1 \
    --eval_max_samples 2000 --log_every_steps 50 --save_every_epochs 5 \
    --seed 2024 --bf16 \
    2>&1 | tee "$LOG_DIR/${arm_name}_stage1.log"

  local align_ckpt="$stage1_dir/best.pt"
  if [[ ! -s "$align_ckpt" ]]; then
    echo "Stage 1 completed without a valid checkpoint: $align_ckpt" >&2
    exit 1
  fi

  "$TORCHRUN" --standalone --nproc_per_node=3 projection_train/train_projection_bc.py \
    --data_path "$STAGE2_DATA" \
    --llm_path "$MODEL_PATH" --align_ckpt "$align_ckpt" \
    --output_dir "$stage2_dir" \
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
    2>&1 | tee "$LOG_DIR/${arm_name}_stage2.log"

  local stage2_ckpt="$stage2_dir/best.pt"
  if [[ ! -s "$stage2_ckpt" ]]; then
    echo "Stage 2 completed without a valid checkpoint: $stage2_ckpt" >&2
    exit 1
  fi

  run_online_eval "$arm_name" "$stage2_ckpt" "$online_dir" 2024 0 &
  local pid_2024="$!"
  run_online_eval "$arm_name" "$stage2_ckpt" "$online_dir" 42 1 &
  local pid_42="$!"
  run_online_eval "$arm_name" "$stage2_ckpt" "$online_dir" 123 2 &
  local pid_123="$!"

  local eval_status=0
  wait "$pid_2024" || eval_status=1
  wait "$pid_42" || eval_status=1
  wait "$pid_123" || eval_status=1
  if [[ "$eval_status" -ne 0 ]]; then
    echo "At least one online evaluation failed for $arm_name; see $LOG_DIR/${arm_name}_online_seed_*.log" >&2
    exit "$eval_status"
  fi
  summarize_online "$online_dir" | tee "$LOG_DIR/${arm_name}_online_summary.log"
  printf 'arm=%s finished_at=%s\n' "$arm_name" "$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/run_config.txt"
}

weight_to_arm_name() {
  local weight="$1"
  if [[ "$weight" == "0" || "$weight" == "0.0" || "$weight" == "0.00" ]]; then
    printf 'compact_clip\n'
    return 0
  fi
  printf 'compact_action_aux%s\n' "$(printf '%s' "$weight" | tr '.' 'p' | tr -cd '0-9p')"
}

for action_aux_weight in $ACTION_AUX_WEIGHTS; do
  run_arm "$(weight_to_arm_name "$action_aux_weight")" "$action_aux_weight"
done

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/run_config.txt"
