#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/home/data/model/qwen-vl-8B}"
MODEL_LABEL_DEFAULT="${MODEL_PATH%/}"
MODEL_LABEL_DEFAULT="${MODEL_LABEL_DEFAULT##*/}"
MODEL_LABEL_DEFAULT="${MODEL_LABEL_DEFAULT//[^A-Za-z0-9_.-]/_}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_LABEL_DEFAULT:-model}}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_${MODEL_LABEL}_$(date +%Y%m%d_%H%M%S)}"
ONLINE_MEMO="${ONLINE_MEMO:-online_tests/repro_jinan3_qwshort_targeted_20260805_epoch15/${MODEL_LABEL}/epoch_15}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS// /,}"
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
TRAIN_NPROC="${TRAIN_NPROC:-${#TRAIN_GPU_ARRAY[@]}}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_SEED="${EVAL_SEED:-2024}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-2400}"
REUSE_STAGE1_CKPT="${REUSE_STAGE1_CKPT:-}"

MODEL_PATH_HINT="${MODEL_PATH,,}"
MODEL_FAMILY_HINT="other"
if [[ "$MODEL_PATH_HINT" == *"qwen3.5"* || "$MODEL_PATH_HINT" == *"qwen3_5"* ]]; then
  MODEL_FAMILY_HINT="qwen3_5"
fi

STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-2}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-4}"
STAGE2_BATCH_SIZE_DEFAULT=4
STAGE2_GRAD_ACCUM_STEPS_DEFAULT=4
STAGE2_ACTION_EVAL_CHUNK_SIZE_DEFAULT=0
if [[ "$MODEL_FAMILY_HINT" == "qwen3_5" ]]; then
  STAGE2_BATCH_SIZE_DEFAULT=1
  STAGE2_GRAD_ACCUM_STEPS_DEFAULT=16
  STAGE2_ACTION_EVAL_CHUNK_SIZE_DEFAULT=1
fi
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-$STAGE2_BATCH_SIZE_DEFAULT}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-$STAGE2_GRAD_ACCUM_STEPS_DEFAULT}"
STAGE2_ACTION_EVAL_CHUNK_SIZE="${STAGE2_ACTION_EVAL_CHUNK_SIZE:-$STAGE2_ACTION_EVAL_CHUNK_SIZE_DEFAULT}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"

DATA_PATHS=(
  "$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
  "$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
  "$REPO_ROOT/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl"
)
TRAFFIC_FILE="anon_3_4_jinan_real.json"
STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2/checkpoints"
ONLINE_DIR="$RUN_ROOT/online_eval/seed_${EVAL_SEED}"
LOG_DIR="$RUN_ROOT/logs"
SUMMARY_JSON="$RUN_ROOT/online_eval/summary.json"

notify_feishu() {
  local title="$1"
  local body="$2"
  local template="${3:-blue}"
  local config
  local webhook

  for config in "${FEISHU_CONFIG:-}" "$HOME/.claude/feishu.json" "$HOME/.codex/feishu.json" "/root/.claude/feishu.json"; do
    [[ -n "$config" && -r "$config" ]] && break
  done
  [[ -r "$config" ]] || return 0

  webhook="$("$PYTHON" - "$config" <<'PY' 2>/dev/null || true
import json
import sys
try:
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
mode = str(cfg.get("mode", cfg.get("notify_mode", "on")) or "on").lower()
webhook = str(cfg.get("webhook_url", cfg.get("webhook", "")) or "")
if mode in {"off", "none", "disabled", "false", "0"} or not webhook:
    raise SystemExit(0)
print(webhook)
PY
)"
  [[ -n "$webhook" ]] || return 0

  "$PYTHON" - "$title" "$body" "$template" <<'PY' | \
    curl --silent --show-error --connect-timeout 5 --max-time 10 \
      --output /dev/null --request POST "$webhook" \
      --header 'Content-Type: application/json' --data @- || true
import json
import sys
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

preflight_model() {
  "$PYTHON" - "$MODEL_PATH" <<'PY'
import sys
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

model_path = sys.argv[1]
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model_type = str(getattr(config, "model_type", "") or "").lower()
architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
hint = " ".join([model_type, *[item.lower() for item in architectures], model_path.lower()])

model_cls = None
if "qwen3_5" in hint or "qwen3.5" in hint:
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        model_cls = Qwen3_5ForConditionalGeneration
    except Exception:
        model_cls = None
elif "qwen3_vl" in hint or "qwen3-vl" in hint or "qwen-vl" in hint:
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except Exception:
        model_cls = None
if model_cls is None:
    try:
        from transformers import AutoModelForImageTextToText as model_cls
    except Exception:
        from transformers import AutoModelForCausalLM as model_cls

vision_config = getattr(config, "vision_config", None)
if vision_config is None:
    raise RuntimeError(f"{model_path} does not expose config.vision_config; a VLM backbone is required.")

print(
    "[model preflight] "
    f"path={model_path} "
    f"model_type={getattr(config, 'model_type', '')} "
    f"architectures={architectures} "
    f"processor={processor.__class__.__name__} "
    f"tokenizer={tokenizer.__class__.__name__} "
    f"selected_class={model_cls.__name__} "
    f"vision_hidden={getattr(vision_config, 'hidden_size', '')} "
    f"vision_out={getattr(vision_config, 'out_hidden_size', '')}"
)
PY
}

latest_status() {
  {
    printf 'Run: %s\n' "$RUN_ROOT"
    printf 'Time: %s\n' "$(date --iso-8601=seconds)"
    [[ -s "$STAGE1_DIR/best.pt" ]] && printf 'Stage 1: best.pt ready\n' || printf 'Stage 1: pending/running\n'
    [[ -s "$STAGE2_DIR/epoch_15.pt" ]] && printf 'Stage 2: epoch_15.pt ready\n' || printf 'Stage 2: pending/running\n'
    if [[ -s "$STAGE2_DIR/metrics.jsonl" ]]; then
      printf 'Latest Stage 2 metric: %s\n' "$(tail -n 1 "$STAGE2_DIR/metrics.jsonl")"
    fi
    if [[ -s "$ONLINE_DIR/result.json" ]]; then
      printf 'Online result: %s\n' "$(tr '\n' ' ' < "$ONLINE_DIR/result.json" | cut -c 1-500)"
    fi
  } | sed -n '1,20p'
}

monitor_progress() {
  while true; do
    sleep "$PROGRESS_INTERVAL_SECONDS"
    notify_feishu "08-05 Jinan Repro Progress" "$(latest_status)" "blue"
  done
}

notify_completion() {
  local status="$?"
  local template="red"
  [[ "$status" -eq 0 ]] && template="green"
  {
    printf 'exit_status=%s\n' "$status"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'stage1_best=%s\n' "$STAGE1_DIR/best.pt"
    printf 'stage2_epoch15=%s\n' "$STAGE2_DIR/epoch_15.pt"
    printf 'online_result=%s\n' "$ONLINE_DIR/result.json"
    printf 'summary_json=%s\n' "$SUMMARY_JSON"
  } > "$LOG_DIR/completion.status" 2>/dev/null || true
  [[ -n "${MONITOR_PID:-}" ]] && kill "$MONITOR_PID" 2>/dev/null || true
  notify_feishu "08-05 Jinan Repro Finished" "Exit status: $status\n$(latest_status)" "$template"
  trap - EXIT
  exit "$status"
}

trap notify_completion EXIT

for path in "$PYTHON" "$TORCHRUN" "$MODEL_PATH" "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE" "${DATA_PATHS[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 1
  fi
done

preflight_model

if [[ -e "$RUN_ROOT" ]]; then
  existing="$(find "$RUN_ROOT" -mindepth 1 -print -quit 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    echo "RUN_ROOT already exists and is not empty: $RUN_ROOT" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR" "$STAGE1_DIR" "$STAGE2_DIR" "$ONLINE_DIR"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source_archive=/mnt/data/LLMSTSCrecords/records/online_tests/jinan3_qwshort_targeted_20260805_epoch9_epoch15/epoch_15/anon_3_4_jinan_real.json_08_05_12_35_29\n'
  printf 'python=%s\n' "$PYTHON"
  printf 'torchrun=%s\n' "$TORCHRUN"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'model_label=%s\n' "$MODEL_LABEL"
  printf 'data_paths=%s\n' "${DATA_PATHS[*]}"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'train_gpus=%s\n' "$TRAIN_GPUS"
  printf 'train_nproc=%s\n' "$TRAIN_NPROC"
  printf 'eval_gpu=%s\n' "$EVAL_GPU"
  printf 'eval_seed=%s\n' "$EVAL_SEED"
  printf 'reuse_stage1_ckpt=%s\n' "$REUSE_STAGE1_CKPT"
  printf 'model_family_hint=%s\n' "$MODEL_FAMILY_HINT"
  printf 'stage1_batch_size=%s\n' "$STAGE1_BATCH_SIZE"
  printf 'stage1_grad_accum_steps=%s\n' "$STAGE1_GRAD_ACCUM_STEPS"
  printf 'stage1_num_workers=%s\n' "$STAGE1_NUM_WORKERS"
  printf 'stage2_batch_size=%s\n' "$STAGE2_BATCH_SIZE"
  printf 'stage2_grad_accum_steps=%s\n' "$STAGE2_GRAD_ACCUM_STEPS"
  printf 'stage2_action_eval_chunk_size=%s\n' "$STAGE2_ACTION_EVAL_CHUNK_SIZE"
  printf 'stage2_num_workers=%s\n' "$STAGE2_NUM_WORKERS"
  printf 'pytorch_cuda_alloc_conf=%s\n' "$PYTORCH_CUDA_ALLOC_CONF"
  printf 'stage1_config=pt8_hd64_temp0.05_epochs15_decision_only\n'
  printf 'stage2_config=pt8_hd64_queue_wait_short_epochs15_targeted_resample_fitted_to_20260805_archive\n'
} | tee "$LOG_DIR/run.info"

monitor_progress &
MONITOR_PID="$!"

if [[ -n "$REUSE_STAGE1_CKPT" ]]; then
  if [[ ! -s "$REUSE_STAGE1_CKPT" ]]; then
    echo "REUSE_STAGE1_CKPT is not a readable checkpoint: $REUSE_STAGE1_CKPT" >&2
    exit 1
  fi
  cp -p "$REUSE_STAGE1_CKPT" "$STAGE1_DIR/best.pt"
  {
    printf 'status=reused\n'
    printf 'source_checkpoint=%s\n' "$REUSE_STAGE1_CKPT"
    printf 'copied_to=%s\n' "$STAGE1_DIR/best.pt"
    printf 'copied_at=%s\n' "$(date --iso-8601=seconds)"
  } > "$STAGE1_DIR/reused_stage1.info"
else
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" train_align.py \
  --data_path "${DATA_PATHS[@]}" \
  --model_path "$MODEL_PATH" \
  --output_dir "$STAGE1_DIR" \
  --decision_only \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix \
  --temperature 0.05 \
  --batch_size "$STAGE1_BATCH_SIZE" --grad_accum_steps "$STAGE1_GRAD_ACCUM_STEPS" \
  --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --epochs 15 --num_workers "$STAGE1_NUM_WORKERS" --val_ratio 0.1 \
  --eval_max_samples 1000 --log_every_steps 20 --save_every_epochs 5 \
  --seed 2024 --bf16 \
  > "$LOG_DIR/stage1.log" 2>&1
fi

if [[ ! -s "$STAGE1_DIR/best.pt" ]]; then
  echo "Stage 1 completed without a valid checkpoint: $STAGE1_DIR/best.pt" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" projection_train/train_projection_bc.py \
  --data_path "${DATA_PATHS[@]}" \
  --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_DIR/best.pt" \
  --output_dir "$STAGE2_DIR" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --prompt_mode queue_wait_short \
  --action_eval_chunk_size "$STAGE2_ACTION_EVAL_CHUNK_SIZE" \
  --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" \
  --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --val_ratio 0.1 --split_strategy stratified --num_workers "$STAGE2_NUM_WORKERS" \
  --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
  --best_metric val_macro_f1 --best_mode max --best_min_delta 0.0 \
  --resample_mode targeted \
  --resample_late_mode date_step --resample_late_interval_seconds 2 --resample_late_start_step 316 --resample_late_weight 1 \
  --resample_worst_intersections intersection_1_1,intersection_1_2,intersection_1_3,intersection_2_2,intersection_3_2,intersection_4_2 \
  --resample_worst_intersection_weight 2 \
  --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
  --resample_high_queue_threshold 14 --resample_high_queue_weight 1 \
  --resample_high_wait_threshold 84 \
  --resample_target_action_counts ETWT:4248,NTST:3934,ELWL:2055,NLSL:1885 \
  --log_interval 20 --save_every 1 --seed 2024 --bf16 \
  --ddp_timeout_seconds 1800 \
  > "$LOG_DIR/stage2.log" 2>&1

if [[ ! -s "$STAGE2_DIR/epoch_15.pt" ]]; then
  echo "Stage 2 completed without a valid epoch_15 checkpoint: $STAGE2_DIR/epoch_15.pt" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" projection_train/run_projection_online.py \
  --memo "$ONLINE_MEMO" \
  --projection_ckpt "$STAGE2_DIR/epoch_15.pt" \
  --llm_path "$MODEL_PATH" \
  --dataset jinan \
  --traffic_file "$TRAFFIC_FILE" \
  --run_counts 3600 \
  --seed "$EVAL_SEED" \
  --proj_name Jinan3-qwshort-targeted-online \
  --prompt_mode queue_wait_short \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --traffic_series_interval 2 \
  --bf16 \
  --result_json "$ONLINE_DIR/result.json" \
  > "$LOG_DIR/online_seed_${EVAL_SEED}.log" 2>&1

"$PYTHON" - "$ONLINE_DIR/result.json" "$SUMMARY_JSON" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
payload = json.load(open(result_path, encoding="utf-8"))
metrics = payload.get("results") or {}
if isinstance(metrics, list):
    metrics = metrics[-1] if metrics else {}
summary = {
    "source_archive_metrics": {
        "test_avg_queue_len_over": 157.18333333333334,
        "test_avg_travel_time_over": 268.40778395552024,
        "test_avg_waiting_time_over": 42.847602042067585,
        "test_queuing_vehicle_num_over": 18862.0,
        "test_reward_over": -599.1445932133923,
    },
    "repro_metrics": metrics,
    "result_json": str(result_path),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
