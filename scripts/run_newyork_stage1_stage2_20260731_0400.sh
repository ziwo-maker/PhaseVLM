#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/lijn99/miniconda3/envs/lab/bin/python3.10
TORCHRUN_BIN=/home/lijn99/miniconda3/envs/lab/bin/torchrun
RUN_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_double_qwctx_20260731_040000"
SIM_ROOT=/mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857
RAW_DATA="$SIM_ROOT/raw/traffic_prompt_series_2s.jsonl"
EXPECTED_LABELS=23520
MONITOR_PID=""

mkdir -p "$RUN_ROOT"
exec >"$RUN_ROOT/pipeline.log" 2>&1

notify_feishu() {
    local title="$1"
    local body="$2"
    local config=/root/.claude/feishu.json
    local mode url payload

    [ -f "$config" ] || return 0
    mode="$(jq -r '.mode // "off"' "$config" 2>/dev/null || true)"
    [ "$mode" = "push" ] || return 0
    url="$(jq -r '.webhook_url // ""' "$config" 2>/dev/null || true)"
    [ -n "$url" ] || return 0
    payload="$(jq -n --arg title "$title" --arg body "$body" '{msg_type:"interactive",card:{header:{title:{tag:"plain_text",content:$title},template:"blue"},elements:[{tag:"markdown",content:$body}]}}')"
    curl -fsS --max-time 10 -X POST "$url" -H 'Content-Type: application/json' -d "$payload" >/dev/null || true
}

on_exit() {
    local exit_code=$?
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    if [ "$exit_code" -eq 0 ]; then
        printf 'completed_at=%s\nstatus=success\n' "$(date -Is)" >"$RUN_ROOT/completion.status"
        notify_feishu "NewYork Stage 1 + Stage 2 Complete" "Run: $RUN_ROOT\nBoth training stages completed successfully."
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\n' "$(date -Is)" "$exit_code" >"$RUN_ROOT/completion.status"
        notify_feishu "NewYork Stage 1 + Stage 2 Failed" "Run: $RUN_ROOT\nExit code: $exit_code\nInspect: $RUN_ROOT/pipeline.log"
    fi
    exit "$exit_code"
}
trap on_exit EXIT

start_progress_monitor() {
    local pipeline_pid=$$
    (
        while kill -0 "$pipeline_pid" 2>/dev/null; do
            sleep 2400
            kill -0 "$pipeline_pid" 2>/dev/null || break
            local_metric=""
            if [ -s "$RUN_ROOT/stage2/checkpoints/metrics.jsonl" ]; then
                local_metric="$(tail -n 1 "$RUN_ROOT/stage2/checkpoints/metrics.jsonl")"
                notify_feishu "NewYork Training Progress" "Stage 2 is running.\nLatest metric: $local_metric\nRun: $RUN_ROOT"
            elif [ -s "$RUN_ROOT/stage1/metrics.jsonl" ]; then
                local_metric="$(tail -n 1 "$RUN_ROOT/stage1/metrics.jsonl")"
                notify_feishu "NewYork Training Progress" "Stage 1 is running.\nLatest metric: $local_metric\nRun: $RUN_ROOT"
            else
                notify_feishu "NewYork Training Progress" "Data validation or initialization is in progress.\nRun: $RUN_ROOT"
            fi
        done
    ) &
    MONITOR_PID=$!
}

cd "$REPO_ROOT"
printf 'scheduled_start_at=%s\n' "$(date -Is)" >"$RUN_ROOT/completion.status"

if [ -s "$SIM_ROOT/pid" ]; then
    sim_pid="$(tr -d '[:space:]' <"$SIM_ROOT/pid")"
    if [ -n "$sim_pid" ] && kill -0 "$sim_pid" 2>/dev/null; then
        echo "SimExpert is still running (pid=$sim_pid); refusing incomplete data."
        exit 12
    fi
fi

"$PYTHON_BIN" - "$RAW_DATA" "$EXPECTED_LABELS" <<'PY'
import json
import sys

path = sys.argv[1]
expected_labels = int(sys.argv[2])
labels = 0
missing_context = 0
missing_queue = 0
missing_wait = 0
last_snapshot = ""

with open(path, encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        record = json.loads(line)
        if not record.get("expert_action"):
            continue
        labels += 1
        missing_context += not bool(record.get("prompt_context"))
        missing_queue += not bool(record.get("phase_queue"))
        missing_wait += not bool(record.get("phase_wait"))
        last_snapshot = str(record.get("snapshot_path") or last_snapshot)

if labels != expected_labels:
    raise SystemExit(f"expected {expected_labels} labeled records, found {labels}")
if missing_context or missing_queue or missing_wait:
    raise SystemExit(
        "missing required context fields: "
        f"prompt_context={missing_context}, phase_queue={missing_queue}, phase_wait={missing_wait}"
    )
print(f"validated labels={labels}, last_snapshot={last_snapshot}")
PY

start_progress_monitor
notify_feishu "NewYork Stage 1 + Stage 2 Started" "Data validation passed: $EXPECTED_LABELS labeled samples with queue/wait context.\nRun: $RUN_ROOT"

printf 'stage1_started_at=%s\n' "$(date -Is)" >>"$RUN_ROOT/completion.status"
CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN_BIN" --standalone --nproc_per_node=2 \
    train_align.py \
    --data_path "$RAW_DATA" \
    --model_path /home/data/model/qwen-vl-8B \
    --output_dir "$RUN_ROOT/stage1" \
    --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix \
    --temperature 0.07 \
    --batch_size 4 --grad_accum_steps 2 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 50 --save_every_epochs 5 --seed 2024 --bf16

printf 'stage1_completed_at=%s\n' "$(date -Is)" >>"$RUN_ROOT/completion.status"
notify_feishu "NewYork Stage 1 Complete" "Stage 1 checkpoint: $RUN_ROOT/stage1/best.pt\nStage 2 will start now."

printf 'stage2_started_at=%s\n' "$(date -Is)" >>"$RUN_ROOT/completion.status"
CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN_BIN" --standalone --nproc_per_node=2 \
    projection_train/train_projection_bc.py \
    --data_path "$RAW_DATA" \
    --llm_path /home/data/model/qwen-vl-8B \
    --align_ckpt "$RUN_ROOT/stage1/best.pt" \
    --output_dir "$RUN_ROOT/stage2/checkpoints" \
    --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --prompt_mode queue_wait_short \
    --batch_size 4 --grad_accum_steps 4 --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy blocked --num_workers 4 --seed 2024 --bf16 \
    --log_interval 20 --save_every 1 --best_metric val_macro_f1 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --ddp_timeout_seconds 1800

printf 'stage2_completed_at=%s\n' "$(date -Is)" >>"$RUN_ROOT/completion.status"
