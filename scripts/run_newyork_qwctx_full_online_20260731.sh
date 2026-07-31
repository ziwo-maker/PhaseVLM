#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_double_qwctx_20260731_040000}"
TEST_ROOT="${2:-$RUN_ROOT/online_test_full_queue_wait_short_$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="/home/code/LLMSTSC"
PYTHON="/home/lijn99/miniconda3/envs/lab/bin/python"
LLM_PATH="/home/data/model/qwen-vl-8B"
CKPT="$RUN_ROOT/stage2/checkpoints/best.pt"
PROMPT_MODE="queue_wait_short"

notify_feishu() {
  local title="$1"
  local body="$2"
  local mode=""
  local webhook=""
  local payload=""

  [[ -r /root/.claude/feishu.json ]] || return 0
  mode="$(sed -n 's/^[[:space:]]*"mode"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' /root/.claude/feishu.json | sed -n '1p')"
  [[ "$mode" != "off" ]] || return 0
  webhook="$(sed -n 's/^[[:space:]]*"webhook_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' /root/.claude/feishu.json | sed -n '1p')"
  [[ -n "$webhook" ]] || return 0

  payload="$("$PYTHON" - "$title" "$body" <<'PY'
import json
import sys

title, body = sys.argv[1], sys.argv[2]
print(json.dumps({
    "msg_type": "interactive",
    "card": {
        "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
        "elements": [{"tag": "markdown", "content": body}],
    },
}, ensure_ascii=False))
PY
)"
  curl --silent --show-error --connect-timeout 5 --max-time 10 --output /dev/null \
    --request POST "$webhook" \
    --header 'Content-Type: application/json' \
    --data "$payload" || true
}

latest_progress() {
  local latest_log=""
  local latest_result=""
  local status_tail=""
  status_tail="$(tail -n 5 "$TEST_ROOT/status.tsv" 2>/dev/null || true)"
  latest_log="$(ls -t "$TEST_ROOT"/logs/*.log 2>/dev/null | sed -n '1p')"
  latest_result="$(ls -t "$TEST_ROOT"/artifacts/*.json 2>/dev/null | sed -n '1p')"

  {
    printf 'Run: %s\nPrompt: %s\nCheckpoint: %s\n\n' "$TEST_ROOT" "$PROMPT_MODE" "$CKPT"
    if [[ -n "$status_tail" ]]; then
      printf 'Status:\n%s\n\n' "$status_tail"
    fi
    if [[ -n "$latest_result" ]]; then
      "$PYTHON" - "$latest_result" <<'PY' || true
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    obj = json.loads(p.read_text())
except Exception as exc:
    print(f"Latest result: {p} ({type(exc).__name__})")
    raise SystemExit
print(f"Latest result: {p}")
for key in ("average_travel_time", "avg_travel_time", "reward", "final_score", "avg_queue_length", "avg_waiting_time"):
    if key in obj:
        print(f"{key}: {obj[key]}")
metrics = obj.get("metrics")
if isinstance(metrics, dict):
    for key in ("average_travel_time", "avg_travel_time", "reward", "avg_queue_length", "avg_waiting_time"):
        if key in metrics:
            print(f"metrics.{key}: {metrics[key]}")
PY
    elif [[ -n "$latest_log" ]]; then
      printf 'Latest log: %s\n' "$latest_log"
      tail -n 20 "$latest_log" 2>/dev/null || true
    else
      printf 'No online log has been written yet.\n'
    fi
  }
}

progress_monitor() {
  local parent_pid="$1"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sleep 2400
    kill -0 "$parent_pid" 2>/dev/null || break
    notify_feishu "NewYork Full Online Progress" "$(latest_progress)"
  done
}

run_online_test() {
  local traffic_file="$1"
  local stem="${traffic_file%.json}"
  local log_path="$TEST_ROOT/logs/${stem}.log"
  local result_json="$TEST_ROOT/artifacts/${stem}.json"
  local memo="pipeline_runs/newyork28x7_double_qwctx_20260731_040000/${stem}"
  local status=0

  printf '%s\t%s\tstarted\n' "$(date -Is)" "$traffic_file" >> "$TEST_ROOT/status.tsv"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" projection_train/run_projection_online.py \
    --memo "$memo" \
    --projection_ckpt "$CKPT" \
    --llm_path "$LLM_PATH" \
    --dataset newyork_28x7 \
    --traffic_file "$traffic_file" \
    --run_counts 3600 \
    --num_rounds 1 \
    --proj_name "NewYork28x7-full-online-qwctx" \
    --device auto \
    --seed 2024 \
    --result_json "$result_json" \
    --prompt_mode "$PROMPT_MODE" \
    --seq_len 16 \
    --image_size 224 \
    --periodicity 16 \
    --input_dim 8 \
    --hidden_dim 64 \
    --prefix_tokens 8 \
    --prefix_mode prefix \
    --max_length 64 \
    --inference_batch_size 64 \
    --bf16 \
    > "$log_path" 2>&1
  status=$?
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$traffic_file" "$status" "$result_json" >> "$TEST_ROOT/status.tsv"
  return "$status"
}

main() {
  cd "$REPO_ROOT" || exit 1
  mkdir -p "$TEST_ROOT/logs" "$TEST_ROOT/artifacts"
  printf 'timestamp\ttraffic_file\tstatus\tresult_json\n' > "$TEST_ROOT/status.tsv"
  printf '%s\n' "$$" > "$TEST_ROOT/launcher.pid"

  if [[ ! -s "$CKPT" ]]; then
    echo "Missing checkpoint: $CKPT" >&2
    exit 2
  fi

  notify_feishu "NewYork Full Online Started" "Run: $TEST_ROOT\nCheckpoint: $CKPT\nPrompt: $PROMPT_MODE\nTraffic: double + triple\nrun_counts=3600, inference_batch_size=64"
  progress_monitor "$$" &
  monitor_pid=$!
  printf '%s\n' "$monitor_pid" > "$TEST_ROOT/progress_monitor.pid"

  overall_status=0
  run_online_test anon_28_7_newyork_real_double.json || overall_status=1
  run_online_test anon_28_7_newyork_real_triple.json || overall_status=1

  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  if [[ "$overall_status" -eq 0 ]]; then
    notify_feishu "NewYork Full Online Complete" "$(latest_progress)"
  else
    notify_feishu "NewYork Full Online Failed" "$(latest_progress)"
  fi
  exit "$overall_status"
}

main "$@"
