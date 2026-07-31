#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 TRAIN_RUN_ROOT [DELAY_SECONDS]" >&2
  exit 2
fi

TRAIN_RUN_ROOT="$1"
DELAY_SECONDS="${2:-14400}"
REPO_ROOT="/home/code/LLMSTSC"
PYTHON="/home/lijn99/miniconda3/envs/lab/bin/python"
LLM_PATH="/home/data/model/qwen-vl-8B"
TRAIN_STATUS="$TRAIN_RUN_ROOT/logs/completion.status"
STAGE1_CKPT="$TRAIN_RUN_ROOT/stage1/best.pt"
STAGE2_CKPT="$TRAIN_RUN_ROOT/stage2/checkpoints/best.pt"
GUARD_LOG="$TRAIN_RUN_ROOT/logs/full_online_guard.log"

timestamp() {
  date --iso-8601=seconds
}

json_escape() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  printf '%s' "$value"
}

notify() {
  local title="$1"
  local template="$2"
  local body="$3"
  local mode=""
  local webhook=""
  local payload=""

  [[ -r /root/.claude/feishu.json ]] || return 0
  mode="$(sed -n 's/^[[:space:]]*"mode"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' /root/.claude/feishu.json | sed -n '1p')"
  [[ "$mode" != "off" ]] || return 0
  webhook="$(sed -n 's/^[[:space:]]*"webhook_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' /root/.claude/feishu.json | sed -n '1p')"
  [[ -n "$webhook" ]] || return 0

  payload=$(printf '{"msg_type":"interactive","card":{"header":{"title":{"tag":"plain_text","content":"%s"},"template":"%s"},"elements":[{"tag":"markdown","content":"%s"}]}}' \
    "$(json_escape "$title")" "$template" "$(json_escape "$body")")
  curl --silent --show-error --connect-timeout 5 --max-time 10 --output /dev/null \
    --request POST "$webhook" \
    --header 'Content-Type: application/json' \
    --data "$payload" || true
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$GUARD_LOG"
}

run_online_test() {
  local traffic_file="$1"
  local test_root="$2"
  local traffic_stem="${traffic_file%.json}"
  local test_log="$test_root/logs/${traffic_stem}.log"
  local result_json="$test_root/artifacts/${traffic_stem}.json"
  local memo="pipeline_runs/exp_newyork28x7/${RUN_ID}/${traffic_stem}"
  local status=0

  log "online_test_start traffic_file=$traffic_file result_json=$result_json"
  set +e
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "$memo" \
      --projection_ckpt "$STAGE2_CKPT" \
      --llm_path "$LLM_PATH" \
      --dataset newyork_28x7 \
      --traffic_file "$traffic_file" \
      --run_counts 3600 \
      --num_rounds 1 \
      --proj_name "NewYork28x7-full-online" \
      --device auto \
      --seed 2024 \
      --result_json "$result_json" \
      --prompt_mode image_only_short \
      --seq_len 16 \
      --image_size 224 \
      --periodicity 16 \
      --input_dim 8 \
      --hidden_dim 64 \
      --prefix_tokens 8 \
      --prefix_mode prefix \
      --max_length 64 \
      --bf16 \
      2>&1 | tee "$test_log"
  status=${PIPESTATUS[0]}
  set -e

  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$traffic_file" "$status" "$result_json" \
    >> "$test_root/status.tsv"
  log "online_test_end traffic_file=$traffic_file status=$status"
  return "$status"
}

if ! [[ "$DELAY_SECONDS" =~ ^[0-9]+$ ]] || [[ "$DELAY_SECONDS" -eq 0 ]]; then
  echo "DELAY_SECONDS must be a positive integer." >&2
  exit 2
fi

if [[ ! -d "$TRAIN_RUN_ROOT" ]]; then
  echo "Training run root is missing: $TRAIN_RUN_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$TRAIN_RUN_ROOT/logs"
touch "$GUARD_LOG"
log "scheduled_check delay_seconds=$DELAY_SECONDS"
sleep "$DELAY_SECONDS"

if [[ ! -r "$TRAIN_STATUS" ]]; then
  log "training_not_complete status_file_missing=$TRAIN_STATUS"
  notify "NewYork Training Check" "yellow" "Four-hour check found no completion status. Full online evaluation was not started. Run: $TRAIN_RUN_ROOT"
  exit 3
fi

train_exit_status="$(sed -n 's/^exit_status=//p' "$TRAIN_STATUS" | sed -n '1p')"
stage1_status="$(sed -n 's/^stage1=//p' "$TRAIN_STATUS" | sed -n '1p')"
stage2_status="$(sed -n 's/^stage2=//p' "$TRAIN_STATUS" | sed -n '1p')"
if [[ "$train_exit_status" != "0" || "$stage1_status" != "checkpoint_ready" || "$stage2_status" != "checkpoint_ready" || ! -s "$STAGE1_CKPT" || ! -s "$STAGE2_CKPT" ]]; then
  log "training_failed_or_incomplete exit_status=$train_exit_status stage1=$stage1_status stage2=$stage2_status"
  notify "NewYork Training Check" "red" "Four-hour check found incomplete or failed training. Full online evaluation was not started. exit_status=$train_exit_status stage1=$stage1_status stage2=$stage2_status"
  exit 4
fi

RUN_ID="online_full_$(date +%Y%m%d_%H%M%S)"
TEST_ROOT="$TRAIN_RUN_ROOT/$RUN_ID"
mkdir -p "$TEST_ROOT/logs" "$TEST_ROOT/artifacts"
printf 'timestamp\ttraffic_file\tstatus\tresult_json\n' > "$TEST_ROOT/status.tsv"
log "training_success; starting_full_online_test test_root=$TEST_ROOT"
notify "NewYork Full Online Test Started" "blue" "Training completed normally. Starting 3600-second online tests for NewYork double and triple traffic files. Results: $TEST_ROOT"

overall_status=0
run_online_test anon_28_7_newyork_real_double.json "$TEST_ROOT" || overall_status=1
run_online_test anon_28_7_newyork_real_triple.json "$TEST_ROOT" || overall_status=1

if [[ "$overall_status" -eq 0 ]]; then
  log "full_online_test_success test_root=$TEST_ROOT"
  notify "NewYork Full Online Test Complete" "green" "Both 3600-second NewYork online tests completed successfully. Results: $TEST_ROOT"
else
  log "full_online_test_failed test_root=$TEST_ROOT"
  notify "NewYork Full Online Test Failed" "red" "One or more 3600-second NewYork online tests failed. Inspect: $TEST_ROOT/status.tsv"
fi

exit "$overall_status"
