#!/usr/bin/env bash
# Evaluate one frozen-SimExpert checkpoint on the Jinan real flow only.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
TARGET_DIR="${2:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
CHECKPOINT="${3:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
MEMO_PREFIX="${4:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${FROZEN_SIMEXPERT_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /home/sysu/anaconda3/envs/hnr5090/bin/python ]]; then
        PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
if [[ -n "${FROZEN_SIMEXPERT_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$FROZEN_SIMEXPERT_LD_PRELOAD"
fi

LLM_PATH=/home/data/model/qwen-vl-8B
FLOW=anon_3_4_jinan_real.json
EVAL_SEEDS=(2024 42 123)
ONLINE_ROOT="$TARGET_DIR/evaluation"
LOG_ROOT="$TARGET_DIR/logs/online"
STATUS_PATH="$TARGET_DIR/online.status"
PROTOCOL_PATH="$TARGET_DIR/online_protocol.env"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
EVAL_GPU="${FROZEN_SIMEXPERT_EVAL_GPU:-0}"
LAST_NOTIFICATION_EPOCH=0

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

discover_eval_gpu() {
    local gpu used
    [[ "$EVAL_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid FROZEN_SIMEXPERT_EVAL_GPU: $EVAL_GPU" >&2; return 1; }
    while IFS=, read -r gpu used; do
        gpu="${gpu//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        if [[ "$gpu" == "$EVAL_GPU" && "$used" =~ ^[0-9]+$ ]] && (( used < 500 )); then
            return 0
        fi
    done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "Evaluation GPU $EVAL_GPU is not idle" >&2
    return 1
}

latest_result() {
    local seed result
    for seed in 123 42 2024; do
        result="$ONLINE_ROOT/${FLOW%.json}/seed_${seed}/result.json"
        if [[ -s "$result" ]]; then
            "$PYTHON_BIN" - "$result" <<'PY' 2>/dev/null || true
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = payload.get("results", {})
print("latest=%s AQL=%s AWT=%s ATT=%s" % (
    sys.argv[1],
    metrics.get("test_avg_queue_len_over", "NA"),
    metrics.get("test_avg_waiting_time_over", "NA"),
    metrics.get("test_avg_travel_time_over", "NA"),
))
PY
            return 0
        fi
    done
    printf 'latest=not_available'
}

maybe_progress_notification() {
    local phase="$1"
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "jinan_real frozen SimExpert online progress: ${phase}; $(latest_result)"
}

result_matches_protocol() {
    local result="$1" seed="$2"
    "$PYTHON_BIN" - "$result" "$CHECKPOINT" "$REPO_ROOT/projection_train/run_projection_online.py" "$FLOW" "$seed" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

result_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
evaluator_path = Path(sys.argv[3])
expected_flow = sys.argv[4]
expected_seed = int(sys.argv[5])

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    args = payload["args"]
    hashes = payload["artifact_hashes"]
    determinism = payload["determinism"]
    assert args["traffic_file"] == expected_flow
    assert int(args["seed"]) == expected_seed
    assert int(args["run_counts"]) == 3600
    assert Path(args["projection_ckpt"]).resolve() == checkpoint_path.resolve()
    assert hashes["projection_ckpt_sha256"] == sha256_file(checkpoint_path)
    assert hashes["online_evaluator_sha256"] == sha256_file(evaluator_path)
    assert args["deterministic"] is True
    assert determinism["requested"] is True
    assert determinism["torch_deterministic_algorithms"] is True
    assert determinism["cudnn_benchmark"] is False
    assert determinism["cudnn_deterministic"] is True
    assert determinism["allow_tf32"] is False
    assert all(float(args.get(key, 0.0) or 0.0) == 0.0 for key in (
        "max_phase_override_gap",
        "max_phase_override_left_wait_protect",
        "max_phase_override_target_queue_floor",
        "max_phase_override_total_queue_floor",
    ))
    assert isinstance(payload["results"], dict)
    assert all(math.isfinite(float(value)) for value in payload["results"].values())
except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

snapshot_seed() {
    local seed="$1" label="$2" pid="$3"
    local snapshot="$LOG_ROOT/${FLOW%.json}.startup_monitor.log"
    local log="$LOG_ROOT/${FLOW%.json}/seed_${seed}.log"
    {
        printf '\n[%s] %s child_pid=%s\n' "$(timestamp)" "$label" "$pid"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | \
            awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 80 || true
        if [[ -f "$log" ]]; then
            timeout --foreground 10s tail -n 80 "$log" || true
            timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log" | tail -n 40 || true
        fi
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$snapshot" 2>&1
}

run_seed() {
    local seed="$1"
    local flow_name="${FLOW%.json}"
    local result="$ONLINE_ROOT/$flow_name/seed_${seed}/result.json"
    local log="$LOG_ROOT/$flow_name/seed_${seed}.log"
    local memo="${MEMO_PREFIX}_${flow_name}_eval${seed}"
    local pid minute

    mkdir -p "$(dirname "$result")" "$(dirname "$log")"
    if [[ -e "$result" ]]; then
        if result_matches_protocol "$result" "$seed"; then
            printf '[%s] reuse verified online result: flow=%s seed=%s gpu=%s\n' "$(timestamp)" "$FLOW" "$seed" "$EVAL_GPU"
            return 0
        fi
        printf '[%s] refusing stale or invalid online result: %s\n' "$(timestamp)" "$result" >&2
        return 1
    fi

    setsid env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
        "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
        --memo "$memo" --projection_ckpt "$CHECKPOINT" --llm_path "$LLM_PATH" \
        --dataset jinan --traffic_file "$FLOW" --run_counts 3600 --num_rounds 1 --seed "$seed" \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 --deterministic --disable_snapshots --result_json "$result" \
        > "$log" 2>&1 &
    pid="$!"
    printf '[%s] online launched: flow=%s seed=%s gpu=%s pid=%s\n' "$(timestamp)" "$FLOW" "$seed" "$EVAL_GPU" "$pid"

    snapshot_seed "$seed" "seed_${seed}_startup_t0" "$pid"
    for minute in 1 2 3 4 5; do
        sleep 60
        snapshot_seed "$seed" "seed_${seed}_startup_t${minute}m" "$pid"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" && [[ -s "$result" ]]
            result_matches_protocol "$result" "$seed"
            return
        fi
        if timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log"; then
            kill -- -"$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        fi
    done
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        maybe_progress_notification "flow=${FLOW}:seed=${seed}"
    done
    wait "$pid" && [[ -s "$result" ]]
    result_matches_protocol "$result" "$seed"
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\ncheckpoint=%s\ncheckpoint_sha256=%s\nevaluator_sha256=%s\nevaluation_flow=%s\neval_gpu=%s\ndeterministic=true\n' \
            "$(timestamp)" "$CHECKPOINT" "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" \
            "$(sha256sum "$REPO_ROOT/projection_train/run_projection_online.py" | awk '{print $1}')" "$FLOW" "$EVAL_GPU" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\ncheckpoint=%s\nevaluation_flow=%s\n' \
            "$(timestamp)" "$code" "$CHECKPOINT" "$FLOW" > "$STATUS_PATH"
    fi
    trap - EXIT
    exit "$code"
}
trap on_exit EXIT

[[ -s "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -s "$REPO_ROOT/data/Jinan/3_4/$FLOW" ]] || { echo "Missing Jinan real traffic file: $FLOW" >&2; exit 1; }
mkdir -p "$ONLINE_ROOT" "$LOG_ROOT"
discover_eval_gpu
printf 'started_at=%s\nstatus=running\ncheckpoint=%s\nevaluation_flow=%s\neval_seeds=%s\neval_gpu=%s\ndeterministic=true\n' \
    "$(timestamp)" "$CHECKPOINT" "$FLOW" "${EVAL_SEEDS[*]}" "$EVAL_GPU" > "$STATUS_PATH"
printf 'created_at=%s\ntraining_flow=anon_3_4_jinan_real.json\nevaluation_flow=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\neval_gpu=%s\neval_seed_execution=serial_single_gpu\neval_seeds=%s\ndeterministic=true\nrun_counts=3600\nonline_phase_override=disabled\n' \
    "$(timestamp)" "$FLOW" "$CHECKPOINT" "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" "$EVAL_GPU" "${EVAL_SEEDS[*]}" > "$PROTOCOL_PATH"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "jinan_real frozen SimExpert online suite launched: target=$(basename "$TARGET_DIR"), flow=${FLOW}, eval_seeds=${EVAL_SEEDS[*]}, fixed_gpu=${EVAL_GPU}."
for seed in "${EVAL_SEEDS[@]}"; do
    run_seed "$seed"
done
notify "jinan_real frozen SimExpert online suite completed: target=$(basename "$TARGET_DIR"); $(latest_result)"
