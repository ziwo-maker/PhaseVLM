#!/usr/bin/env bash
# Evaluate one checkpoint on the complete Jinan generalization matrix.
# Keep all online seeds on one fixed GPU so seed effects are not confounded
# with GPU-specific numerical execution differences.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
TARGET_DIR="${2:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
CHECKPOINT="${3:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
MEMO_PREFIX="${4:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${DIRECT_URGENCY_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /home/sysu/anaconda3/envs/hnr5090/bin/python ]]; then
        PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
if [[ -n "${DIRECT_URGENCY_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$DIRECT_URGENCY_LD_PRELOAD"
fi
LLM_PATH=/home/data/model/qwen-vl-8B
ONLINE_ROOT="$TARGET_DIR/evaluation"
LOG_ROOT="$TARGET_DIR/logs/online"
STATUS_PATH="$TARGET_DIR/online.status"
PROTOCOL_PATH="$TARGET_DIR/online_protocol.env"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
FLOWS=(
    anon_3_4_jinan_real.json
    anon_3_4_jinan_real_2000.json
    anon_3_4_jinan_real_2500.json
)
EVAL_SEEDS=(2024 42 123)
GPUS=()
EVAL_GPU="${DIRECT_URGENCY_EVAL_GPU:-}"
LAST_NOTIFICATION_EPOCH=0

discover_idle_gpus() {
    local gpu used
    GPUS=()
    if [[ -n "${DIRECT_URGENCY_GPUS:-}" ]]; then
        IFS=',' read -r -a GPUS <<< "$DIRECT_URGENCY_GPUS"
    else
        while IFS=, read -r gpu used; do
            gpu="${gpu//[[:space:]]/}"
            used="${used//[[:space:]]/}"
            [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
            if (( used < 500 )); then
                GPUS+=("$gpu")
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    fi
    ((${#GPUS[@]} > 0)) || { echo "No idle GPU is available for online evaluation" >&2; return 1; }
    if [[ -z "$EVAL_GPU" ]]; then
        EVAL_GPU="${GPUS[0]}"
    fi
    [[ "$EVAL_GPU" =~ ^[0-9]+$ ]] || { echo "Invalid DIRECT_URGENCY_EVAL_GPU: $EVAL_GPU" >&2; return 1; }
    if [[ ! " ${GPUS[*]} " =~ " $EVAL_GPU " ]]; then
        echo "Evaluation GPU $EVAL_GPU is not idle" >&2
        return 1
    fi
}

mkdir -p "$ONLINE_ROOT" "$LOG_ROOT"
[[ -s "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
discover_idle_gpus

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

latest_result() {
    local result
    result="$(find "$ONLINE_ROOT" -name result.json -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "$result" ]]; then
        "$PYTHON_BIN" - "$result" <<'PY' 2>/dev/null || true
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = payload.get("results", {})
print("latest=%s AQL=%s AWT=%s ATT=%s" % (
    sys.argv[1],
    metrics.get("test_avg_queue_len_over", "NA"),
    metrics.get("test_avg_waiting_time_over", "NA"),
    metrics.get("test_avg_travel_time_over", "NA"),
))
PY
    else
        printf 'latest=not_available'
    fi
}

maybe_progress_notification() {
    local phase="$1"
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "jinan_real direct-urgency online progress: phase=${phase}; $(latest_result)"
}

result_matches_protocol() {
    local result="$1" flow="$2" seed="$3"
    "$PYTHON_BIN" - "$result" "$CHECKPOINT" "$REPO_ROOT/projection_train/run_projection_online.py" "$flow" "$seed" <<'PY'
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

snapshot_wave() {
    local flow="$1"
    local label="$2"
    shift 2
    local snapshot="$LOG_ROOT/${flow%.json}.startup_monitor.log"
    {
        printf '\n[%s] %s pids=%s\n' "$(timestamp)" "$label" "$*"
        for pid in "$@"; do
            timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | \
                awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 80 || true
        done
        for seed in "${EVAL_SEEDS[@]}"; do
            local log="$LOG_ROOT/${flow%.json}/seed_${seed}.log"
            if [[ -f "$log" ]]; then
                timeout --foreground 10s tail -n 80 "$log" || true
                timeout --foreground 10s grep -En 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log" | tail -n 40 || true
            fi
        done
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$snapshot" 2>&1
}

terminate_wave() {
    for pid in "$@"; do
        kill -- -"$pid" 2>/dev/null || true
    done
}

check_wave_logs() {
    local flow="$1"
    local seed
    for seed in "${EVAL_SEEDS[@]}"; do
        local log="$LOG_ROOT/${flow%.json}/seed_${seed}.log"
        [[ -f "$log" ]] || continue
        if timeout --foreground 10s grep -Eq 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log"; then
            return 1
        fi
    done
    return 0
}

run_seed() {
    local flow="$1" seed="$2"
    local flow_name="${flow%.json}"
    local result="$ONLINE_ROOT/$flow_name/seed_${seed}/result.json"
    local log="$LOG_ROOT/$flow_name/seed_${seed}.log"
    local memo="${MEMO_PREFIX}_${flow_name}_eval${seed}"
    local pid minute

    mkdir -p "$(dirname "$result")" "$(dirname "$log")"
    if [[ -e "$result" ]]; then
        if result_matches_protocol "$result" "$flow" "$seed"; then
            printf '[%s] reuse verified online result: flow=%s seed=%s gpu=%s\n' "$(timestamp)" "$flow" "$seed" "$EVAL_GPU"
            return 0
        fi
        printf '[%s] refusing stale or invalid online result: %s\n' "$(timestamp)" "$result" >&2
        return 1
    fi

    setsid env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
        "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
        --memo "$memo" \
        --projection_ckpt "$CHECKPOINT" --llm_path "$LLM_PATH" \
        --dataset jinan --traffic_file "$flow" --run_counts 3600 --num_rounds 1 --seed "$seed" \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 --deterministic --disable_snapshots --result_json "$result" \
        > "$log" 2>&1 &
    pid="$!"
    printf '[%s] online launched: flow=%s seed=%s gpu=%s pid=%s\n' "$(timestamp)" "$flow" "$seed" "$EVAL_GPU" "$pid"

    snapshot_wave "$flow" "seed_${seed}_startup_t0" "$pid"
    for minute in 1 2 3 4 5; do
        sleep 60
        snapshot_wave "$flow" "seed_${seed}_startup_t${minute}m" "$pid"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" && [[ -s "$result" ]]
            return
        fi
        if ! check_wave_logs "$flow"; then
            terminate_wave "$pid"
            wait "$pid" || true
            return 1
        fi
    done

    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        maybe_progress_notification "online:${flow}:seed_${seed}"
    done
    wait "$pid" && [[ -s "$result" ]]
}

run_flow() {
    local flow="$1" seed
    for seed in "${EVAL_SEEDS[@]}"; do
        run_seed "$flow" "$seed"
    done
    printf '[%s] online flow completed: %s\n' "$(timestamp)" "$flow"
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\ncheckpoint=%s\ncheckpoint_sha256=%s\nevaluator_sha256=%s\neval_gpu=%s\ndeterministic=true\n' \
            "$(timestamp)" "$CHECKPOINT" "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" \
            "$(sha256sum "$REPO_ROOT/projection_train/run_projection_online.py" | awk '{print $1}')" "$EVAL_GPU" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\ncheckpoint=%s\n' "$(timestamp)" "$code" "$CHECKPOINT" > "$STATUS_PATH"
    fi
    trap - EXIT
    exit "$code"
}
trap on_exit EXIT

printf 'started_at=%s\nstatus=running\ncheckpoint=%s\nflows=%s\neval_seeds=%s\neval_gpu=%s\ndeterministic=true\n' \
    "$(timestamp)" "$CHECKPOINT" "${FLOWS[*]}" "${EVAL_SEEDS[*]}" "$EVAL_GPU" > "$STATUS_PATH"
printf 'created_at=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\neval_gpu=%s\neval_seed_execution=serial_single_gpu\ndeterministic=true\n' \
    "$(timestamp)" "$CHECKPOINT" "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" "$EVAL_GPU" > "$PROTOCOL_PATH"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "jinan_real direct-urgency online suite launched: target=$(basename "$TARGET_DIR"), flows=${FLOWS[*]}, eval_seeds=${EVAL_SEEDS[*]}, fixed_gpu=${EVAL_GPU}."
for flow in "${FLOWS[@]}"; do
    run_flow "$flow"
done
notify "jinan_real direct-urgency online suite completed: target=$(basename "$TARGET_DIR"); $(latest_result)"
