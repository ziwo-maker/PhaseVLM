#!/usr/bin/env bash
# Deterministically evaluate one wait25 refinement checkpoint on Jinan real only.
set -Eeuo pipefail

RUN_ROOT="${1:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
TARGET_DIR="${2:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
CHECKPOINT="${3:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
MEMO_PREFIX="${4:?Usage: $0 RUN_ROOT TARGET_DIR CHECKPOINT MEMO_PREFIX}"
REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${WAIT25_IMPROVEMENT_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
FLOW=anon_3_4_jinan_real.json
EVAL_SEEDS=(2024 42 123)
LLM_PATH=/home/data/model/qwen-vl-8B
ONLINE_ROOT="$TARGET_DIR/evaluation"
LOG_ROOT="$TARGET_DIR/logs/online"
STATUS_PATH="$TARGET_DIR/online.status"
PROTOCOL_PATH="$TARGET_DIR/online_protocol.env"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
EVAL_GPUS_RAW="${WAIT25_IMPROVEMENT_EVAL_GPUS:-0,1,2}"
HIDDEN_DIM="${WAIT25_IMPROVEMENT_HIDDEN_DIM:-64}"
PREFIX_TOKENS="${WAIT25_IMPROVEMENT_PREFIX_TOKENS:-8}"
STARTUP_MONITOR_MODE="${WAIT25_IMPROVEMENT_ONLINE_STARTUP_MONITOR:-full}"
LAST_NOTIFICATION_EPOCH=0
EVAL_GPUS=()
declare -A PID_BY_SEED=()
declare -A GPU_BY_SEED=()

timestamp() { date -Is; }

notify() { [[ -f "$FEISHU_SCRIPT" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" || true; }

discover_eval_gpus() {
    local gpu used requested
    IFS=',' read -r -a EVAL_GPUS <<< "$EVAL_GPUS_RAW"
    ((${#EVAL_GPUS[@]} == ${#EVAL_SEEDS[@]})) || { echo "Need one evaluation GPU per seed; got ${#EVAL_GPUS[@]} GPUs for ${#EVAL_SEEDS[@]} seeds" >&2; return 1; }
    declare -A seen=()
    for requested in "${EVAL_GPUS[@]}"; do
        requested="${requested//[[:space:]]/}"
        [[ "$requested" =~ ^[0-9]+$ && -z "${seen[$requested]:-}" ]] || { echo "Invalid or duplicate evaluation GPU: $requested" >&2; return 1; }
        seen["$requested"]=1
        used="$(timeout --foreground 10s nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$requested" | tr -dc '0-9')"
        [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 ]] || { echo "Evaluation GPU ${requested} is not idle (${used:-unknown} MiB)" >&2; return 1; }
    done
}

latest_result() {
    local seed path
    for seed in 123 42 2024; do
        path="$ONLINE_ROOT/${FLOW%.json}/seed_${seed}/result.json"
        [[ -s "$path" ]] || continue
        "$PYTHON_BIN" - "$path" <<'PY' 2>/dev/null || true
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
print("AQL=%s AWT=%s ATT=%s" % (r.get("test_avg_queue_len_over"), r.get("test_avg_waiting_time_over"), r.get("test_avg_travel_time_over")))
PY
        return 0
    done
    printf 'online_result=not_available'
}

maybe_notify_progress() {
    local now
    now="$(date +%s)"; (( now - LAST_NOTIFICATION_EPOCH >= 2400 )) || return 0
    LAST_NOTIFICATION_EPOCH="$now"
    notify "jinan_real wait25 refinement online progress: target=$(basename "$TARGET_DIR"); $(latest_result)"
}

result_matches_protocol() {
    local result="$1" seed="$2"
    "$PYTHON_BIN" - "$result" "$CHECKPOINT" "$REPO_ROOT/projection_train/run_projection_online.py" "$FLOW" "$seed" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

result = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
evaluator = Path(sys.argv[3])
flow = sys.argv[4]
seed = int(sys.argv[5])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
try:
    p = json.loads(result.read_text(encoding="utf-8")); a = p["args"]; h = p["artifact_hashes"]; d = p["determinism"]
    assert a["traffic_file"] == flow and int(a["seed"]) == seed and int(a["run_counts"]) == 3600
    assert Path(a["projection_ckpt"]).resolve() == checkpoint.resolve()
    assert h["projection_ckpt_sha256"] == digest(checkpoint) and h["online_evaluator_sha256"] == digest(evaluator)
    assert a["deterministic"] is True and d["requested"] is True and d["torch_deterministic_algorithms"] is True and d["cudnn_deterministic"] is True and d["cudnn_benchmark"] is False and d["allow_tf32"] is False
    assert all(float(a.get(key, 0.0) or 0.0) == 0.0 for key in ("max_phase_override_gap", "max_phase_override_left_wait_protect", "max_phase_override_target_queue_floor", "max_phase_override_total_queue_floor"))
    assert isinstance(p["results"], dict) and all(math.isfinite(float(value)) for value in p["results"].values())
except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

snapshot() {
    local seed="$1" label="$2" pid="$3" gpu="$4" log=
    log="$LOG_ROOT/${FLOW%.json}/seed_${seed}.log"
    {
        printf '\n[%s] %s child_pid=%s gpu=%s\n' "$(timestamp)" "$label" "$pid" "$gpu"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 80 || true
        [[ -f "$log" ]] && timeout --foreground 10s tail -n 60 "$log" || true
        [[ -f "$log" ]] && timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log" | tail -n 30 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$LOG_ROOT/${FLOW%.json}.startup_monitor.log" 2>&1
}

launch_seed() {
    local seed="$1" gpu="$2" name="${FLOW%.json}" result= log= pid=
    result="$ONLINE_ROOT/${FLOW%.json}/seed_${seed}/result.json"
    log="$LOG_ROOT/${FLOW%.json}/seed_${seed}.log"
    mkdir -p "$(dirname "$result")" "$(dirname "$log")"
    if [[ -e "$result" ]]; then result_matches_protocol "$result" "$seed" && return 0; echo "Invalid existing result: $result" >&2; return 1; fi
    setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" --memo "${MEMO_PREFIX}_${name}_eval${seed}" --projection_ckpt "$CHECKPOINT" --llm_path "$LLM_PATH" --dataset jinan --traffic_file "$FLOW" --run_counts 3600 --num_rounds 1 --seed "$seed" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim "$HIDDEN_DIM" --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix --max_length 64 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots --result_json "$result" > "$log" 2>&1 &
    pid=$!
    PID_BY_SEED["$seed"]="$pid"; GPU_BY_SEED["$seed"]="$gpu"
    snapshot "$seed" "seed_${seed}_startup_t0" "$pid" "$gpu"
}

monitor_parallel_seeds() {
    # Bash with `set -u` expands associative indexes while parsing the
    # assignment. Initialize every local used in those expansions first.
    local minute= seed= pid= gpu= log= result= all_finished=
    if [[ "$STARTUP_MONITOR_MODE" == "initial" ]]; then
        while ((${#PID_BY_SEED[@]})); do
            sleep 60; maybe_notify_progress
            for seed in "${EVAL_SEEDS[@]}"; do
                pid="${PID_BY_SEED[$seed]:-}"; [[ -n "$pid" ]] || continue
                log="$LOG_ROOT/${FLOW%.json}/seed_${seed}.log"
                result="$ONLINE_ROOT/${FLOW%.json}/seed_${seed}/result.json"
                if timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log"; then
                    for seed in "${EVAL_SEEDS[@]}"; do
                        pid="${PID_BY_SEED[$seed]:-}"; [[ -n "$pid" ]] && kill -- -"$pid" 2>/dev/null || true
                    done
                    wait || true
                    return 1
                fi
                if ! kill -0 "$pid" 2>/dev/null; then
                    wait "$pid" && [[ -s "$result" ]] && result_matches_protocol "$result" "$seed" || return 1
                    unset 'PID_BY_SEED[$seed]'
                fi
            done
        done
        return 0
    fi
    for minute in 1 2 3 4 5; do
        sleep 60
        all_finished=1
        for seed in "${EVAL_SEEDS[@]}"; do
            pid="${PID_BY_SEED[$seed]:-}"; gpu="${GPU_BY_SEED[$seed]:-}"
            [[ -n "$pid" ]] || continue
            log="$LOG_ROOT/${FLOW%.json}/seed_${seed}.log"
            result="$ONLINE_ROOT/${FLOW%.json}/seed_${seed}/result.json"
            snapshot "$seed" "seed_${seed}_startup_t${minute}m" "$pid" "$gpu"
            if kill -0 "$pid" 2>/dev/null; then
                all_finished=0
                if timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf' "$log"; then
                    for seed in "${EVAL_SEEDS[@]}"; do
                        pid="${PID_BY_SEED[$seed]:-}"; [[ -n "$pid" ]] && kill -- -"$pid" 2>/dev/null || true
                    done
                    wait || true
                    return 1
                fi
            else
                wait "$pid" && [[ -s "$result" ]] && result_matches_protocol "$result" "$seed" || return 1
                unset 'PID_BY_SEED[$seed]'
            fi
        done
        (( all_finished )) && return 0
    done
    while ((${#PID_BY_SEED[@]})); do
        sleep 60; maybe_notify_progress
        for seed in "${EVAL_SEEDS[@]}"; do
            pid="${PID_BY_SEED[$seed]:-}"; [[ -n "$pid" ]] || continue
            result="$ONLINE_ROOT/${FLOW%.json}/seed_${seed}/result.json"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" && [[ -s "$result" ]] && result_matches_protocol "$result" "$seed" || return 1
                unset 'PID_BY_SEED[$seed]'
            fi
        done
    done
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\ncheckpoint=%s\ncheckpoint_sha256=%s\nevaluator_sha256=%s\nevaluation_flow=%s\neval_gpus=%s\ndeterministic=true\n' "$(timestamp)" "$CHECKPOINT" "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" "$(sha256sum "$REPO_ROOT/projection_train/run_projection_online.py" | awk '{print $1}')" "$FLOW" "${EVAL_GPUS[*]}" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\ncheckpoint=%s\nevaluation_flow=%s\n' "$(timestamp)" "$code" "$CHECKPOINT" "$FLOW" > "$STATUS_PATH"
    fi
    trap - EXIT; exit "$code"
}
trap on_exit EXIT

[[ -s "$CHECKPOINT" && -s "$REPO_ROOT/data/Jinan/3_4/$FLOW" ]] || { echo "Missing online prerequisite" >&2; exit 1; }
mkdir -p "$ONLINE_ROOT" "$LOG_ROOT"; discover_eval_gpus
printf 'started_at=%s\nstatus=running\ncheckpoint=%s\nevaluation_flow=%s\neval_seeds=%s\neval_gpus=%s\ndeterministic=true\n' "$(timestamp)" "$CHECKPOINT" "$FLOW" "${EVAL_SEEDS[*]}" "${EVAL_GPUS[*]}" > "$STATUS_PATH"
printf 'created_at=%s\ntraining_flow=anon_3_4_jinan_real.json\nevaluation_flow=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\neval_gpus=%s\neval_seed_execution=parallel_one_seed_per_gpu\neval_seeds=%s\ndeterministic=true\nrun_counts=3600\nonline_phase_override=disabled\n' "$(timestamp)" "$FLOW" "$CHECKPOINT" "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" "${EVAL_GPUS[*]}" "${EVAL_SEEDS[*]}" > "$PROTOCOL_PATH"
LAST_NOTIFICATION_EPOCH="$(date +%s)"; notify "jinan_real wait25 refinement online launched: target=$(basename "$TARGET_DIR"), flow=${FLOW}, eval_seeds=${EVAL_SEEDS[*]}, gpus=${EVAL_GPUS[*]}, execution=parallel."
for index in "${!EVAL_SEEDS[@]}"; do launch_seed "${EVAL_SEEDS[$index]}" "${EVAL_GPUS[$index]}"; done
monitor_parallel_seeds
notify "jinan_real wait25 refinement online completed: target=$(basename "$TARGET_DIR"); $(latest_result)"
