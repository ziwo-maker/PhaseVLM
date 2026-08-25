#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate a frozen checkpoint in exactly one named demand/time scenario. The
# parent campaign assigns one independent evaluator to each physical GPU.

RUN_ROOT="${1:?Usage: $0 RUN_ROOT ARM SEED SCENARIO GPU_ID}"
ARM="${2:?Usage: $0 RUN_ROOT ARM SEED SCENARIO GPU_ID}"
SEED="${3:?Usage: $0 RUN_ROOT ARM SEED SCENARIO GPU_ID}"
SCENARIO="${4:?Usage: $0 RUN_ROOT ARM SEED SCENARIO GPU_ID}"
GPU_ID="${5:?Usage: $0 RUN_ROOT ARM SEED SCENARIO GPU_ID}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
MODEL_PATH=/home/data/model/qwen-vl-8B
SOURCE_RUN_ROOT="$REPO_ROOT/records/pipeline_runs/hangzhou_real_direct_urgency_20260820"
NEW_CKPT="$RUN_ROOT/arms/$ARM/seed_$SEED/stage2/best.pt"
EXISTING_CKPT="$SOURCE_RUN_ROOT/arms/$ARM/seed_$SEED/stage2/best.pt"
EVAL_ROOT="$RUN_ROOT/evaluations/$SCENARIO/$ARM/seed_$SEED"
LOG_DIR="$EVAL_ROOT/logs"
RESULT_PATH="$EVAL_ROOT/result.json"
STATUS_PATH="$EVAL_ROOT/status.txt"
CONFIG_PATH="$EVAL_ROOT/config.json"
PREFIX_TOKENS="${PREFIX_TOKENS:-32}"

[[ "$PREFIX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "PREFIX_TOKENS must be a positive integer: $PREFIX_TOKENS" >&2; exit 2; }

case "$SCENARIO" in
    primary_real)
        TRAFFIC_FILE=anon_4_4_hangzhou_real.json
        RUN_COUNTS=3600
        SCENARIO_ROLE=target_full_demand
        ;;
    source_5816)
        TRAFFIC_FILE=anon_4_4_hangzhou_real_5816.json
        RUN_COUNTS=3600
        SCENARIO_ROLE=training_source_demand_reference
        ;;
    synthetic_high)
        TRAFFIC_FILE=anon_4_4_hangzhou_synthetic_24000_60min.json
        RUN_COUNTS=3600
        SCENARIO_ROLE=synthetic_high_demand_stress
        ;;
    time_early)
        TRAFFIC_FILE=anon_4_4_hangzhou_real_t0000_1200.json
        RUN_COUNTS=1200
        SCENARIO_ROLE=target_real_nonoverlapping_time_slice
        ;;
    time_mid)
        TRAFFIC_FILE=anon_4_4_hangzhou_real_t1200_2400.json
        RUN_COUNTS=1200
        SCENARIO_ROLE=target_real_nonoverlapping_time_slice
        ;;
    time_late)
        TRAFFIC_FILE=anon_4_4_hangzhou_real_t2400_3600.json
        RUN_COUNTS=1200
        SCENARIO_ROLE=target_real_nonoverlapping_time_slice
        ;;
    *)
        echo "Unknown scenario: $SCENARIO" >&2
        exit 2
        ;;
esac

if [[ -s "$NEW_CKPT" ]]; then
    CHECKPOINT="$NEW_CKPT"
    CHECKPOINT_ORIGIN=robustness_extension
elif [[ -s "$EXISTING_CKPT" ]]; then
    CHECKPOINT="$EXISTING_CKPT"
    CHECKPOINT_ORIGIN=original_3seed_campaign
else
    echo "No checkpoint for $ARM seed=$SEED" >&2
    exit 1
fi
[[ -s "$REPO_ROOT/data/Hangzhou/4_4/$TRAFFIC_FILE" ]] || { echo "Missing traffic input: $TRAFFIC_FILE" >&2; exit 1; }
[[ ! -e "$EVAL_ROOT" ]] || { echo "Evaluation directory already exists: $EVAL_ROOT" >&2; exit 1; }

mkdir -p "$LOG_DIR"
"$PYTHON_BIN" - "$CONFIG_PATH" "$ARM" "$SEED" "$SCENARIO" "$TRAFFIC_FILE" "$RUN_COUNTS" "$CHECKPOINT" "$CHECKPOINT_ORIGIN" "$GPU_ID" "$SCENARIO_ROLE" "$PREFIX_TOKENS" <<'PY'
import json
import sys
from datetime import datetime, timezone

(config_path, arm, seed, scenario, traffic_file, run_counts, checkpoint, origin, gpu_id, role, prefix_tokens) = sys.argv[1:]
payload = {
    "schema": "hangzhou_ducpd_robustness_eval_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "arm": arm,
    "training_seed": int(seed),
    "scenario": scenario,
    "scenario_role": role,
    "traffic_file": traffic_file,
    "run_counts": int(run_counts),
    "online_seed": 2024,
    "physical_gpu": int(gpu_id),
    "checkpoint": checkpoint,
    "checkpoint_origin": origin,
    "architecture": {"seq_len": 16, "image_size": 224, "hidden_dim": 256, "prefix_tokens": int(prefix_tokens)},
    "online_phase_override": "disabled",
    "metric_semantics": "vehicle tail = per-vehicle maximum continuous observed wait at action boundaries",
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 && -s "$RESULT_PATH" ]]; then
        printf 'completed_at=%s\nstatus=success\nresult=%s\n' "$(date -Is)" "$RESULT_PATH" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\n' "$(date -Is)" "$code" > "$STATUS_PATH"
    fi
    exit "$code"
}
trap on_exit EXIT

printf 'started_at=%s\nstatus=running\nscenario=%s\ntraffic=%s\ncheckpoint=%s\n' "$(date -Is)" "$SCENARIO" "$TRAFFIC_FILE" "$CHECKPOINT" > "$STATUS_PATH"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

"$PYTHON_BIN" projection_train/run_projection_online.py \
    --memo "hangzhou_ducpd_robust_${SCENARIO}_${ARM}_seed${SEED}" \
    --projection_ckpt "$CHECKPOINT" --llm_path "$MODEL_PATH" \
    --dataset hangzhou --traffic_file "$TRAFFIC_FILE" --run_counts "$RUN_COUNTS" --seed 2024 \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 256 --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix --max_length 64 --bf16 \
    --result_json "$RESULT_PATH" > "$LOG_DIR/online.log" 2>&1
