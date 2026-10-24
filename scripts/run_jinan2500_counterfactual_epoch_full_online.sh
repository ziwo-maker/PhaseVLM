#!/usr/bin/env bash
# Full same-flow online evaluation of every saved epoch from each counterfactual arm.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/counterfactual_validation/jinan2500_epoch15_counterfactual_retry_20260904_124359}"
RUN_ID="${RUN_ID:-jinan2500_counterfactual_epoch_full_online_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/online_tests/$RUN_ID}"
GPUS_RAW="${GPUS:-0 1 2 3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
MAX_LENGTH="${MAX_LENGTH:-64}"
ARMS=(frozen_c30 responsive_h30 responsive_h120)
EPOCHS=(1 2 3 4 5)

read -r -a GPUS <<< "$GPUS_RAW"
read -r -a SEEDS <<< "$SEEDS_RAW"
(( ${#GPUS[@]} > 0 )) || { echo "No GPUs supplied" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
[[ -f "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real_2500.json" ]] || exit 1

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
cd "$REPO_ROOT"

RUN_ROOT="$RUN_ROOT" SOURCE_RUN="$SOURCE_RUN" RUN_COUNTS="$RUN_COUNTS" MAX_LENGTH="$MAX_LENGTH" GPUS_RAW="$GPUS_RAW" SEEDS_RAW="$SEEDS_RAW" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone

manifest = {
    "schema": "jinan2500_counterfactual_epoch_full_online_v1",
    "source_run": os.environ["SOURCE_RUN"],
    "flow": "anon_3_4_jinan_real_2500.json",
    "scope": "same_flow_full_online",
    "run_counts": int(os.environ["RUN_COUNTS"]),
    "max_length": int(os.environ["MAX_LENGTH"]),
    "arms": ["frozen_c30", "responsive_h30", "responsive_h120"],
    "epochs": [1, 2, 3, 4, 5],
    "seeds": [int(value) for value in os.environ["SEEDS_RAW"].split()],
    "gpus": [int(value) for value in os.environ["GPUS_RAW"].split()],
    "phase_override": False,
    "deterministic": True,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
with open(os.path.join(os.environ["RUN_ROOT"], "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
PY

jobs=()
for arm in "${ARMS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="$SOURCE_RUN/training/$arm/epoch_${epoch}.pt"
        [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
        for seed in "${SEEDS[@]}"; do
            jobs+=("$arm:$epoch:$seed")
        done
    done
done
printf 'total_jobs=%s\n' "${#jobs[@]}" > "$RUN_ROOT/status.txt"

run_job() {
    local arm="$1" epoch="$2" seed="$3" gpu="$4"
    local checkpoint="$SOURCE_RUN/training/$arm/epoch_${epoch}.pt"
    local output="$RUN_ROOT/results/$arm/epoch_${epoch}/seed_${seed}"
    local result="$output/result.json"
    mkdir -p "$output"
    if [[ -s "$result" ]]; then
        echo "skip existing $arm epoch=$epoch seed=$seed"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
        --memo "$RUN_ID/$arm/epoch_${epoch}/seed_${seed}" --projection_ckpt "$checkpoint" --llm_path "$MODEL_PATH" \
        --dataset jinan --traffic_file anon_3_4_jinan_real_2500.json --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length "$MAX_LENGTH" \
        --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic \
        --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
        --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
        --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
        --result_json "$result" > "$RUN_ROOT/logs/${arm}_epoch${epoch}_seed${seed}.log" 2>&1
}

failed=0
for ((start=0; start<${#jobs[@]}; start+=${#GPUS[@]})); do
    pids=()
    for ((offset=0; offset<${#GPUS[@]} && start+offset<${#jobs[@]}; offset++)); do
        IFS=: read -r arm epoch seed <<< "${jobs[$((start+offset))]}"
        run_job "$arm" "$epoch" "$seed" "${GPUS[$offset]}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
done
(( failed == 0 )) || { echo "one or more online jobs failed" >&2; exit 1; }

RUN_ROOT="$RUN_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
metrics = {
    "AWT": "test_avg_waiting_time_over",
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "queued": "test_queuing_vehicle_num_over",
    "reward": "test_reward_over",
    "p95_wait": "test_vehicle_wait_p95_over",
    "max_wait": "test_vehicle_wait_max_over",
}
summary = []
for result in sorted(root.glob("results/*/epoch_*/seed_*/result.json")):
    payload = json.loads(result.read_text())
    values = payload.get("results", payload)
    parts = result.parts
    summary.append({
        "arm": parts[-4], "epoch": int(parts[-3].split("_")[1]), "seed": int(parts[-2].split("_")[1]),
        **{name: float(values[key]) for name, key in metrics.items()},
        "path": str(result),
    })
groups = {}
for row in summary:
    groups.setdefault((row["arm"], row["epoch"]), []).append(row)
aggregate = []
for (arm, epoch), rows in sorted(groups.items()):
    aggregate.append({
        "arm": arm, "epoch": epoch, "n": len(rows),
        **{name: statistics.fmean(row[name] for row in rows) for name in metrics},
        **{f"{name}_std": statistics.pstdev(row[name] for row in rows) if len(rows) > 1 else 0.0 for name in metrics},
    })
with open(root / "aggregate.json", "w", encoding="utf-8") as handle:
    json.dump({"per_seed": summary, "by_checkpoint": aggregate}, handle, indent=2)
best = min(aggregate, key=lambda row: (row["AWT"], row["ATT"], row["AQL"]))
with open(root / "status.txt", "a", encoding="utf-8") as handle:
    handle.write(f"completed_jobs={len(summary)}\n")
    handle.write(f"best_by_awt={best['arm']}:epoch_{best['epoch']} AWT={best['AWT']:.6f}\n")
PY

echo "completed: $RUN_ROOT"
