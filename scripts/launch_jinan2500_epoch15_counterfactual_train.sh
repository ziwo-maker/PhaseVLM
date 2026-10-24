#!/usr/bin/env bash
# Same-flow counterfactual label validation for an archived Jinan-2500 policy.
# Produces data and trained checkpoints only; it deliberately performs no online evaluation.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO_ROOT/online_test_best_weights/anon_3_4_jinan_real_2500/jinan2500_repro_4gpu_epoch15_seed2024.pt}"
ALIGN_CKPT="${ALIGN_CKPT:-$REPO_ROOT/records/repro/jinan2500_repro_4gpu_20260904_005005/stage1/best.pt}"
MAX_LENGTH="${MAX_LENGTH:-64}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-}"
RUN_ID="${RUN_ID:-jinan2500_epoch15_counterfactual_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/counterfactual_validation/$RUN_ID}"
FLOW="anon_3_4_jinan_real_2500.json"
FLOW_PATH="$REPO_ROOT/data/Jinan/3_4/$FLOW"
ROADNET="$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json"

COLLECT_GPU="${COLLECT_GPU:-0}"
H30_GPU="${H30_GPU:-0}"
H120_GPUS="${H120_GPUS:-1 2}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/collect"
cd "$REPO_ROOT"

require_file() { [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }
require_file "$PYTHON_BIN"
require_file "$SOURCE_CKPT"
require_file "$ALIGN_CKPT"
require_file "$FLOW_PATH"
require_file "$ROADNET"
require_dir "$MODEL_PATH"

SOURCE_SHA256="$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')"
[[ -z "$EXPECTED_SOURCE_SHA256" || "$SOURCE_SHA256" == "$EXPECTED_SOURCE_SHA256" ]] || {
    echo "Unexpected source checkpoint checksum: $SOURCE_SHA256 (expected $EXPECTED_SOURCE_SHA256)" >&2
    exit 1
}

RUN_ROOT="$RUN_ROOT" FLOW="$FLOW" SOURCE_CKPT="$SOURCE_CKPT" SOURCE_SHA256="$SOURCE_SHA256" ALIGN_CKPT="$ALIGN_CKPT" MAX_LENGTH="$MAX_LENGTH" \
    "$PYTHON_BIN" - <<'PY'
import json
import os

manifest = {
    "schema": "jinan2500_epoch15_counterfactual_validation_v1",
    "flow": os.environ["FLOW"],
    "scope": "same_flow",
    "source_checkpoint": os.environ["SOURCE_CKPT"],
    "source_checkpoint_sha256": os.environ["SOURCE_SHA256"],
    "align_checkpoint": os.environ["ALIGN_CKPT"],
    "architecture": {"seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "hidden_dim": 64, "prefix_tokens": 8, "prefix_mode": "prefix", "max_length": int(os.environ["MAX_LENGTH"])},
    "collection": {"seed": 2024, "run_counts": 3600},
    "teachers": ["frozen_c30", "responsive_h30", "responsive_h120"],
    "training": {"epochs": 5, "seed": 2024, "online_evaluation": False},
}
with open(os.path.join(os.environ["RUN_ROOT"], "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
PY

COLLECTED="$RUN_ROOT/collect/collected_states.jsonl"
if [[ ! -s "$COLLECTED" ]]; then
    echo "[$(date '+%F %T')] collecting same-flow decision states"
    CUDA_VISIBLE_DEVICES="$COLLECT_GPU" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
        --memo "$RUN_ID/collect" --projection_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" \
        --dataset jinan --traffic_file "$FLOW" --run_counts 3600 --num_rounds 1 --seed 2024 \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length "$MAX_LENGTH" \
        --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 \
        --collect_states_output "$COLLECTED" > "$RUN_ROOT/logs/collect.log" 2>&1
fi

"$PYTHON_BIN" - "$COLLECTED" <<'PY'
import json
import os
import sys

records = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))
snapshots = {str(row.get("snapshot_path", "")) for row in records}
snapshots.discard("")
bad = []
for path in sorted(snapshots):
    try:
        with open(path, encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError):
        bad.append(path)
if bad:
    bad_set = set(bad)
    retained = [row for row in records if str(row.get("snapshot_path", "")) not in bad_set]
    backup = sys.argv[1] + ".unfiltered"
    if not os.path.exists(backup):
        os.replace(sys.argv[1], backup)
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    records = retained
    snapshots -= bad_set
print(f"collected_records={len(records)} valid_snapshots={len(snapshots)} filtered_snapshots={len(bad)}")
if len(snapshots) < 32:
    raise SystemExit("Collected data is insufficient for counterfactual training (need at least 32 snapshots)")
PY

FROZEN_LABEL="$RUN_ROOT/relabeled_simexpert_c30.jsonl"
H30_LABEL="$RUN_ROOT/relabeled_policy_responsive_h30.jsonl"
H120_LABEL="$RUN_ROOT/relabeled_policy_responsive_h120.jsonl"

if [[ ! -s "$FROZEN_LABEL" ]]; then
    ( HORIZON=30 OUTPUT_DIR="$RUN_ROOT" TRAFFIC_FILE="$FLOW_PATH" \
      bash scripts/run_jinan_real_relabel_frozen_c30.sh > "$RUN_ROOT/logs/relabel_frozen_c30.log" 2>&1 ) &
    P_FROZEN=$!
else P_FROZEN=""; fi

if [[ ! -s "$H30_LABEL" ]]; then
    ( CUDA_VISIBLE_DEVICES="$H30_GPU" "$PYTHON_BIN" -u scripts/relabel_policy_responsive.py \
      --input "$COLLECTED" --output "$H30_LABEL" --roadnet "$ROADNET" --traffic_file "$FLOW_PATH" \
      --dataset jinan --num_rows 3 --num_cols 4 --source_policy_ckpt "$SOURCE_CKPT" --llm_path "$MODEL_PATH" \
      --work_dir "$RUN_ROOT/runtime_h30" --sim_horizon 30 --min_action_time 30 --traffic_series_interval 2 \
      --cost_scope global --inference_batch_size 16 --bf16 --seq_len 16 --image_size 224 --periodicity 16 \
      --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length "$MAX_LENGTH" \
      > "$RUN_ROOT/logs/relabel_h30.log" 2>&1 ) &
    P_H30=$!
else P_H30=""; fi

if [[ ! -s "$H120_LABEL" ]]; then
    ( OUTPUT_DIR="$RUN_ROOT" SOURCE_CKPT="$SOURCE_CKPT" TRAFFIC_FILE="$FLOW_PATH" HORIZON=120 \
      NUM_SHARDS=2 GPUS="$H120_GPUS" bash scripts/run_jinan_real_relabel_parallel.sh \
      > "$RUN_ROOT/logs/relabel_h120.log" 2>&1 ) &
    P_H120=$!
else P_H120=""; fi

rc=0
for pid in "$P_FROZEN" "$P_H30" "$P_H120"; do
    [[ -z "$pid" ]] || wait "$pid" || rc=1
done
(( rc == 0 )) || { echo "Counterfactual relabeling failed; inspect $RUN_ROOT/logs" >&2; exit "$rc"; }
require_file "$FROZEN_LABEL"; require_file "$H30_LABEL"; require_file "$H120_LABEL"

read -r -a TRAIN_GPU_ARR <<< "$TRAIN_GPUS"
(( ${#TRAIN_GPU_ARR[@]} >= 3 )) || { echo "TRAIN_GPUS must specify three GPUs" >&2; exit 1; }
train_arm() {
    local gpu="$1" label="$2" arm="$3"
    local out="$RUN_ROOT/training/$arm"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/train_projection_bc.py \
        --data_path "$label" --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --init_policy_ckpt "$SOURCE_CKPT" \
        --output_dir "$out" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length "$MAX_LENGTH" --batch_size 2 --grad_accum_steps 8 \
        --epochs 5 --lr 1e-5 --min_lr 1e-6 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 \
        --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 --num_workers 4 \
        --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --best_metric val_top1_cost_regret \
        --best_mode min --log_interval 20 --save_every 1 --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 \
        > "$RUN_ROOT/logs/train_${arm}.log" 2>&1
}

train_arm "${TRAIN_GPU_ARR[0]}" "$FROZEN_LABEL" frozen_c30 & P0=$!
train_arm "${TRAIN_GPU_ARR[1]}" "$H30_LABEL" responsive_h30 & P1=$!
train_arm "${TRAIN_GPU_ARR[2]}" "$H120_LABEL" responsive_h120 & P2=$!
rc=0
for pid in "$P0" "$P1" "$P2"; do wait "$pid" || rc=1; done
(( rc == 0 )) || { echo "One or more training arms failed; inspect $RUN_ROOT/logs" >&2; exit "$rc"; }
for arm in frozen_c30 responsive_h30 responsive_h120; do require_file "$RUN_ROOT/training/$arm/best.pt"; done
echo "[$(date '+%F %T')] complete: datasets and trained checkpoints are in $RUN_ROOT"
