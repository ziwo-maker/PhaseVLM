#!/usr/bin/env bash
# Multi-seed evaluation script for Stage 2/Stage 3 checkpoints.
#
# Runs N independent online tests with different random seeds on the same
# traffic flow, collecting ATT, AQL, AWT, p95 wait, and left-turn wait.
# Outputs a summary JSON with mean ± std across seeds.
#
# Usage:
#   bash scripts/run_multiseed_eval.sh \
#     --ckpt records/.../best.pt \
#     --dataset jinan \
#     --traffic_file anon_3_4_jinan_real_2500.json \
#     --seeds 5 \
#     --output_dir records/multiseed_eval/jinan_stage3_default
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Argument parsing ───────────────────────────────────────────────
CKPT=""
DATASET="jinan"
TRAFFIC_FILE=""
SEEDS=5
OUTPUT_DIR=""
LLM_PATH="/home/data/model/qwen-vl-8B"
GPU=0
PROMPT_MODE="queue_wait_short"
RUN_COUNTS=3600

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --traffic_file) TRAFFIC_FILE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --llm_path) LLM_PATH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --prompt_mode) PROMPT_MODE="$2"; shift 2 ;;
    --run_counts) RUN_COUNTS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${CKPT}" || -z "${TRAFFIC_FILE}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 --ckpt <path> --traffic_file <name> --output_dir <dir> [--seeds 5]"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PY_BIN="${PY_BIN:-python}"
echo "=== Multi-seed evaluation ==="
echo "Checkpoint: ${CKPT}"
echo "Dataset:    ${DATASET}"
echo "Flow:       ${TRAFFIC_FILE}"
echo "Seeds:      ${SEEDS}"
echo "Output:     ${OUTPUT_DIR}"

RESULTS_JSON="${OUTPUT_DIR}/all_results.json"
declare -a RESULT_PATHS=()

for ((seed=2024; seed<2024+SEEDS; seed++)); do
  SEED_DIR="${OUTPUT_DIR}/seed${seed}"
  mkdir -p "${SEED_DIR}"
  RESULT_FILE="${SEED_DIR}/result.json"
  RESULT_PATHS+=("${RESULT_FILE}")

  echo "[$(date +%H:%M:%S)] Running seed=${seed}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY_BIN}" projection_train/run_projection_online.py \
    --memo "multiseed_eval/$(basename "${OUTPUT_DIR}")/seed${seed}" \
    --projection_ckpt "${CKPT}" \
    --llm_path "${LLM_PATH}" \
    --dataset "${DATASET}" \
    --traffic_file "${TRAFFIC_FILE}" \
    --run_counts "${RUN_COUNTS}" \
    --num_rounds 1 \
    --seed "${seed}" \
    --prompt_mode "${PROMPT_MODE}" \
    --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --inference_batch_size 64 --bf16 \
    --result_json "${RESULT_FILE}" \
    2>&1 | tail -5
done

# ── Aggregate ──────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Aggregating results..."

"${PY_BIN}" -c "
import json, sys, os
from math import sqrt

metrics = ['test_avg_queue_len_over', 'test_avg_waiting_time_over', 'test_avg_travel_time_over']
results = []
for seed in range(2024, 2024 + ${SEEDS}):
    path = '${OUTPUT_DIR}/seed' + str(seed) + '/result.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        r = data.get('results', {})
        results.append(r)

if not results:
    print('No results found!')
    sys.exit(1)

summary = {'n_seeds': len(results)}
for m in metrics:
    values = [float(r[m]) for r in results if m in r]
    if values:
        mean = sum(values) / len(values)
        var = sum((v - mean)**2 for v in values) / (len(values) - 1) if len(values) > 1 else 0
        std = sqrt(var)
        summary[m + '_mean'] = round(mean, 4)
        summary[m + '_std'] = round(std, 4)
        summary[m + '_values'] = [round(v, 4) for v in values]

summary['att_ci95_half'] = round(1.96 * summary.get('test_avg_travel_time_over_std', 0), 4)

with open('${RESULTS_JSON}', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
"

echo "[$(date +%H:%M:%S)] Done. Results: ${RESULTS_JSON}"
