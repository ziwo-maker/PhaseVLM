#!/bin/bash
# Stage 3: Cross-Network Few-Shot Adaptation
#
# Implements the protocol recommended by the external reviewer:
#   1. Source training (Jinan Stage 1+2) — already done
#   2. Zero-shot transfer to Hangzhou — standard CityFlowEnv
#   3. Collect target-network states
#   4. Relabel with SimExpert
#   5. Few-shot fine-tune prefix/TS2I params
#   6. Multi-seed evaluation on held-out test flow
#
# This replaces the old PPO-based Stage 3 which had an environment mismatch
# and produced no improvement over BC on the standard benchmark.
#
# Usage:
#   bash scripts/run_stage3_cross_network_adaptation.sh
#
# Key differences from old Stage 3:
#   - ALL evaluation uses standard CityFlowEnv (NO TrajectoryCityFlowEnv)
#   - No PPO / reinforcement learning
#   - Supervised fine-tuning on SimExpert-labeled target states
#   - Cross-network transfer focus (Jinan ↔ Hangzhou)
#   - Multi-seed evaluation with correct t-intervals

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================
# Configuration
# ============================================================

# Source network
SOURCE_DATASET="${SOURCE_DATASET:-jinan}"
SOURCE_STAGE2_CKPT="${SOURCE_STAGE2_CKPT:-}"
SOURCE_STAGE1_CKPT="${SOURCE_STAGE1_CKPT:-}"

# Target network
TARGET_DATASET="${TARGET_DATASET:-hangzhou}"

# Flow files (must differ!)
COLLECTION_FLOW="${COLLECTION_FLOW:-anon_4_4_hangzhou_real.json}"
TEST_FLOW="${TEST_FLOW:-anon_4_4_hangzhou_real_5816.json}"

# Adaptation parameters
NUM_LABELS="${NUM_LABELS:-128}"
LABEL_BUDGETS="${LABEL_BUDGETS:-0 32 128 512}"
SEEDS="${SEEDS:-2024 42 123}"

# Hardware
LLM_PATH="${LLM_PATH:-/home/data/model/qwen-vl-8B}"
COLLECT_GPU="${COLLECT_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-2}"

# Output
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-records/pipeline_runs/stage3_cross_network_${SOURCE_DATASET}2${TARGET_DATASET}_${TIMESTAMP}}"

# ============================================================
# Validation
# ============================================================

if [ -z "${SOURCE_STAGE2_CKPT}" ]; then
    echo "ERROR: SOURCE_STAGE2_CKPT is required"
    echo "Set it to the Stage 2 best.pt trained on ${SOURCE_DATASET}"
    exit 1
fi

if [ -z "${SOURCE_STAGE1_CKPT}" ]; then
    echo "ERROR: SOURCE_STAGE1_CKPT is required"
    echo "Set it to the Stage 1 best.pt for ${SOURCE_DATASET}"
    exit 1
fi

if [ ! -f "${SOURCE_STAGE2_CKPT}" ]; then
    echo "ERROR: Source Stage 2 checkpoint not found: ${SOURCE_STAGE2_CKPT}"
    exit 1
fi

if [ ! -f "${SOURCE_STAGE1_CKPT}" ]; then
    echo "ERROR: Source Stage 1 checkpoint not found: ${SOURCE_STAGE1_CKPT}"
    exit 1
fi

echo "============================================================"
echo "Stage 3: Cross-Network Few-Shot Adaptation"
echo "============================================================"
echo "Source: ${SOURCE_DATASET} -> Target: ${TARGET_DATASET}"
echo "Collection flow: ${COLLECTION_FLOW}"
echo "Test flow: ${TEST_FLOW}"
echo "Label budgets: ${LABEL_BUDGETS}"
echo "Seeds: ${SEEDS}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# Save run config
cat > "${OUTPUT_DIR}/run_config.json" << EOF
{
    "source_dataset": "${SOURCE_DATASET}",
    "target_dataset": "${TARGET_DATASET}",
    "source_stage2_ckpt": "${SOURCE_STAGE2_CKPT}",
    "source_stage1_ckpt": "${SOURCE_STAGE1_CKPT}",
    "collection_flow": "${COLLECTION_FLOW}",
    "test_flow": "${TEST_FLOW}",
    "num_labels": ${NUM_LABELS},
    "label_budgets": [$(echo "${LABEL_BUDGETS}" | sed 's/ /, /g')],
    "seeds": [$(echo "${SEEDS}" | sed 's/ /, /g')],
    "timestamp": "$(date -Iseconds)",
    "llm_path": "${LLM_PATH}"
}
EOF

# ============================================================
# Execute Pipeline
# ============================================================

CUDA_VISIBLE_DEVICES="${COLLECT_GPU}" python scripts/run_stage3_cross_network_adaptation.py \
    --source_dataset "${SOURCE_DATASET}" \
    --target_dataset "${TARGET_DATASET}" \
    --source_stage2_ckpt "${SOURCE_STAGE2_CKPT}" \
    --source_stage1_ckpt "${SOURCE_STAGE1_CKPT}" \
    --collection_flow "${COLLECTION_FLOW}" \
    --test_flow "${TEST_FLOW}" \
    --label_budgets ${LABEL_BUDGETS} \
    --seeds ${SEEDS} \
    --llm_path "${LLM_PATH}" \
    --train_gpus "${TRAIN_GPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${SMOKE_TEST:+--allow_ruleonly_fallback}

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Stage 3 pipeline completed successfully."
    echo "Results: ${OUTPUT_DIR}"
else
    echo "Stage 3 pipeline failed with exit code ${EXIT_CODE}"
    echo "Check logs in: ${OUTPUT_DIR}"
fi
echo "============================================================"

exit ${EXIT_CODE}
