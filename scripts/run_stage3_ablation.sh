#!/bin/bash
# Stage 3 Cross-Network Adaptation: Full Ablation Matrix (A-D)
#
# Implements the causal ablation protocol from:
#   README/stage3_cross_network_adaptation_20260810.md
#
# Ablation matrix:
#   A: RuleOnly + Hard BC (baseline)
#   B: Full SimExpert + Hard BC (label quality)
#   C: Full SimExpert + Soft counterfactual preference distillation
#   D: Full SimExpert + Preference distillation + Pareto constraints
#   E: (DEFERRED) D + Topology adapter + meta-training
#
# Usage:
#   bash scripts/run_stage3_ablation.sh [--test] [--subset A,B,C]
#
# Set --test for a quick smoke test with K=8 only.
# Set --subset to run specific ablations (e.g., --subset A,B).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ---- Configuration ----
SOURCE_DATASET="jinan"
TARGET_DATASET="hangzhou"
SOURCE_STAGE2_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt"
SOURCE_STAGE1_CKPT="records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
LLM_PATH="/home/data/model/qwen-vl-8B"
TRAIN_GPUS="0,1"
COLLECTION_FLOW="anon_4_4_hangzhou_real.json"
TEST_FLOW="anon_4_4_hangzhou_real_5816.json"
LABEL_BUDGETS=(8 32 120)
SEEDS=(2024 42 123)
SIM_HORIZON=30

# Cost weights for SimExpert (equal weights, constraints handle trade-off)
COST_ATT=1.0
COST_AQL=1.0
COST_AWT=1.0
COST_SPILLBACK=0.2

# Constraint epsilons
AWT_EPSILON=5.0
SPILLBACK_EPSILON=0.05

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="records/pipeline_runs/stage3_ablation_${TIMESTAMP}"

# ---- Mode flags ----
SMOKE_TEST=false
SUBSET=""

for arg in "$@"; do
    case $arg in
        --test) SMOKE_TEST=true ;;
        --subset=*) SUBSET="${arg#*=}" ;;
    esac
done

if $SMOKE_TEST; then
    LABEL_BUDGETS=(8)
    echo "[SMOKE TEST] K=8 only, single seed"
fi

run_ablation() {
    local ABLATION_ID="$1"
    local TARGET_MODE="$2"
    local USE_SIMEXPERT="$3"    # "true" or "false"
    local ENABLE_PARETO="$4"    # "true" or "false"
    local RANKING_WEIGHT="$5"
    local SOFT_ALPHA="$6"
    local BEST_METRIC="$7"

    # Skip if subset specified and this ID not in it
    if [ -n "$SUBSET" ] && [[ ! ",${SUBSET}," =~ ",${ABLATION_ID}," ]]; then
        echo "Skipping ablation ${ABLATION_ID} (not in subset: ${SUBSET})"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo " Ablation ${ABLATION_ID}: target_mode=${TARGET_MODE}"
    echo " SimExpert=${USE_SIMEXPERT} Pareto=${ENABLE_PARETO}"
    echo " ranking=${RANKING_WEIGHT} soft_alpha=${SOFT_ALPHA}"
    echo "============================================================"

    local OUTPUT_DIR="${BASE_OUTPUT}/${ABLATION_ID}"

    local ALLOW_RULEONLY=""
    local DISABLE_PARETO=""
    local DISABLE_CONSTRAINT=""

    if [ "$USE_SIMEXPERT" = "false" ]; then
        ALLOW_RULEONLY="--allow_ruleonly_fallback"
    fi
    if [ "$ENABLE_PARETO" = "false" ]; then
        DISABLE_PARETO="--disable_pareto_filter"
        DISABLE_CONSTRAINT="--disable_constraint_filter"
    fi

    python3 scripts/run_stage3_cross_network_adaptation.py \
        --source_dataset "$SOURCE_DATASET" \
        --target_dataset "$TARGET_DATASET" \
        --source_stage2_ckpt "$SOURCE_STAGE2_CKPT" \
        --source_stage1_ckpt "$SOURCE_STAGE1_CKPT" \
        --output_dir "$OUTPUT_DIR" \
        --collection_flow "$COLLECTION_FLOW" \
        --test_flow "$TEST_FLOW" \
        --label_budgets "${LABEL_BUDGETS[@]}" \
        --llm_path "$LLM_PATH" \
        --train_gpus "$TRAIN_GPUS" \
        --seeds "${SEEDS[@]}" \
        --target_mode "$TARGET_MODE" \
        --ranking_loss_weight "$RANKING_WEIGHT" \
        --soft_target_alpha "$SOFT_ALPHA" \
        --cost_att_weight "$COST_ATT" \
        --cost_aql_weight "$COST_AQL" \
        --cost_awt_weight "$COST_AWT" \
        --cost_spillback_weight "$COST_SPILLBACK" \
        --sim_horizon "$SIM_HORIZON" \
        --awt_constraint_epsilon "$AWT_EPSILON" \
        --spillback_constraint_epsilon "$SPILLBACK_EPSILON" \
        --best_metric "$BEST_METRIC" \
        --ablation_id "$ABLATION_ID" \
        $ALLOW_RULEONLY \
        $DISABLE_PARETO \
        $DISABLE_CONSTRAINT

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[${ABLATION_ID}] COMPLETED successfully"
    else
        echo "[${ABLATION_ID}] FAILED with exit code ${EXIT_CODE}"
    fi
    return $EXIT_CODE
}

# ---- Main ----
echo "============================================================"
echo " Stage 3 Ablation Matrix"
echo " Output: ${BASE_OUTPUT}"
echo " Source: ${SOURCE_DATASET} -> Target: ${TARGET_DATASET}"
echo " Smoke test: ${SMOKE_TEST}"
echo "============================================================"
mkdir -p "$BASE_OUTPUT"

# Save config
cat > "${BASE_OUTPUT}/ablation_config.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "source_dataset": "$SOURCE_DATASET",
    "target_dataset": "$TARGET_DATASET",
    "source_stage2_ckpt": "$SOURCE_STAGE2_CKPT",
    "source_stage1_ckpt": "$SOURCE_STAGE1_CKPT",
    "collection_flow": "$COLLECTION_FLOW",
    "test_flow": "$TEST_FLOW",
    "label_budgets": [${LABEL_BUDGETS[*]}],
    "seeds": [${SEEDS[*]}],
    "sim_horizon": $SIM_HORIZON,
    "cost_weights": {"att": $COST_ATT, "aql": $COST_AQL, "awt": $COST_AWT, "spillback": $COST_SPILLBACK},
    "awt_constraint_epsilon": $AWT_EPSILON,
    "spillback_constraint_epsilon": $SPILLBACK_EPSILON,
    "smoke_test": $SMOKE_TEST
}
EOF

# ---- A: RuleOnly + Hard BC (baseline replication) ----
run_ablation "A" "hard" "false" "false" "0.0" "0.0" "val_acc"

# ---- B: Full SimExpert + Hard BC (label quality) ----
run_ablation "B" "hard" "true" "false" "0.0" "0.0" "val_acc"

# ---- C: Full SimExpert + Soft counterfactual preference distillation ----
run_ablation "C" "soft" "true" "false" "0.0" "0.3" "val_top1_cost_regret"

# ---- D: Full SimExpert + Preference distillation + Pareto constraints ----
run_ablation "D" "mixed" "true" "true" "0.05" "0.3" "val_top1_cost_regret"

# ---- Summary ----
echo ""
echo "============================================================"
echo " Ablation matrix complete."
echo " Results: ${BASE_OUTPUT}"
echo "============================================================"

# Print results summary
echo ""
echo "Results summary:"
for ablation in A B C D; do
    OUTPUT_DIR="${BASE_OUTPUT}/${ablation}"
    if [ -f "${OUTPUT_DIR}/config.json" ]; then
        echo "  [${ablation}] config OK"
        for k in "${LABEL_BUDGETS[@]}"; do
            BEST_PT="${OUTPUT_DIR}/adapted_k${k}/checkpoints/best.pt"
            EVAL_JSON="${OUTPUT_DIR}/adapted_k${k}/evaluation/eval_summary.json"
            if [ -f "$BEST_PT" ]; then
                echo "    K=${k}: best.pt OK"
            fi
            if [ -f "$EVAL_JSON" ]; then
                python3 -c "
import json
with open('${EVAL_JSON}') as f:
    data = json.load(f)
summary = data.get('summary', {})
for metric, stats in summary.items():
    print(f'      {metric}: mean={stats[\"mean\"]} std={stats[\"std\"]} ci=[{stats[\"ci95_low\"]}, {stats[\"ci95_high\"]}]')
"
            fi
        done
    else
        echo "  [${ablation}] NOT FOUND"
    fi
done
