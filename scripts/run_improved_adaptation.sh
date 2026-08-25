#!/bin/bash
# Stage 3 Improved: awt=2.0 relabeling + cost ranking loss
# Uses balanced cost weights from Stage 2 README findings
set -euo pipefail

CONDA_PYTHON="/home/data/sysu/anaconda3/envs/hnr5090/bin/python"
REPO_ROOT="/home/code/LLMSTSC"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="records/pipeline_runs/stage3_improved_awt2.0_rank_${TIMESTAMP}"

cd "${REPO_ROOT}"

echo "============================================================"
echo "Stage 3 Improved: awt=2.0 + cost ranking (K=8,32,120)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
mkdir -p "${OUTPUT_DIR}"

${CONDA_PYTHON} -c "
import sys, os, runpy
sys.path.insert(len(sys.path), '${REPO_ROOT}')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('USE_TF', '0')

sys.argv = [
    'run_stage3_cross_network_adaptation.py',
    '--source_dataset', 'jinan',
    '--target_dataset', 'hangzhou',
    '--source_stage2_ckpt', 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt',
    '--source_stage1_ckpt', 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt',
    '--collection_flow', 'anon_4_4_hangzhou_real.json',
    '--test_flow', 'anon_4_4_hangzhou_real_5816.json',
    '--label_budgets', '8', '32', '120',
    '--seeds', '2024',
    '--train_gpus', '0,1',
    '--output_dir', '${OUTPUT_DIR}',
    '--allow_ruleonly_fallback',
    # Improved relabeling: awt=2.0 per Stage 2 README findings
    '--cost_awt_weight', '2.0',
    '--cost_aql_weight', '1.0',
    '--cost_att_weight', '1.0',
    # Cost ranking loss (Stage 2 proven config)
    '--ranking_loss_weight', '0.05',
    '--soft_target_alpha', '0.3',
]
runpy.run_path(os.path.join('${REPO_ROOT}', 'scripts/run_stage3_cross_network_adaptation.py'), run_name='__main__')
" 2>&1 | tee "${OUTPUT_DIR}/pipeline.log"

echo "Exit: ${PIPESTATUS[0]}"
