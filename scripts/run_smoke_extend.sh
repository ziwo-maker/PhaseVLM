#!/bin/bash
# Extend smoke test to K=32, K=120
set -euo pipefail

CONDA_PYTHON="/home/data/sysu/anaconda3/envs/hnr5090/bin/python"
REPO_ROOT="/home/code/LLMSTSC"
OUTPUT_DIR="records/pipeline_runs/stage3_smoke_jinan2hangzhou_k8_20260810"

cd "${REPO_ROOT}"

echo "============================================================"
echo "Stage 3 Extend: K=32, K=120"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

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
    '--label_budgets', '32', '120',
    '--seeds', '2024',
    '--train_gpus', '0,1',
    '--output_dir', '${OUTPUT_DIR}',
    '--allow_ruleonly_fallback',
]
runpy.run_path(os.path.join('${REPO_ROOT}', 'scripts/run_stage3_cross_network_adaptation.py'), run_name='__main__')
" 2>&1 | tee -a "${OUTPUT_DIR}/pipeline_extend.log"

echo "Exit: ${PIPESTATUS[0]}"
