#!/bin/bash
# Stage 3 Smoke Test Launcher
# Uses hnr5090 conda environment (Python 3.10 + cityflow + torch)
set -euo pipefail

CONDA_PYTHON="/home/data/sysu/anaconda3/envs/hnr5090/bin/python"
REPO_ROOT="/home/code/LLMSTSC"

# Fix PYTHONPATH: exclude repo root to prevent local cityflow/ from shadowing
# the installed cityflow package. We still need repo in path for imports.
export PYTHONPATH="${REPO_ROOT}"
# But remove it if it's already there
unset PYTHONPATH
# Instead, add repo to sys.path at the END (after site-packages)
export STAGE3_REPO="${REPO_ROOT}"

cd "${REPO_ROOT}"

OUTPUT_DIR="${1:-records/pipeline_runs/stage3_smoke_jinan2hangzhou_k8_20260810}"

echo "============================================================"
echo "Stage 3 Smoke Test: Jinan -> Hangzhou (K=8, RuleOnly)"
echo "Python: ${CONDA_PYTHON}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

${CONDA_PYTHON} -c "
import sys
# Add repo to path AFTER site-packages so cityflow is found correctly
sys.path.insert(len(sys.path), '${REPO_ROOT}')
# Verify cityflow
import cityflow
assert hasattr(cityflow, 'Engine'), 'cityflow.Engine not found!'
print(f'cityflow OK (Engine available)')
" 2>&1

echo ""
echo "Launching pipeline..."

${CONDA_PYTHON} -c "
import sys, os, runpy
sys.path.insert(len(sys.path), '${REPO_ROOT}')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('USE_TF', '0')

# Set up args and run the pipeline module
sys.argv = [
    'run_stage3_cross_network_adaptation.py',
    '--source_dataset', 'jinan',
    '--target_dataset', 'hangzhou',
    '--source_stage2_ckpt', 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt',
    '--source_stage1_ckpt', 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt',
    '--collection_flow', 'anon_4_4_hangzhou_real.json',
    '--test_flow', 'anon_4_4_hangzhou_real_5816.json',
    '--label_budgets', '0', '8',
    '--seeds', '2024',
    '--train_gpus', '0,1',
    '--output_dir', '${OUTPUT_DIR}',
    '--allow_ruleonly_fallback',
]
runpy.run_path(os.path.join('${REPO_ROOT}', 'scripts/run_stage3_cross_network_adaptation.py'), run_name='__main__')
" 2>&1 | tee "${OUTPUT_DIR}/pipeline.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "============================================================"
echo "Pipeline exit code: ${EXIT_CODE}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
