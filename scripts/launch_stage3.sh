#!/bin/bash
# Wrapper to launch Stage 3 experiment with conda environment
set -euo pipefail
source /home/sysu/anaconda3/etc/profile.d/conda.sh
conda activate hnr5090
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
cd /home/code/LLMSTSC
exec bash scripts/run_stage3_full_experiment.sh
