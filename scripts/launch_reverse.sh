#!/bin/bash
source /home/sysu/anaconda3/etc/profile.d/conda.sh
conda activate hnr5090
cd /home/code/LLMSTSC
exec bash scripts/run_stage3_reverse.sh
