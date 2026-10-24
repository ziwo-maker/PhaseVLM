#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
TORCHRUN=${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
DATA=$REPO/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
OUT=$REPO/recordlocal/runs/stage1_data_efficiency_s2024_$(date +%Y%m%d_%H%M%S)
GPUS=${GPUS:-0,1,2,3}
SEED=2024
mkdir -p $OUT/logs
# Data-efficiency protocol: use identical Stage-II settings, only reduce policy supervision.
# max_samples is used as a deterministic proxy for 10/25/50/100 percent subsets.
# The exact sample counts are recorded for reproduction.
for ratio in 10 25 50 100; do
  for arm in full without_stage1; do
    d=$OUT/${ratio}pct/$arm
    mkdir -p $d
    if [ "$ratio" = 10 ]; then n=1000; elif [ "$ratio" = 25 ]; then n=2500; elif [ "$ratio" = 50 ]; then n=5000; else n=0; fi
    init="--init_mode random"
    if [ "$arm" = full ]; then
      init="--init_mode align --align_ckpt $REPO/recordlocal/runs/stage1_convergence_jinan1_s2024_20261009_202803/full/stage1/best.pt"
    fi
    extra=""
    if [ $n -gt 0 ]; then extra="--max_samples $n"; fi
    CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --standalone --nproc_per_node=4 projection_train/train_projection_bc.py \
      --train_data_path $DATA --llm_path $MODEL $init --output_dir $d \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
      --batch_size 1 --grad_accum_steps 8 --epochs 15 --seed $SEED --bf16 \
      --best_metric val_top1_cost_regret --best_mode min $extra \
      > $OUT/logs/${ratio}_${arm}.log 2>&1
  done
done
