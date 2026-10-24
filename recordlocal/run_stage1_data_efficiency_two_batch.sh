#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
TORCHRUN=${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
DATA=$REPO/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
SEED=2024
OUT=$REPO/recordlocal/runs/stage1_data_efficiency_s2024_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/logs"

# Two-batch execution avoids loading four Qwen3-VL-8B models simultaneously.
# Each job uses two GPUs with DDP, one seed only.

run_one() {
  local ratio=$1
  local arm=$2
  local gpus=$3
  local d="$OUT/${ratio}pct/${arm}"
  mkdir -p "$d"

  local n=0
  case "$ratio" in
    10) n=1000 ;;
    25) n=2500 ;;
    50) n=5000 ;;
    100) n=0 ;;
  esac

  local init="--init_mode random"
  if [ "$arm" = "full" ]; then
    init="--init_mode align --align_ckpt $REPO/recordlocal/runs/stage1_convergence_jinan1_s2024_20261010_155953/full/stage1/best.pt"
  fi

  local extra=""
  if [ "$n" -gt 0 ]; then
    extra="--max_samples $n"
  fi

  CUDA_VISIBLE_DEVICES="$gpus" "$TORCHRUN" --standalone --nproc_per_node=2 \
    projection_train/train_projection_bc.py \
    --train_data_path "$DATA" --llm_path "$MODEL" $init \
    --output_dir "$d" \
    --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --batch_size 1 --grad_accum_steps 8 \
    --epochs 15 --seed "$SEED" --bf16 \
    --best_metric val_top1_cost_regret --best_mode min $extra \
    > "$OUT/logs/${ratio}_${arm}.log" 2>&1
}

# Batch 1: low supervision settings
run_one 10 full 0,1 &
run_one 10 without_stage1 2,3 &
wait

# Batch 2
run_one 25 full 0,1 &
run_one 25 without_stage1 2,3 &
wait

run_one 50 full 0,1 &
run_one 50 without_stage1 2,3 &
wait

run_one 100 full 0,1 &
run_one 100 without_stage1 2,3 &
wait

"$PYTHON" "$REPO/recordlocal/stage1_efficiency_helper.py" summarize --root "$OUT" || true

echo "completed: $OUT"
