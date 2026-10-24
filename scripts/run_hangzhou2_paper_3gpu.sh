#!/usr/bin/env bash
# Train the Hangzhou-2 source policy for the paper's within-network transfer protocol.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
TORCHRUN=/home/sysu/anaconda3/envs/hnr5090/bin/torchrun
MODEL_PATH=/mnt/data/Qwen3-VL-8B-Instruct
DATA_PATH="$REPO_ROOT/data/processed/hangzhou_real_5816_stage3_queue_wait_rule_gap2_p60_20260531.jsonl"
GPU_LIST=${GPU_LIST:-0,2,3}
RUN_ID=${RUN_ID:-hangzhou2_paper_3gpu_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/$RUN_ID}
STATUS="$RUN_ROOT/status.txt"

on_exit() {
  local code=$?
  if (( code != 0 )); then
    printf 'failed exit_code=%s\n' "$code" > "$STATUS"
  fi
}
trap on_exit EXIT

cd "$REPO_ROOT"
for path in "$PYTHON_BIN" "$TORCHRUN" "$DATA_PATH" train_align.py projection_train/train_projection_bc.py; do
  [[ -s "$path" ]] || { printf 'Missing file: %s\n' "$path" >&2; exit 1; }
done
[[ -d "$MODEL_PATH" ]] || { printf 'Missing model: %s\n' "$MODEL_PATH" >&2; exit 1; }
mkdir -p "$RUN_ROOT/stage1" "$RUN_ROOT/stage2" "$RUN_ROOT/logs"

{
  printf 'run_id=%s\nsource_scenario=Hangzhou-2\ntraining_flow=anon_4_4_hangzhou_real_5816.json\n' "$RUN_ID"
  printf 'target_scenario=Hangzhou-1 (later zero-shot evaluation; no evaluation in this job)\n'
  printf 'dataset=%s\nmodel=%s\ngpus=%s\n' "$DATA_PATH" "$MODEL_PATH" "$GPU_LIST"
  printf 'architecture=seq_len=16,image_size=224,periodicity=16,input_dim=8,hidden_dim=256,prefix_tokens=8,prefix_mode=prefix,max_length=512\n'
  printf 'stage1=temperature=0.05,per_gpu_batch=1,grad_accum=5,effective_global_batch=15,epochs=15,lr=1e-4,bf16\n'
  printf 'stage2=per_gpu_batch=1,grad_accum=11,effective_global_batch=33,epochs=15,lr=1e-4,bf16,target_mode=mixed,alpha=0.3,soft_temperature=0.7,ranking_weight=0.15,base_kl_weight=0.02\n'
  printf 'hardware_adjustment=three GPUs yield effective batches 15 and 33 versus paper 16 and 32; Stage 1 contrastive batch is 3 per forward\n'
  printf 'cost_weights=att:1,aql:1,awt:2,spillback:0.2 (source expert selection)\n'
} > "$RUN_ROOT/manifest.txt"

export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

printf 'running stage1\n' > "$STATUS"
printf '[%s] Stage 1 start, GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_LIST"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=3 train_align.py \
  --data_path "$DATA_PATH" --model_path "$MODEL_PATH" --output_dir "$RUN_ROOT/stage1" --decision_only \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
  --temperature 0.05 --batch_size 1 --grad_accum_steps 5 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 \
  > "$RUN_ROOT/logs/stage1.log" 2>&1
[[ -s "$RUN_ROOT/stage1/best.pt" ]] || { printf 'Stage 1 checkpoint missing\n' >&2; exit 1; }

printf 'running stage2\n' > "$STATUS"
printf '[%s] Stage 2 start, GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_LIST"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node=3 projection_train/train_projection_bc.py \
  --data_path "$DATA_PATH" --llm_path "$MODEL_PATH" --align_ckpt "$RUN_ROOT/stage1/best.pt" --output_dir "$RUN_ROOT/stage2" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 8 --prefix_mode prefix --max_length 512 --prompt_mode queue_wait_short \
  --batch_size 1 --grad_accum_steps 11 --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
  --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
  --ranking_loss_weight 0.15 --base_policy_kl_weight 0.02 --cost_att_weight 1 --cost_aql_weight 1 --cost_awt_weight 2 --cost_spillback_weight 0.2 \
  --best_metric val_top1_cost_regret --best_mode min --save_every 1 --ddp_timeout_seconds 1800 \
  > "$RUN_ROOT/logs/stage2.log" 2>&1
[[ -s "$RUN_ROOT/stage2/best.pt" ]] || { printf 'Stage 2 checkpoint missing\n' >&2; exit 1; }

printf 'completed\n' > "$STATUS"
printf '[%s] Stage 1/2 completed\n' "$(date --iso-8601=seconds)"
