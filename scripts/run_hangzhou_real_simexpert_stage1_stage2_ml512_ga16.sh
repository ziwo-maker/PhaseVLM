#!/usr/bin/env bash
# Train Stage 1 then Stage 2 on the selected Hangzhou-real SimExpert run.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SOURCE_RUN="${SOURCE_RUN:-/mnt/data/LLMSTSCrecords/records/sim_expert_validation_search/hangzhou_real_simexpert_small_sweep_20260907_dependency_retry03}"
DATA_PATH="${DATA_PATH:-$SOURCE_RUN/best/traffic_prompt_series_2s.jsonl}"
RUN_ID="${RUN_ID:-hangzhou_real_simexpert_h30_att1_aql1_awt1_spill0p2_stage12_ml512_ga16_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/$RUN_ID}"
STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2"
LOG_DIR="$RUN_ROOT/logs"
MANIFEST="$RUN_ROOT/manifest.env"
STATUS="$RUN_ROOT/status.txt"

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
require_file() { [[ -s "$1" ]] || { log "Missing required file: $1"; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { log "Missing required directory: $1"; exit 1; }; }

record_exit() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    printf 'failed exit_code=%s ended_at=%s\n' "$exit_code" "$(date --iso-8601=seconds)" > "$STATUS"
  fi
}
trap record_exit EXIT

write_manifest() {
  umask 077
  {
    printf 'run_id=%q\n' "$RUN_ID"
    printf 'run_root=%q\n' "$RUN_ROOT"
    printf 'source_run=%q\n' "$SOURCE_RUN"
    printf 'source_data=%q\n' "$DATA_PATH"
    printf 'source_data_sha256=%q\n' "$(sha256sum "$DATA_PATH" | awk '{print $1}')"
    printf 'model_path=%q\n' "$MODEL_PATH"
    printf 'gpu_list=%q\n' "$GPU_LIST"
    printf 'training_flow=%q\n' 'anon_4_4_hangzhou_real.json'
    printf 'testing_flow=%q\n' 'anon_4_4_hangzhou_real.json (same-flow online evaluation is not part of this training launch)'
    printf 'sim_expert_selection=%q\n' 'horizon_s=30,att=1.0,aql=1.0,awt=1.0,spillback=0.2,selection_objective=weighted_minmax_full_run_metrics'
    printf 'architecture=%q\n' 'seq_len=16,image_size=224,periodicity=16,input_dim=8,hidden_dim=256,prefix_tokens=32,prefix_mode=prefix,max_length=512'
    printf 'stage1=%q\n' 'batch_size=1,grad_accum_steps=4,epochs=15,lr=1e-4,min_lr=1e-5,warmup_ratio=0.1,lr_schedule=cosine,seed=2024,bf16'
    printf 'stage2=%q\n' 'batch_size=1,grad_accum_steps=16,epochs=15,lr=1e-4,min_lr=1e-5,warmup_ratio=0.1,lr_schedule=cosine,seed=2024,bf16,target_mode=mixed,ranking_loss_weight=0.5,cost_att=1.0,cost_aql=1.0,cost_awt=1.0,cost_spillback=0.2'
    printf 'stage2_global_effective_batch=%q\n' "$((NPROC_PER_NODE * 16)) (ranks=$NPROC_PER_NODE x per-rank batch=1 x grad accumulation=16)"
  } > "$MANIFEST"
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN"
  require_file "$REPO_ROOT/train_align.py"
  require_file "$REPO_ROOT/projection_train/train_projection_bc.py"
  require_file "$DATA_PATH"
  require_dir "$MODEL_PATH"
  [[ "$NPROC_PER_NODE" -eq 4 ]] || { log 'NPROC_PER_NODE must be 4 for this multi-GPU launch'; exit 2; }
  [[ "$GPU_LIST" == *,*,* ]] || { log 'GPU_LIST must provide four CUDA devices'; exit 2; }
  [[ ! -e "$RUN_ROOT" ]] || { log "Refusing to reuse existing RUN_ROOT: $RUN_ROOT"; exit 2; }

  mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$LOG_DIR"
  write_manifest
  printf 'running stage1 started_at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS"
  export PYTHONPATH="$REPO_ROOT"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=4
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  log "Stage 1 start: Hangzhou real SimExpert h30 selected data, 4 GPUs, max_length=512"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_align.py \
    --data_path "$DATA_PATH" --model_path "$MODEL_PATH" --output_dir "$STAGE1_DIR" --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --temperature 0.07 --batch_size 1 --grad_accum_steps 4 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 \
    > "$LOG_DIR/stage1.log" 2>&1
  require_file "$STAGE1_DIR/best.pt"

  printf 'running stage2 started_at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS"
  log "Stage 2 start: 4 GPUs, max_length=512, per-rank batch=1, grad accumulation=16"
  timeout --foreground 21600s env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" projection_train/train_projection_bc.py \
    --data_path "$DATA_PATH" --llm_path "$MODEL_PATH" --align_ckpt "$STAGE1_DIR/best.pt" --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 --prompt_mode queue_wait_short \
    --batch_size 1 --grad_accum_steps 16 --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 --target_mode mixed --ranking_loss_weight 0.5 \
    --cost_att_weight 1.0 --cost_aql_weight 1.0 --cost_awt_weight 1.0 --cost_spillback_weight 0.2 \
    --best_metric val_top1_cost_regret --best_mode min --enable_class_balance --overconf_penalty_weight 0.1 --save_every 3 --ddp_timeout_seconds 1800 \
    > "$LOG_DIR/stage2.log" 2>&1
  require_file "$STAGE2_DIR/best.pt"
  printf 'completed ended_at=%s\n' "$(date --iso-8601=seconds)" > "$STATUS"
  log 'Stage 1/2 completed'
}

main "$@"
