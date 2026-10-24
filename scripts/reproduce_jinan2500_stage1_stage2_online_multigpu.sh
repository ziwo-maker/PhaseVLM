#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}"
MODEL_PATH="${MODEL_PATH:-/home/data/model/qwen-vl-8B}"
RUN_ID="${RUN_ID:-jinan2500_repro_multigpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/repro/$RUN_ID}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
MIN_GPU_MEM="${MIN_GPU_MEM:-1000}"

DATA1="$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DATA2="$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
TRAFFIC_FILE="anon_3_4_jinan_real_2500.json"
STAGE1="$RUN_ROOT/stage1"
STAGE2="$RUN_ROOT/stage2"
ONLINE="$RUN_ROOT/online"
LOGS="$RUN_ROOT/logs"

mkdir -p "$STAGE1" "$STAGE2" "$ONLINE" "$LOGS"
cd "$REPO_ROOT"

require_file() { [[ -s "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
require_file "$PYTHON_BIN"; require_file "$TORCHRUN"; require_file "$DATA1"; require_file "$DATA2"
require_file "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"
[[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 1; }

# Select idle GPUs unless GPU_LIST is explicitly supplied (comma-separated).
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v lim="$MIN_GPU_MEM" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2+0) < lim) print $1}')
fi
(( ${#GPUS[@]} > 0 )) || { echo "No idle GPU found; set GPU_LIST=0,1,... after checking nvidia-smi" >&2; exit 1; }
GPU_CSV="$(IFS=,; echo "${GPUS[*]}")"
echo "run_root=$RUN_ROOT"
echo "gpus=$GPU_CSV"
echo "seeds=$SEEDS_RAW"

if [[ ! -s "$STAGE1/best.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_CSV" "$TORCHRUN" --standalone --nproc_per_node="${#GPUS[@]}" \
    train_align.py --data_path "$DATA1" "$DATA2" --model_path "$MODEL_PATH" --output_dir "$STAGE1" --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix \
    --temperature 0.07 --batch_size 2 --grad_accum_steps 2 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 \
    --lr_schedule cosine --epochs 15 --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 50 --save_every_epochs 1 --seed 2024 --bf16 2>&1 | tee "$LOGS/stage1.log"
fi
require_file "$STAGE1/best.pt"

if [[ ! -s "$STAGE2/best.pt" || ! -s "$STAGE2/epoch_15.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_CSV" "$TORCHRUN" --standalone --nproc_per_node="${#GPUS[@]}" \
    projection_train/train_projection_bc.py --data_path "$DATA1" "$DATA2" --llm_path "$MODEL_PATH" \
    --align_ckpt "$STAGE1/best.pt" --output_dir "$STAGE2" --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
    --batch_size 2 --grad_accum_steps 8 --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy blocked --validation_purge_samples 16 \
    --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --best_metric val_top1_cost_regret --best_mode min --best_min_delta 0.0 --log_interval 20 --save_every 1 \
    --seed 2024 --bf16 --checkpoint_slim --ddp_timeout_seconds 1800 2>&1 | tee "$LOGS/stage2.log"
fi
require_file "$STAGE2/best.pt"; require_file "$STAGE2/epoch_15.pt"

read -r -a SEEDS_ARR <<< "$SEEDS_RAW"
run_online_batch() {
  local tag="$1" ckpt="$2" start="$3" end="$4" idx seed gpu out
  for ((idx=start; idx<end; idx++)); do
    seed="${SEEDS_ARR[$idx]}"; gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"; out="$ONLINE/$tag/seed_$seed"
    mkdir -p "$out"
    ( CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
      --memo "repro_jinan2500/$tag/seed_$seed" --projection_ckpt "$ckpt" --llm_path "$MODEL_PATH" \
      --dataset jinan --traffic_file "$TRAFFIC_FILE" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 --min_action_time 30 --traffic_series_interval 2 \
      --inference_batch_size 64 --bf16 --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 \
      --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
      --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
      --result_json "$out/result.json" > "$LOGS/online_${tag}_seed_${seed}.log" 2>&1 ) &
  done
  wait
}

for tag_ckpt in "best:$STAGE2/best.pt" "epoch15:$STAGE2/epoch_15.pt"; do
  tag="${tag_ckpt%%:*}"; ckpt="${tag_ckpt#*:}"
  for ((start=0; start<${#SEEDS_ARR[@]}; start+=${#GPUS[@]})); do
    end=$((start + ${#GPUS[@]})); (( end > ${#SEEDS_ARR[@]} )) && end=${#SEEDS_ARR[@]}
    run_online_batch "$tag" "$ckpt" "$start" "$end"
  done
done

echo "completed: $RUN_ROOT"
