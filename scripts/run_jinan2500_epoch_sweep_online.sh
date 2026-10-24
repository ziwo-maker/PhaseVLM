#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to the completed Stage 1/2 run directory}"
STAGE2_DIR="${STAGE2_DIR:-$SOURCE_RUN/stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_RUN/online_epoch_sweep_seed2024}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEED="${SEED:-2024}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EPOCHS_RAW="${EPOCHS:-3 4 6 10 15}"

cd "$REPO_ROOT"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
read -r -a EPOCHS_ARR <<< "$EPOCHS_RAW"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_LIST is empty" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 1; }
[[ -f "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real_2500.json" ]] || exit 1
mkdir -p "$OUTPUT_DIR/logs"

run_batch() {
  local start="$1" end="$2" idx epoch gpu ckpt result
  for ((idx=start; idx<end; idx++)); do
    epoch="${EPOCHS_ARR[$idx]}"
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    ckpt="$STAGE2_DIR/epoch_${epoch}.pt"
    result="$OUTPUT_DIR/epoch_${epoch}/result.json"
    [[ -s "$ckpt" ]] || { echo "Missing checkpoint: $ckpt" >&2; exit 1; }
    mkdir -p "$(dirname "$result")"
    (
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
        --memo "jinan2500_epoch_sweep/epoch_${epoch}/seed_${SEED}" \
        --projection_ckpt "$ckpt" --llm_path "$MODEL_PATH" \
        --dataset jinan --traffic_file anon_3_4_jinan_real_2500.json \
        --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$SEED" \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 \
        --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
        --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 \
        --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 \
        --max_phase_override_max_wait_protect 0 --max_phase_override_target_queue_floor 0 \
        --max_phase_override_total_queue_floor 0 --max_phase_override_queue_weight 1 \
        --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
        --result_json "$result" > "$OUTPUT_DIR/logs/epoch_${epoch}.log" 2>&1
    ) &
  done
  wait
}

for ((start=0; start<${#EPOCHS_ARR[@]}; start+=${#GPUS[@]})); do
  end=$((start + ${#GPUS[@]}))
  (( end > ${#EPOCHS_ARR[@]} )) && end=${#EPOCHS_ARR[@]}
  run_batch "$start" "$end"
done

echo "completed: $OUTPUT_DIR"
