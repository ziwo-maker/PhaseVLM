#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
TORCHRUN=/home/sysu/anaconda3/envs/hnr5090/bin/torchrun
MODEL=/mnt/data/Qwen3-VL-8B-Instruct
DATA=$ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
ALIGN=$ROOT/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/stage1/best.pt
RUN_ROOT=${RUN_ROOT:-$ROOT/records/pipeline_runs/jinan_single_dataset_hard_soft_seed2024_20260929}
ARM=${1:?Usage: script ARM GPU (ARM=hard_only or hard_soft)}
GPU=${2:?Usage: script ARM GPU}
SEED=2024

case "$ARM" in
  hard_only) LOSS_ARGS=(--target_mode hard --soft_target_alpha 0 --ranking_loss_weight 0 --base_policy_kl_weight 0) ;;
  hard_soft) LOSS_ARGS=(--target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight 0 --base_policy_kl_weight 0) ;;
  *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac

ARM_ROOT=$RUN_ROOT/$ARM
CKPT_DIR=$ARM_ROOT/checkpoints
LOG_DIR=$ARM_ROOT/logs
RESULT=$ARM_ROOT/online/result.json
STATUS=$ARM_ROOT/status.txt
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$(dirname "$RESULT")"

for path in "$PYTHON" "$TORCHRUN" "$MODEL/config.json" "$DATA" "$ALIGN"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf 'status=failed\nexit_code=%s\nended_at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  fi
}
trap on_exit EXIT

printf 'status=training\nstarted_at=%s\narm=%s\ngpu=%s\n' "$(date --iso-8601=seconds)" "$ARM" "$GPU" > "$STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" --standalone --nproc_per_node=1 "$ROOT/projection_train/train_projection_bc.py" \
  --data_path "$DATA" \
  --llm_path "$MODEL" --align_ckpt "$ALIGN" --output_dir "$CKPT_DIR" \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
  --action_eval_chunk_size 0 --batch_size 1 --grad_accum_steps 16 \
  --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
  --weight_decay 1e-4 --max_grad_norm 1.0 \
  --val_ratio 0.1 --split_strategy stratified --num_workers 4 \
  --best_metric val_macro_f1 --best_mode max --best_min_delta 0 \
  --resample_mode targeted \
  --resample_late_mode date_step --resample_late_interval_seconds 2 \
  --resample_late_start_step 316 --resample_late_weight 1 \
  --resample_worst_intersections intersection_1_1,intersection_1_2,intersection_1_3,intersection_2_2,intersection_3_2,intersection_4_2 \
  --resample_worst_intersection_weight 2 \
  --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
  --resample_high_queue_threshold 14 --resample_high_queue_weight 1 \
  --resample_high_wait_threshold 84 \
  --resample_target_action_counts ETWT:4248,NTST:3934,ELWL:2055,NLSL:1885 \
  --log_interval 20 --save_every 1 --seed "$SEED" --bf16 \
  --ddp_timeout_seconds 1800 "${LOSS_ARGS[@]}" \
  > "$LOG_DIR/train.log" 2>&1

[[ -s "$CKPT_DIR/best.pt" ]] || { echo "Training finished without best.pt" >&2; exit 1; }
printf 'status=evaluating\nstarted_at=%s\narm=%s\ngpu=%s\n' "$(date --iso-8601=seconds)" "$ARM" "$GPU" > "$STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u "$ROOT/projection_train/run_projection_online.py" \
  --memo "online_tests/jinan_single_dataset_hard_soft_seed2024_20260929/$ARM" \
  --projection_ckpt "$CKPT_DIR/best.pt" --llm_path "$MODEL" \
  --dataset jinan --traffic_file anon_3_4_jinan_real.json \
  --run_counts 3600 --num_rounds 1 --seed "$SEED" \
  --proj_name "Jinan-single-dataset-$ARM" --prompt_mode queue_wait_short \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --traffic_series_interval 2 --inference_batch_size 64 \
  --bf16 --deterministic --disable_snapshots --result_json "$RESULT" \
  > "$LOG_DIR/online.log" 2>&1

[[ -s "$RESULT" ]] || { echo "Online evaluation finished without result.json" >&2; exit 1; }
printf 'status=completed\nended_at=%s\narm=%s\ngpu=%s\ncheckpoint=%s\nresult=%s\n' \
  "$(date --iso-8601=seconds)" "$ARM" "$GPU" "$CKPT_DIR/best.pt" "$RESULT" > "$STATUS"
