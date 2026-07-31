#!/usr/bin/env bash
set -euo pipefail
cd /home/code/LLMSTSC

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TERM=dumb
export NO_COLOR=1
export FORCE_COLOR=0
export VLLM_LOGGING_COLOR=0
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export WANDB_MODE=offline
export WANDB_SILENT=true

DATASET_ID=anon_4_4_hangzhou_real
ROOT=records/sim_expert_validation_search/${DATASET_ID}
OUT_DIR=${ROOT}/best
RAW_JSONL=${OUT_DIR}/traffic_prompt_series_2s.jsonl

mkdir -p "${ROOT}" "${OUT_DIR}"

echo "[$(date '+%F %T')] Stage1 DeepSeek pipeline start: ${DATASET_ID}"
echo "[$(date '+%F %T')] Step 1/4: SimExpert horizon/weight search"
if [ ! -s "${RAW_JSONL}" ]; then
  python -u scripts/run_sim_expert_horizon_weight_search.py \
    --dataset hangzhou \
    --traffic_files "${DATASET_ID}.json" \
    --horizons 30,60,90 \
    --weight_sets "1,1,1,0.2;2,1,1,0.2;1,2,1,0.2;1,1,2,0.2;1,1,1,0;1,1,1,0.5" \
    --run_id "${DATASET_ID}" \
    --output_dir "${ROOT}" \
    --cuda_visible_devices 0 \
    --skip_clean \
    --skip_llm_backfill \
    --skip_llm_init
else
  echo "[$(date '+%F %T')] Existing raw JSONL found, skipping search: ${RAW_JSONL}"
fi

echo "[$(date '+%F %T')] Step 2/4: prepare no-leak train/val/test splits"
python -u utils/prepare_stage2_splits.py \
  --inputs "${RAW_JSONL}" \
  --train_output "${OUT_DIR}/align_train.cleaned.jsonl" \
  --val_output "${OUT_DIR}/align_val.cleaned.jsonl" \
  --test_output "${OUT_DIR}/align_test.cleaned.jsonl" \
  --summary_output "${OUT_DIR}/stage2_split_summary.json" \
  --seq_len 16 \
  --dedup_similar_captions \
  --caption_hamming_threshold 3 \
  --caption_bands 4 \
  --train_balance_mode none \
  2>&1 | tee "${OUT_DIR}/stage2_split.log"

echo "[$(date '+%F %T')] Step 3/4: DeepSeek API caption backfill for Stage 1 train split"
python -u utils/backfill_llm_captions_api.py \
  --input "${OUT_DIR}/align_train.cleaned.jsonl" \
  --output "${OUT_DIR}/align_train.cleaned.llm_caption_ds.jsonl" \
  --api_base https://api.deepseek.com \
  --model deepseek-v4-flash \
  --batch_size 8 \
  --workers 8 \
  --language en \
  --max_tokens 64 \
  --temperature 0.8 \
  --top_p 0.95 \
  --thinking disabled \
  --max_caption_retries 3 \
  --diversity_hamming_threshold 4 \
  2>&1 | tee "${OUT_DIR}/backfill_deepseek.log"

echo "[$(date '+%F %T')] Step 4/4: build Stage 1 compact dataset"
python -u scripts/build_stage1_compact_dataset.py \
  --input "${OUT_DIR}/align_train.cleaned.llm_caption_ds.jsonl" \
  --output "${OUT_DIR}/align_train.stage1_compact_ds.jsonl" \
  --seq_len 16 \
  --target_groups 2500 \
  --min_per_bucket 1 \
  --per_caption_cap 3 \
  --per_bucket_cap 18 \
  --per_intersection_cap 220 \
  2>&1 | tee "${OUT_DIR}/stage1_compact_deepseek.log"

echo "[$(date '+%F %T')] Output line counts"
wc -l \
  "${RAW_JSONL}" \
  "${OUT_DIR}/align_train.cleaned.jsonl" \
  "${OUT_DIR}/align_val.cleaned.jsonl" \
  "${OUT_DIR}/align_test.cleaned.jsonl" \
  "${OUT_DIR}/align_train.cleaned.llm_caption_ds.jsonl" \
  "${OUT_DIR}/align_train.stage1_compact_ds.jsonl"

echo "[$(date '+%F %T')] Stage1 DeepSeek pipeline complete"
