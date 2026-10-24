#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
MODEL=/mnt/data/Qwen3-VL-8B-Instruct
SOURCE="$ROOT/records/information_matched_representation_ablation_seed2024_20260927_004529"
RETRY="$SOURCE/retry_ml1024"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONHASHSEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
[[ -s "$SOURCE/information_match_audit.json" ]] || exit 2
mkdir -p "$RETRY"/{logs,status,online,train}
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$RETRY/gpu_info.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$RETRY/gpu_processes.csv"
"$PY" - "$RETRY/config.json" "$SOURCE" <<'PY'
import json,sys
from pathlib import Path
out,source=sys.argv[1:]
payload=json.loads((Path(source)/"config.json").read_text())
payload.update({
 "retry_reason":"Jinan-1 Text online prompt reached 517 tokens and failed at max_length=512; no truncation or OOM",
 "train_max_length":512,"online_max_length":1024,
 "gpu_assignment":{"Jinan-1/Text":2,"Jinan-1/TS2I":2,"Hangzhou-1/Text":3,"Hangzhou-1/TS2I":0},
 "checkpoint_paths":{"Jinan-1/Text":str(Path(source)/"train/Jinan-1/Text/best.pt"),
                     "Jinan-1/TS2I":str(Path(source)/"train/Jinan-1/TS2I/best.pt"),
                     "Hangzhou-1/Text":str(Path(source)/"train/Hangzhou-1/Text/best.pt"),
                     "Hangzhou-1/TS2I":str(Path(out).parent/"train/Hangzhou-1/TS2I/best.pt")},
})
Path(out).write_text(json.dumps(payload,indent=2)+"\n")
PY

train_hangzhou_ts2i() {
  local dir="$RETRY/train/Hangzhou-1/TS2I" log="$RETRY/logs/train_Hangzhou-1_TS2I_gpu0.log"
  mkdir -p "$dir"
  [[ -s "$dir/best.pt" ]] && return 0
  printf 'running train gpu=0 pid=%s\n' "$BASHPID" > "$RETRY/status/Hangzhou-1_TS2I.status"
  if ! timeout --foreground 86400s env CUDA_VISIBLE_DEVICES=0 "$PY" projection_train/train_projection_bc.py \
    --data_path data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl \
    --llm_path "$MODEL" --init_mode random --output_dir "$dir" --representation_mode ts2i \
    --prompt_mode information_matched --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 --action_eval_chunk_size 1 \
    --batch_size 1 --grad_accum_steps 16 --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 \
    --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1 --val_ratio 0.1 --split_strategy stratified \
    --num_workers 2 --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0.02 \
    --base_policy_prior_mass 0.70 --base_policy_kl_mode match_or_low_weight \
    --best_metric val_top1_cost_regret --best_mode min --save_every 1 --seed 2024 \
    --bf16 --deterministic --checkpoint_slim > "$log" 2>&1; then
    printf 'failed train gpu=0 log=%s\n' "$log" > "$RETRY/status/Hangzhou-1_TS2I.status"
    return 1
  fi
  [[ -s "$dir/best.pt" ]]
}

eval_one() {
  local scenario="$1" arm="$2" gpu="$3" checkpoint dataset flow mode dir log
  case "$scenario" in
    Jinan-1) dataset=jinan; flow=anon_3_4_jinan_real.json ;;
    Hangzhou-1) dataset=hangzhou; flow=anon_4_4_hangzhou_real.json ;;
    *) return 2 ;;
  esac
  case "$arm" in Text) mode=history_text ;; TS2I) mode=ts2i ;; *) return 2 ;; esac
  if [[ "$scenario" == Hangzhou-1 && "$arm" == TS2I ]]; then
    checkpoint="$RETRY/train/$scenario/$arm/best.pt"
  else
    checkpoint="$SOURCE/train/$scenario/$arm/best.pt"
  fi
  [[ -s "$checkpoint" ]] || return 1
  dir="$RETRY/online/$scenario/$arm"
  log="$RETRY/logs/online_${scenario}_${arm}_gpu${gpu}.log"
  mkdir -p "$dir"
  [[ -s "$dir/result.json" ]] && return 0
  printf 'running online gpu=%s pid=%s checkpoint=%s\n' "$gpu" "$BASHPID" "$checkpoint" > "$RETRY/status/${scenario}_${arm}.status"
  if ! timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="$gpu" "$PY" projection_train/run_projection_online.py \
    --memo "information_matched/$(basename "$SOURCE")/retry_ml1024/$scenario/$arm" \
    --projection_ckpt "$checkpoint" --llm_path "$MODEL" --dataset "$dataset" --traffic_file "$flow" \
    --run_counts 3600 --num_rounds 1 --seed 2024 --representation_mode "$mode" --prompt_mode information_matched \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 \
    --prefix_mode prefix --max_length 1024 --min_action_time 30 --traffic_series_interval 2 \
    --inference_batch_size 1 --action_eval_chunk_size 1 --bf16 --deterministic --disable_snapshots \
    --result_json "$dir/result.json" > "$log" 2>&1; then
    printf 'failed online gpu=%s log=%s\n' "$gpu" "$log" > "$RETRY/status/${scenario}_${arm}.status"
    return 1
  fi
  printf 'completed gpu=%s result=%s checkpoint=%s\n' "$gpu" "$dir/result.json" "$checkpoint" > "$RETRY/status/${scenario}_${arm}.status"
}

(train_hangzhou_ts2i && eval_one Hangzhou-1 TS2I 0) > "$RETRY/logs/worker_gpu0.log" 2>&1 & p0=$!
(eval_one Jinan-1 Text 2 && eval_one Jinan-1 TS2I 2) > "$RETRY/logs/worker_gpu2.log" 2>&1 & p2=$!
eval_one Hangzhou-1 Text 3 > "$RETRY/logs/worker_gpu3.log" 2>&1 & p3=$!
printf 'gpu0=%s\ngpu2=%s\ngpu3=%s\n' "$p0" "$p2" "$p3" > "$RETRY/worker_pids.txt"
rc=0
wait "$p0" || rc=1
wait "$p2" || rc=1
wait "$p3" || rc=1
"$PY" scripts/information_matched/summarize.py "$RETRY" > "$RETRY/logs/summarize.log" 2>&1 || rc=1
cp "$RETRY/summary.md" "$SOURCE/summary.md"
cp "$RETRY/results_summary.json" "$SOURCE/results_summary.json"
printf 'state=%s\nfinished_at=%s\n' "$([[ "$rc" == 0 ]] && printf completed || printf failed)" "$(date --iso-8601=seconds)" > "$RETRY/completion.status"
exit "$rc"
