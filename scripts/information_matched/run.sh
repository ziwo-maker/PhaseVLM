#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
MODEL=/mnt/data/Qwen3-VL-8B-Instruct
RUN_ROOT="${RUN_ROOT:-$ROOT/records/information_matched_representation_ablation_seed2024_$(date +%Y%m%d_%H%M%S)}"
cd "$ROOT"
[[ ! -e "$RUN_ROOT" ]] || { printf 'Refusing to reuse %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
export PYTHONPATH="$ROOT"
export PYTHONHASHSEED=2024
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PY" scripts/information_matched/audit.py --output "$RUN_ROOT" --model "$MODEL" > "$RUN_ROOT/audit.log" 2>&1 || {
  printf 'Audit failed: %s\n' "$RUN_ROOT/audit.log" >&2
  exit 1
}
mkdir -p "$RUN_ROOT"/{logs,status,train,online}
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$RUN_ROOT/gpu_info.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$RUN_ROOT/gpu_processes.csv"
"$PY" - "$RUN_ROOT/config.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({
 "seed":2024,"model":"/mnt/data/Qwen3-VL-8B-Instruct","stage1":"none in both arms; visual-only alignment would confound modality",
 "scenarios":{"Jinan-1":{"train":"data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl","dataset":"jinan","flow":"anon_3_4_jinan_real.json"},
              "Hangzhou-1":{"train":"data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl","dataset":"hangzhou","flow":"anon_4_4_hangzhou_real.json"}},
 "arms":{"Text":"history_text","TS2I":"ts2i"},"prompt_mode":"information_matched","seq_len":16,"input_dim":8,"interval_s":2,
 "max_length":512,"batch_size":1,"grad_accum_steps":16,"original_matched_grad_accum_steps":16,"effective_batch_size":16,
 "epochs":15,"lr":0.0001,"min_lr":0.00001,"warmup_ratio":0.1,"lr_schedule":"cosine","weight_decay":0.0001,
 "hidden_dim":64,"prefix_tokens":8,"image_size":224,"bf16":True,"deterministic":True,
 "val_ratio":0.1,"split_strategy":"stratified","target_mode":"mixed","soft_target_alpha":0.3,
 "soft_label_temperature":0.7,"ranking_loss_weight":0.05,"ranking_min_cost_gap":0.02,
 "base_policy_kl_weight":0.02,"base_policy_prior_mass":0.70,"base_policy_kl_mode":"match_or_low_weight",
 "checkpoint_selector":"val_top1_cost_regret/min","run_counts":3600,"min_action_time":30,"phase_override_gap":0,
 "gpu_assignment":{"Jinan-1/Text":0,"Jinan-1/TS2I":2,"Hangzhou-1/Text":3,"Hangzhou-1/TS2I":0}
},indent=2)+"\n")
PY

run_one() {
  local scenario="$1" arm="$2" gpu="$3" data dataset flow mode train_dir result_dir log
  case "$scenario" in
    Jinan-1) data=data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl; dataset=jinan; flow=anon_3_4_jinan_real.json ;;
    Hangzhou-1) data=data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl; dataset=hangzhou; flow=anon_4_4_hangzhou_real.json ;;
    *) return 2 ;;
  esac
  case "$arm" in Text) mode=history_text ;; TS2I) mode=ts2i ;; *) return 2 ;; esac
  train_dir="$RUN_ROOT/train/$scenario/$arm"
  result_dir="$RUN_ROOT/online/$scenario/$arm"
  mkdir -p "$train_dir" "$result_dir"
  printf 'running train gpu=%s pid=%s\n' "$gpu" "$BASHPID" > "$RUN_ROOT/status/${scenario}_${arm}.status"
  log="$RUN_ROOT/logs/train_${scenario}_${arm}_gpu${gpu}.log"
  if ! timeout --foreground 86400s env CUDA_VISIBLE_DEVICES="$gpu" "$PY" projection_train/train_projection_bc.py \
    --data_path "$data" --llm_path "$MODEL" --init_mode random --output_dir "$train_dir" \
    --representation_mode "$mode" --prompt_mode information_matched --seq_len 16 --image_size 224 \
    --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix \
    --max_length 512 --action_eval_chunk_size 1 --batch_size 1 --grad_accum_steps 16 \
    --epochs 15 --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1 --val_ratio 0.1 --split_strategy stratified --num_workers 2 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0.02 \
    --base_policy_prior_mass 0.70 --base_policy_kl_mode match_or_low_weight \
    --best_metric val_top1_cost_regret --best_mode min --save_every 1 --seed 2024 \
    --bf16 --deterministic --checkpoint_slim > "$log" 2>&1; then
    printf 'failed train gpu=%s log=%s\n' "$gpu" "$log" > "$RUN_ROOT/status/${scenario}_${arm}.status"
    return 1
  fi
  [[ -s "$train_dir/best.pt" ]] || return 1
  printf 'running online gpu=%s checkpoint=%s\n' "$gpu" "$train_dir/best.pt" > "$RUN_ROOT/status/${scenario}_${arm}.status"
  log="$RUN_ROOT/logs/online_${scenario}_${arm}_gpu${gpu}.log"
  if ! timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="$gpu" "$PY" projection_train/run_projection_online.py \
    --memo "information_matched/$(basename "$RUN_ROOT")/$scenario/$arm" \
    --projection_ckpt "$train_dir/best.pt" --llm_path "$MODEL" --dataset "$dataset" --traffic_file "$flow" \
    --run_counts 3600 --num_rounds 1 --seed 2024 --representation_mode "$mode" --prompt_mode information_matched \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 \
    --prefix_mode prefix --max_length 512 --min_action_time 30 --traffic_series_interval 2 \
    --inference_batch_size 1 --action_eval_chunk_size 1 --bf16 --deterministic --disable_snapshots \
    --result_json "$result_dir/result.json" > "$log" 2>&1; then
    printf 'failed online gpu=%s log=%s\n' "$gpu" "$log" > "$RUN_ROOT/status/${scenario}_${arm}.status"
    return 1
  fi
  printf 'completed gpu=%s result=%s checkpoint=%s\n' "$gpu" "$result_dir/result.json" "$train_dir/best.pt" > "$RUN_ROOT/status/${scenario}_${arm}.status"
}

(run_one Jinan-1 Text 0 && run_one Hangzhou-1 TS2I 0) > "$RUN_ROOT/logs/worker_gpu0.log" 2>&1 & p0=$!
run_one Jinan-1 TS2I 2 > "$RUN_ROOT/logs/worker_gpu2.log" 2>&1 & p2=$!
run_one Hangzhou-1 Text 3 > "$RUN_ROOT/logs/worker_gpu3.log" 2>&1 & p3=$!
printf 'gpu0=%s\ngpu2=%s\ngpu3=%s\n' "$p0" "$p2" "$p3" > "$RUN_ROOT/worker_pids.txt"
rc=0
wait "$p0" || rc=1
wait "$p2" || rc=1
wait "$p3" || rc=1
"$PY" scripts/information_matched/summarize.py "$RUN_ROOT" > "$RUN_ROOT/logs/summarize.log" 2>&1 || rc=1
printf 'state=%s\nfinished_at=%s\n' "$([[ "$rc" == 0 ]] && printf completed || printf failed)" "$(date --iso-8601=seconds)" > "$RUN_ROOT/completion.status"
exit "$rc"
