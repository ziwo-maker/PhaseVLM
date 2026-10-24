#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
MODEL=/mnt/data/Qwen3-VL-8B-Instruct
RUN_ROOT="${RUN_ROOT:-$ROOT/records/pipeline_runs/representation_matched_jinan_$(date +%Y%m%d_%H%M%S)}"
SEEDS=(2024 42 123)
EVAL_SEEDS=(2024 42 123 7)
DATA=(
  "$ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
  "$ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
  "$ROOT/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl"
)

mkdir -p "$RUN_ROOT"/{logs,status,train,online}
cd "$ROOT"

"$PY" - "$RUN_ROOT/manifest.json" "${DATA[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, *paths = sys.argv[1:]
payload = {
    "schema": "representation_matched_jinan_v1",
    "scope": "same_flow_jinan1_primary",
    "train_files": [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths],
    "test_flow": "anon_3_4_jinan_real.json",
    "model_path": "/mnt/data/Qwen3-VL-8B-Instruct",
    "arms": {
        "A": {"representation": "history_text", "cues": False},
        "B": {"representation": "ts2i", "cues": False},
        "C": {"representation": "history_text", "cues": True},
        "D": {"representation": "ts2i", "cues": True},
    },
    "training_seeds": [2024, 42, 123],
    "simulator_seeds": [2024, 42, 123, 7],
    "stage1": "none; all arms use random Stage-II interface initialization",
    "stage2": {"epochs": 15, "batch_size": 1, "grad_accum_steps": 16,
               "lr": 1e-4, "checkpoint_selector": "val_top1_cost_regret"},
    "input": {"seq_len": 16, "interval_s": 2, "movement_order": ["NT","NL","ST","SL","ET","EL","WT","WL"],
              "image_size": 224, "max_length": 1024, "current_phase": "not supplied"},
    "evaluation": {"run_counts": 3600, "num_rounds": 1, "min_action_time": 30,
                   "phase_overrides": False, "deterministic": True},
    "gpu_assignment": {"A": 0, "B": 2, "C": 0, "D": 3},
    "scheduled_arms": ["A", "C"] if __import__("os").environ.get("RUN_ONLY_TEXT") == "1" else ["A", "B", "C", "D"],
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

run_one() {
  local arm="$1" gpu="$2" seed="$3" mode representation train_dir train_log status_path checkpoint sim out log
  if [[ "$arm" == A || "$arm" == B ]]; then mode=matched_minimal; else mode=matched_structured; fi
  if [[ "$arm" == A || "$arm" == C ]]; then representation=history_text; else representation=ts2i; fi
  train_dir="$RUN_ROOT/train/$arm/seed_$seed"
  train_log="$RUN_ROOT/logs/train_${arm}_${seed}_gpu${gpu}.log"
  status_path="$RUN_ROOT/status/train_${arm}_${seed}.status"
  mkdir -p "$train_dir"
  printf 'state=running\narm=%s\nseed=%s\ngpu=%s\n' "$arm" "$seed" "$gpu" > "$status_path"
  if [[ ! -s "$train_dir/best.pt" ]] && ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" projection_train/train_projection_bc.py \
      --data_path "${DATA[@]}" --llm_path "$MODEL" --init_mode random \
      --output_dir "$train_dir" --representation_mode "$representation" --prompt_mode "$mode" \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 1024 --action_eval_chunk_size 1 \
      --batch_size 1 --grad_accum_steps 16 --epochs 15 --lr 1e-4 --min_lr 1e-5 \
      --warmup_ratio 0.1 --lr_schedule cosine --weight_decay 1e-4 --max_grad_norm 1 \
      --val_ratio 0.1 --split_strategy stratified --num_workers 2 \
      --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
      --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 \
      --base_policy_kl_weight 0.02 --base_policy_prior_mass 0.70 \
      --base_policy_kl_mode match_or_low_weight \
      --best_metric val_top1_cost_regret --best_mode min --save_every 1 \
      --seed "$seed" --bf16 --checkpoint_slim > "$train_log" 2>&1; then
    printf 'state=failed\nlog=%s\n' "$train_log" > "$status_path"
    return 1
  fi
  checkpoint="$train_dir/best.pt"
  [[ -s "$checkpoint" ]] || return 1
  printf 'state=completed\ncheckpoint=%s\n' "$checkpoint" > "$status_path"
  for sim in "${EVAL_SEEDS[@]}"; do
    out="$RUN_ROOT/online/$arm/train_${seed}/sim_${sim}"
    log="$RUN_ROOT/logs/online_${arm}_${seed}_${sim}_gpu${gpu}.log"
    mkdir -p "$out"
    if [[ -s "$out/result.json" ]]; then
      printf 'state=completed\nresult=%s\n' "$out/result.json" > "$RUN_ROOT/status/online_${arm}_${seed}_${sim}.status"
      continue
    fi
    printf 'state=running\n' > "$RUN_ROOT/status/online_${arm}_${seed}_${sim}.status"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" projection_train/run_projection_online.py \
        --memo "representation_matched/$(basename "$RUN_ROOT")/$arm/train_${seed}/sim_${sim}" \
        --projection_ckpt "$checkpoint" --llm_path "$MODEL" --dataset jinan \
        --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --num_rounds 1 --seed "$sim" \
        --representation_mode "$representation" --prompt_mode "$mode" \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 1024 --min_action_time 30 \
        --traffic_series_interval 2 --inference_batch_size 1 --action_eval_chunk_size 1 --bf16 --deterministic \
        --disable_snapshots --result_json "$out/result.json" > "$log" 2>&1; then
      printf 'state=failed\nlog=%s\n' "$log" > "$RUN_ROOT/status/online_${arm}_${seed}_${sim}.status"
      return 1
    fi
    printf 'state=completed\nresult=%s\n' "$out/result.json" > "$RUN_ROOT/status/online_${arm}_${seed}_${sim}.status"
  done
}

worker() {
  local gpu="$1" arm seed
  shift
  for arm in "$@"; do
    for seed in "${SEEDS[@]}"; do
      run_one "$arm" "$gpu" "$seed" || return 1
    done
  done
}

worker 0 A C > "$RUN_ROOT/logs/worker_gpu0.log" 2>&1 & p0=$!
if [[ "${RUN_ONLY_TEXT:-0}" == "1" ]]; then
  printf '%s\n' "$p0" > "$RUN_ROOT/worker_pids.txt"
  printf '%s\n' "$RUN_ROOT" > "$ROOT/records/pipeline_runs/representation_matched_text_latest.path"
  rc=0
  wait "$p0" || rc=1
  printf 'state=%s\n' "$([[ "$rc" -eq 0 ]] && printf completed || printf failed)" > "$RUN_ROOT/completion.status"
  exit "$rc"
fi
worker 2 B > "$RUN_ROOT/logs/worker_gpu2.log" 2>&1 & p2=$!
worker 3 D > "$RUN_ROOT/logs/worker_gpu3.log" 2>&1 & p3=$!
printf '%s\n' "$p0" "$p2" "$p3" > "$RUN_ROOT/worker_pids.txt"
printf '%s\n' "$RUN_ROOT" > "$ROOT/records/pipeline_runs/representation_matched_latest.path"
rc=0
wait "$p0" || rc=1
wait "$p2" || rc=1
wait "$p3" || rc=1
printf 'state=%s\n' "$([[ "$rc" -eq 0 ]] && printf completed || printf failed)" > "$RUN_ROOT/completion.status"
exit "$rc"
