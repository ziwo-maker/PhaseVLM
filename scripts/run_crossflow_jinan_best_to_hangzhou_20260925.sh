#!/usr/bin/env bash
# Cross-flow online evaluation: the three archived Jinan best checkpoints on
# the two available real Hangzhou flows.  Each source policy is evaluated on
# each target flow with three configured simulator seeds.
set -uo pipefail

ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ID=crossflow_jinan_best_to_hangzhou_20260925_$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$ROOT/records/online_tests/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

write_manifest() {
  "$PYTHON" - "$RUN_ROOT/manifest.json" <<'PY'
import json, sys
manifest = {
  "purpose": "Cross-flow online evaluation of current archived Jinan best checkpoints on the two available real Hangzhou flows",
  "source_flows": ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json", "anon_3_4_jinan_real_2500.json"],
  "target_dataset": "hangzhou",
  "target_flows": ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json"],
  "seeds": [2024, 42, 123],
  "run_counts": 3600,
  "phase_overrides": "disabled",
  "workers": {"0": "jinan_real", "2": "jinan_2000", "3": "jinan_2500"},
}
with open(sys.argv[1], "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

run_worker() {
  local gpu="$1" source_id="$2" ckpt="$3" llm_path="$4" hidden_dim="$5" prefix_tokens="$6" max_length="$7"
  local target seed target_id out_dir result status log
  for target in anon_4_4_hangzhou_real.json anon_4_4_hangzhou_real_5816.json; do
    target_id="${target%.json}"
    for seed in 2024 42 123; do
      out_dir="$RUN_ROOT/results/$source_id/$target_id/seed_$seed"
      result="$out_dir/result.json"
      status="$out_dir/status.txt"
      log="$LOG_DIR/${source_id}__${target_id}__seed_${seed}__gpu${gpu}.log"
      mkdir -p "$out_dir"
      printf 'started_at=%s\ngpu=%s\nsource=%s\ntarget=%s\nseed=%s\ncheckpoint=%s\n' \
        "$(date --iso-8601=seconds)" "$gpu" "$source_id" "$target" "$seed" "$ckpt" > "$status"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u projection_train/run_projection_online.py \
        --memo "crossflow_jinan_best_to_hangzhou/$source_id/$target_id/seed_$seed" \
        --projection_ckpt "$ckpt" --llm_path "$llm_path" \
        --dataset hangzhou --traffic_file "$target" --run_counts 3600 --num_rounds 1 --seed "$seed" \
        --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
        --hidden_dim "$hidden_dim" --prefix_tokens "$prefix_tokens" --prefix_mode prefix --max_length "$max_length" \
        --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
        --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
        --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
        --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
        --result_json "$result" > "$log" 2>&1
      local rc=$?
      printf 'exit_status=%s\ncompleted_at=%s\n' "$rc" "$(date --iso-8601=seconds)" >> "$status"
    done
  done
}

write_manifest
printf '%s\n' "$RUN_ROOT" > "$ROOT/records/online_tests/latest_crossflow_jinan_best_to_hangzhou_run.txt"
run_worker 0 jinan_real \
  "$ROOT/online_test_best_weights/anon_3_4_jinan_real/jinan_real_simexpert_h60_ml512_epoch12_seed2024.pt" \
  /mnt/data/Qwen3-VL-8B-Instruct 256 32 512 &
pid0=$!
run_worker 2 jinan_2000 \
  "$ROOT/online_test_best_weights/anon_3_4_jinan_real_2000/jinan2000_qwen35_wait2_seed42.pt" \
  /mnt/data/Qwen3.5-9B 64 8 64 &
pid2=$!
run_worker 3 jinan_2500 \
  "$ROOT/online_test_best_weights/anon_3_4_jinan_real_2500/jinan2500_maxlen512_counterfactual_responsive_h30_epoch2_trainseed2024.pt" \
  /mnt/data/Qwen3-VL-8B-Instruct 64 8 512 &
pid3=$!
printf 'run_root=%s\nworker_pids=%s,%s,%s\nstarted_at=%s\n' "$RUN_ROOT" "$pid0" "$pid2" "$pid3" "$(date --iso-8601=seconds)" > "$RUN_ROOT/launcher.status"
wait "$pid0" "$pid2" "$pid3"
