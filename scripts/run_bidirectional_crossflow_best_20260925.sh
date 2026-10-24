#!/usr/bin/env bash
# Evaluate archived Jinan-1 and Hangzhou-2 best checkpoints cross-flow.
set -uo pipefail

ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ID="bidirectional_crossflow_best_20260925_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/records/online_tests/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
SEEDS=(2024 42 123)
GPUS=(0 2 3)
JINAN1_CKPT="$ROOT/online_test_best_weights/anon_3_4_jinan_real/paper_main_table_near_jinan1.pt"
HANGZHOU2_CKPT="$ROOT/online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt"
JINAN1_MODEL=/home/data/model/qwen-vl-8B
HANGZHOU2_MODEL=/mnt/data/Qwen3-VL-8B-Instruct

mkdir -p "$LOG_DIR"
cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "purpose": "Bidirectional cross-flow online generalization evaluation with archived Jinan-1 and Hangzhou-2 best checkpoints",
  "run_counts": 3600,
  "num_rounds": 1,
  "seeds": [2024, 42, 123],
  "workers": {"0": "seed_2024", "2": "seed_42", "3": "seed_123"},
  "evaluations": [
    {"source": "Jinan-1", "checkpoint": "online_test_best_weights/anon_3_4_jinan_real/paper_main_table_near_jinan1.pt", "model": "/home/data/model/qwen-vl-8B", "target": "Hangzhou-2", "dataset": "hangzhou", "traffic_file": "anon_4_4_hangzhou_real_5816.json", "architecture": {"hidden_dim": 64, "prefix_tokens": 8, "max_length": 64}},
    {"source": "Hangzhou-2", "checkpoint": "online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt", "model": "/mnt/data/Qwen3-VL-8B-Instruct", "target": "Jinan-3", "dataset": "jinan", "traffic_file": "anon_3_4_jinan_real_2500.json", "architecture": {"hidden_dim": 256, "prefix_tokens": 8, "max_length": 512}}
  ],
  "common_protocol": {"prompt_mode": "queue_wait_short", "seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "prefix_mode": "prefix", "min_action_time": 30, "traffic_series_interval": 2, "inference_batch_size": 64, "precision": "bf16", "deterministic": true, "snapshots": false, "phase_overrides": "disabled"}
}
EOF
printf 'direction\tseed\tgpu\tstatus\tresult\n' > "$RUN_ROOT/status.tsv"
printf '%s\n' "$RUN_ROOT" > "$ROOT/records/online_tests/latest_bidirectional_crossflow_best_run.txt"

run_one() {
  local direction="$1" seed="$2" gpu="$3" ckpt="$4" model="$5" dataset="$6" flow="$7"
  local hidden="$8" tokens="$9" maxlen="${10}"
  local out="$RUN_ROOT/results/$direction/seed_$seed"
  local result="$out/result.json" log="$LOG_DIR/${direction}__seed_${seed}__gpu${gpu}.log"
  mkdir -p "$out"
  printf '%s\t%s\t%s\trunning\t%s\n' "$direction" "$seed" "$gpu" "$result" >> "$RUN_ROOT/status.tsv"
  timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/projection_train/run_projection_online.py" \
    --memo "${RUN_ID}/${direction}/seed_$seed" --projection_ckpt "$ckpt" --llm_path "$model" \
    --dataset "$dataset" --traffic_file "$flow" --run_counts 3600 --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim "$hidden" --prefix_tokens "$tokens" --prefix_mode prefix --max_length "$maxlen" \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
    --result_json "$result" > "$log" 2>&1
  local rc=$?
  if (( rc == 0 )) && [[ -s "$result" ]]; then
    printf 'ok\n' > "$out/status.txt"
    printf '%s\t%s\t%s\tok\t%s\n' "$direction" "$seed" "$gpu" "$result" >> "$RUN_ROOT/status.tsv"
  else
    printf 'failed exit_code=%s\n' "$rc" > "$out/status.txt"
    printf '%s\t%s\t%s\tfailed:%s\t%s\n' "$direction" "$seed" "$gpu" "$rc" "$result" >> "$RUN_ROOT/status.tsv"
    return 1
  fi
}

worker() {
  local seed="$1" gpu="$2" failed=0
  run_one jinan1_to_hangzhou2 "$seed" "$gpu" "$JINAN1_CKPT" "$JINAN1_MODEL" hangzhou \
    anon_4_4_hangzhou_real_5816.json 64 8 64 || failed=1
  run_one hangzhou2_to_jinan3 "$seed" "$gpu" "$HANGZHOU2_CKPT" "$HANGZHOU2_MODEL" jinan \
    anon_3_4_jinan_real_2500.json 256 8 512 || failed=1
  return "$failed"
}

worker "${SEEDS[0]}" "${GPUS[0]}" & p0=$!
worker "${SEEDS[1]}" "${GPUS[1]}" & p1=$!
worker "${SEEDS[2]}" "${GPUS[2]}" & p2=$!
printf 'run_root=%s\nworker_pids=%s,%s,%s\nstarted_at=%s\n' "$RUN_ROOT" "$p0" "$p1" "$p2" "$(date --iso-8601=seconds)" > "$RUN_ROOT/launcher.status"

failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
wait "$p2" || failed=1
"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metric_keys = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
    "queued_vehicles": "test_queuing_vehicle_num_over",
}
summary = {"evaluations": {}}
for direction in ("jinan1_to_hangzhou2", "hangzhou2_to_jinan3"):
    summary["evaluations"][direction] = {}
    for seed in ("2024", "42", "123"):
        path = root / "results" / direction / f"seed_{seed}" / "result.json"
        item = {"status": "missing", "result_path": str(path)}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            results = payload.get("results", payload)
            item["status"] = "ok"
            item.update({name: results.get(key) for name, key in metric_keys.items()})
        summary["evaluations"][direction][seed] = item
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
(root / "status.txt").write_text("completed_with_failures\n" if any(
    entry.get("status") != "ok"
    for direction in summary["evaluations"].values()
    for entry in direction.values()
) else "completed\n", encoding="utf-8")
PY
exit "$failed"
