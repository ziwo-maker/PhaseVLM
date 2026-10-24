#!/usr/bin/env bash
# Complete the five-pair Jinan-1/Hangzhou-2 best-weight cross-flow matrix.
set -uo pipefail

ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
PREVIOUS="$ROOT/records/online_tests/bidirectional_crossflow_best_20260925_20260925_182714"
RUN_ID="crossflow_best_five_flows_20260925_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/records/online_tests/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
JINAN1_CKPT="$ROOT/online_test_best_weights/anon_3_4_jinan_real/paper_main_table_near_jinan1.pt"
HANGZHOU2_CKPT="$ROOT/online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt"
JINAN1_MODEL=/home/data/model/qwen-vl-8B
HANGZHOU2_MODEL=/mnt/data/Qwen3-VL-8B-Instruct
SEED=2024
mkdir -p "$LOG_DIR"

# Reuse the already completed exact pairs, taking their seed-2024 artifacts.
for pair in jinan1_to_hangzhou2 hangzhou2_to_jinan3; do
  mkdir -p "$RUN_ROOT/results/$pair/seed_$SEED"
  cp "$PREVIOUS/results/$pair/seed_$SEED/result.json" "$RUN_ROOT/results/$pair/seed_$SEED/result.json"
  printf 'reused_from=%s\nstatus=ok\n' "$PREVIOUS/results/$pair/seed_$SEED/result.json" \
    > "$RUN_ROOT/results/$pair/seed_$SEED/status.txt"
done

cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "purpose": "Exactly five distinct flow-pair cross-generalization experiments using Jinan-1 and Hangzhou-2 archived best checkpoints",
  "seed": $SEED,
  "run_counts": 3600,
  "num_rounds": 1,
  "pairs": [
    {"id": "jinan1_to_hangzhou1", "source": "Jinan-1", "target": "Hangzhou-1", "dataset": "hangzhou", "traffic_file": "anon_4_4_hangzhou_real.json", "checkpoint": "online_test_best_weights/anon_3_4_jinan_real/paper_main_table_near_jinan1.pt", "model": "/home/data/model/qwen-vl-8B", "hidden_dim": 64, "prefix_tokens": 8, "max_length": 64, "gpu": 0, "execution": "this_run"},
    {"id": "jinan1_to_hangzhou2", "source": "Jinan-1", "target": "Hangzhou-2", "dataset": "hangzhou", "traffic_file": "anon_4_4_hangzhou_real_5816.json", "checkpoint": "online_test_best_weights/anon_3_4_jinan_real/paper_main_table_near_jinan1.pt", "model": "/home/data/model/qwen-vl-8B", "hidden_dim": 64, "prefix_tokens": 8, "max_length": 64, "gpu": 0, "execution": "reused_seed_2024_result"},
    {"id": "hangzhou2_to_jinan1", "source": "Hangzhou-2", "target": "Jinan-1", "dataset": "jinan", "traffic_file": "anon_3_4_jinan_real.json", "checkpoint": "online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt", "model": "/mnt/data/Qwen3-VL-8B-Instruct", "hidden_dim": 256, "prefix_tokens": 8, "max_length": 512, "gpu": 2, "execution": "this_run"},
    {"id": "hangzhou2_to_jinan2", "source": "Hangzhou-2", "target": "Jinan-2", "dataset": "jinan", "traffic_file": "anon_3_4_jinan_real_2000.json", "checkpoint": "online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt", "model": "/mnt/data/Qwen3-VL-8B-Instruct", "hidden_dim": 256, "prefix_tokens": 8, "max_length": 512, "gpu": 3, "execution": "this_run"},
    {"id": "hangzhou2_to_jinan3", "source": "Hangzhou-2", "target": "Jinan-3", "dataset": "jinan", "traffic_file": "anon_3_4_jinan_real_2500.json", "checkpoint": "online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt", "model": "/mnt/data/Qwen3-VL-8B-Instruct", "hidden_dim": 256, "prefix_tokens": 8, "max_length": 512, "gpu": 0, "execution": "reused_seed_2024_result"}
  ],
  "common_protocol": {"prompt_mode": "queue_wait_short", "seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "prefix_mode": "prefix", "min_action_time": 30, "traffic_series_interval": 2, "inference_batch_size": 64, "precision": "bf16", "deterministic": true, "snapshots": false, "phase_overrides": "disabled"},
  "previous_run": "$PREVIOUS"
}
EOF
printf '%s\n' "$RUN_ROOT" > "$ROOT/records/online_tests/latest_crossflow_best_five_flows_run.txt"

run_one() {
  local pair="$1" gpu="$2" ckpt="$3" model="$4" dataset="$5" flow="$6"
  local hidden="$7" maxlen="$8"
  local out="$RUN_ROOT/results/$pair/seed_$SEED"
  local result="$out/result.json" log="$LOG_DIR/${pair}__seed_${SEED}__gpu${gpu}.log"
  mkdir -p "$out"
  printf 'running\n' > "$out/status.txt"
  timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/projection_train/run_projection_online.py" \
    --memo "$RUN_ID/$pair/seed_$SEED" --projection_ckpt "$ckpt" --llm_path "$model" \
    --dataset "$dataset" --traffic_file "$flow" --run_counts 3600 --num_rounds 1 --seed "$SEED" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim "$hidden" --prefix_tokens 8 --prefix_mode prefix --max_length "$maxlen" \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
    --result_json "$result" > "$log" 2>&1
  local rc=$?
  if (( rc == 0 )) && [[ -s "$result" ]]; then
    printf 'ok\n' > "$out/status.txt"
    return 0
  fi
  printf 'failed exit_code=%s\n' "$rc" > "$out/status.txt"
  return 1
}

run_one jinan1_to_hangzhou1 0 "$JINAN1_CKPT" "$JINAN1_MODEL" hangzhou \
  anon_4_4_hangzhou_real.json 64 64 & p0=$!
run_one hangzhou2_to_jinan1 2 "$HANGZHOU2_CKPT" "$HANGZHOU2_MODEL" jinan \
  anon_3_4_jinan_real.json 256 512 & p2=$!
run_one hangzhou2_to_jinan2 3 "$HANGZHOU2_CKPT" "$HANGZHOU2_MODEL" jinan \
  anon_3_4_jinan_real_2000.json 256 512 & p3=$!
printf 'run_root=%s\nworker_pids=%s,%s,%s\nstarted_at=%s\n' "$RUN_ROOT" "$p0" "$p2" "$p3" "$(date --iso-8601=seconds)" > "$RUN_ROOT/launcher.status"

failed=0
wait "$p0" || failed=1
wait "$p2" || failed=1
wait "$p3" || failed=1
"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pairs = [
    ("jinan1_to_hangzhou1", "Jinan-1", "Hangzhou-1", "anon_4_4_hangzhou_real.json", "new"),
    ("jinan1_to_hangzhou2", "Jinan-1", "Hangzhou-2", "anon_4_4_hangzhou_real_5816.json", "reused"),
    ("hangzhou2_to_jinan1", "Hangzhou-2", "Jinan-1", "anon_3_4_jinan_real.json", "new"),
    ("hangzhou2_to_jinan2", "Hangzhou-2", "Jinan-2", "anon_3_4_jinan_real_2000.json", "new"),
    ("hangzhou2_to_jinan3", "Hangzhou-2", "Jinan-3", "anon_3_4_jinan_real_2500.json", "reused"),
]
metric_keys = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
    "queued_vehicles": "test_queuing_vehicle_num_over",
}
summary = {"experiment_count": 5, "seed": 2024, "experiments": []}
for pair, source, target, flow, provenance in pairs:
    base = root / "results" / pair / "seed_2024"
    result_path = base / "result.json"
    status_path = base / "status.txt"
    item = {"id": pair, "source": source, "target": target, "traffic_file": flow,
            "status": "missing", "provenance": provenance, "result_path": str(result_path)}
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        results = payload.get("results", payload)
        item["status"] = "ok"
        item.update({name: results.get(key) for name, key in metric_keys.items()})
    elif status_path.is_file():
        item["status"] = status_path.read_text(encoding="utf-8").strip()
    summary["experiments"].append(item)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with (root / "status.tsv").open("w", encoding="utf-8") as f:
    f.write("experiment\tsource\ttarget\ttraffic_file\tseed\tstatus\tprovenance\tresult\n")
    for item in summary["experiments"]:
        f.write("\t".join(str(item.get(k, "")) for k in
                ("id", "source", "target", "traffic_file")) +
                f"\t2024\t{item['status']}\t{item['provenance']}\t{item['result_path']}\n")
(root / "status.txt").write_text("completed\n" if all(
    item["status"] == "ok" for item in summary["experiments"]
) else "completed_with_failures\n", encoding="utf-8")
PY
exit "$failed"
