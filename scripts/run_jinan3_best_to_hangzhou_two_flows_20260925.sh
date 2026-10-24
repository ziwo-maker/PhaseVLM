#!/usr/bin/env bash
# Evaluate the archived Jinan-3 best checkpoint on Hangzhou-1 and Hangzhou-2.
set -uo pipefail

ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ID="jinan3_best_to_hangzhou_two_flows_20260925_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/records/online_tests/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
CKPT="$ROOT/online_test_best_weights/anon_3_4_jinan_real_2500/paper_main_table_near_jinan3.pt"
MODEL=/mnt/data/Qwen3-VL-8B-Instruct
SEED=2024
mkdir -p "$LOG_DIR"

cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "purpose": "Cross-flow online evaluation of Jinan-3 best checkpoint on the two Hangzhou flows",
  "source": "Jinan-3",
  "checkpoint": "$CKPT",
  "model": "$MODEL",
  "seed": $SEED,
  "run_counts": 3600,
  "num_rounds": 1,
  "architecture": {"prompt_mode": "queue_wait_short", "seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "hidden_dim": 64, "prefix_tokens": 8, "prefix_mode": "prefix", "max_length": 512},
  "common_protocol": {"min_action_time": 30, "traffic_series_interval": 2, "inference_batch_size": 64, "precision": "bf16", "deterministic": true, "snapshots": false, "phase_overrides": "disabled"},
  "targets": [
    {"id": "hangzhou1", "traffic_file": "anon_4_4_hangzhou_real.json", "gpu": 0},
    {"id": "hangzhou2", "traffic_file": "anon_4_4_hangzhou_real_5816.json", "gpu": 2}
  ]
}
EOF
printf '%s\n' "$RUN_ROOT" > "$ROOT/records/online_tests/latest_jinan3_best_to_hangzhou_two_flows_run.txt"

run_one() {
  local target="$1" flow="$2" gpu="$3"
  local out="$RUN_ROOT/results/$target/seed_$SEED"
  local result="$out/result.json" log="$LOG_DIR/${target}__seed_${SEED}__gpu${gpu}.log"
  mkdir -p "$out"
  printf 'running\n' > "$out/status.txt"
  timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/projection_train/run_projection_online.py" \
    --memo "$RUN_ID/$target/seed_$SEED" --projection_ckpt "$CKPT" --llm_path "$MODEL" \
    --dataset hangzhou --traffic_file "$flow" --run_counts 3600 --num_rounds 1 --seed "$SEED" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
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

run_one hangzhou1 anon_4_4_hangzhou_real.json 0 & p0=$!
run_one hangzhou2 anon_4_4_hangzhou_real_5816.json 2 & p2=$!
printf 'run_root=%s\nworker_pids=%s,%s\nstarted_at=%s\n' "$RUN_ROOT" "$p0" "$p2" "$(date --iso-8601=seconds)" > "$RUN_ROOT/launcher.status"

failed=0
wait "$p0" || failed=1
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
summary = {"source": "Jinan-3 best", "seed": 2024, "experiments": []}
for target, flow in (("hangzhou1", "anon_4_4_hangzhou_real.json"),
                     ("hangzhou2", "anon_4_4_hangzhou_real_5816.json")):
    base = root / "results" / target / "seed_2024"
    result_path = base / "result.json"
    item = {"target": target, "traffic_file": flow, "result_path": str(result_path), "status": "missing"}
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        results = payload.get("results", payload)
        item["status"] = "ok"
        item.update({name: results.get(key) for name, key in metric_keys.items()})
    else:
        status = base / "status.txt"
        if status.is_file():
            item["status"] = status.read_text(encoding="utf-8").strip()
    summary["experiments"].append(item)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with (root / "status.tsv").open("w", encoding="utf-8") as f:
    f.write("source\ttarget\ttraffic_file\tseed\tstatus\tresult\n")
    for item in summary["experiments"]:
        f.write(f"Jinan-3\t{item['target']}\t{item['traffic_file']}\t2024\t{item['status']}\t{item['result_path']}\n")
(root / "status.txt").write_text("completed\n" if all(
    item["status"] == "ok" for item in summary["experiments"]
) else "completed_with_failures\n", encoding="utf-8")
PY
exit "$failed"
