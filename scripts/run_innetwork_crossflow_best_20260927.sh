#!/usr/bin/env bash
# Same-road-network cross-flow evaluation using only archived best weights.
# Full requested matrix: Jinan 3x2 + Hangzhou 2x1 = 8 directional pairs.
# Sources without archived weights are recorded as skipped.
set -uo pipefail

ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_ID="innetwork_crossflow_best_20260927_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/records/online_tests/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
SEED=2024
mkdir -p "$LOG_DIR" "$RESULT_DIR"

J1_CKPT="$ROOT/online_test_best_weights/anon_3_4_jinan_real/paper_main_table_near_jinan1.pt"
J3_CKPT="$ROOT/online_test_best_weights/anon_3_4_jinan_real_2500/paper_main_table_near_jinan3.pt"
H2_CKPT="$ROOT/online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt"
J2_CKPT="$ROOT/online_test_best_weights/anon_3_4_jinan_real_2000/paper_main_table_near_jinan2.pt"
H1_CKPT="$ROOT/online_test_best_weights/anon_4_4_hangzhou_real/paper_main_table_near_hangzhou1.pt"

J1_MODEL=/home/data/model/qwen-vl-8B
J3_MODEL=/mnt/data/Qwen3-VL-8B-Instruct
H2_MODEL=/mnt/data/Qwen3-VL-8B-Instruct

printf '%s\n' "$RUN_ROOT" > "$ROOT/records/online_tests/latest_innetwork_crossflow_best_run.txt"

cat > "$RUN_ROOT/manifest.json" <<EOF2
{
  "purpose": "Same-road-network cross-flow generalization: Jinan 3x2 plus Hangzhou 2x1",
  "seed": $SEED,
  "requested_experiment_count": 8,
  "runnable_experiment_count": 5,
  "skipped_experiment_count": 3,
  "protocol": {"run_counts": 3600, "num_rounds": 1, "prompt_mode": "queue_wait_short", "seq_len": 16, "image_size": 224, "periodicity": 16, "input_dim": 8, "prefix_tokens": 8, "prefix_mode": "prefix", "min_action_time": 30, "traffic_series_interval": 2, "inference_batch_size": 64, "precision": "bf16", "deterministic": true, "snapshots": false, "phase_overrides": "disabled"},
  "experiments": [
    {"id":"jinan1_to_jinan2","source":"Jinan-1","target":"Jinan-2","traffic_file":"anon_3_4_jinan_real_2000.json","checkpoint":"$J1_CKPT","model":"$J1_MODEL","hidden_dim":64,"max_length":64,"gpu":0,"status":"queued"},
    {"id":"jinan1_to_jinan3","source":"Jinan-1","target":"Jinan-3","traffic_file":"anon_3_4_jinan_real_2500.json","checkpoint":"$J1_CKPT","model":"$J1_MODEL","hidden_dim":64,"max_length":64,"gpu":2,"status":"queued"},
    {"id":"jinan2_to_jinan1","source":"Jinan-2","target":"Jinan-1","traffic_file":"anon_3_4_jinan_real.json","checkpoint":"$J2_CKPT","status":"skipped_missing_archived_weight"},
    {"id":"jinan2_to_jinan3","source":"Jinan-2","target":"Jinan-3","traffic_file":"anon_3_4_jinan_real_2500.json","checkpoint":"$J2_CKPT","status":"skipped_missing_archived_weight"},
    {"id":"jinan3_to_jinan1","source":"Jinan-3","target":"Jinan-1","traffic_file":"anon_3_4_jinan_real.json","checkpoint":"$J3_CKPT","model":"$J3_MODEL","hidden_dim":64,"max_length":512,"gpu":0,"status":"queued_after_jinan1_to_jinan2"},
    {"id":"jinan3_to_jinan2","source":"Jinan-3","target":"Jinan-2","traffic_file":"anon_3_4_jinan_real_2000.json","checkpoint":"$J3_CKPT","model":"$J3_MODEL","hidden_dim":64,"max_length":512,"gpu":2,"status":"queued_after_jinan1_to_jinan3"},
    {"id":"hangzhou1_to_hangzhou2","source":"Hangzhou-1","target":"Hangzhou-2","traffic_file":"anon_4_4_hangzhou_real_5816.json","checkpoint":"$H1_CKPT","status":"skipped_missing_archived_weight"},
    {"id":"hangzhou2_to_hangzhou1","source":"Hangzhou-2","target":"Hangzhou-1","traffic_file":"anon_4_4_hangzhou_real.json","checkpoint":"$H2_CKPT","model":"$H2_MODEL","hidden_dim":256,"max_length":512,"gpu":3,"status":"queued"}
  ]
}
EOF2

for pair in jinan2_to_jinan1 jinan2_to_jinan3 hangzhou1_to_hangzhou2; do
  mkdir -p "$RESULT_DIR/$pair/seed_$SEED"
  printf 'skipped_missing_archived_weight\n' > "$RESULT_DIR/$pair/seed_$SEED/status.txt"
done

run_one() {
  local pair="$1" gpu="$2" ckpt="$3" model="$4" dataset="$5" flow="$6" hidden="$7" maxlen="$8"
  local out="$RESULT_DIR/$pair/seed_$SEED"
  local result="$out/result.json"
  local log="$LOG_DIR/${pair}__seed_${SEED}__gpu${gpu}.log"
  mkdir -p "$out"
  if [[ ! -s "$ckpt" ]]; then
    printf 'skipped_missing_archived_weight\n' > "$out/status.txt"
    return 0
  fi
  printf 'running gpu=%s started_at=%s\n' "$gpu" "$(date --iso-8601=seconds)" > "$out/status.txt"
  timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/projection_train/run_projection_online.py" \
    --memo "$RUN_ID/$pair/seed_$SEED" \
    --projection_ckpt "$ckpt" --llm_path "$model" \
    --dataset "$dataset" --traffic_file "$flow" \
    --run_counts 3600 --num_rounds 1 --seed "$SEED" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 \
    --hidden_dim "$hidden" --prefix_tokens 8 --prefix_mode prefix --max_length "$maxlen" \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 \
    --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
    --result_json "$result" > "$log" 2>&1
  local rc=$?
  if (( rc == 0 )) && [[ -s "$result" ]]; then
    printf 'ok gpu=%s completed_at=%s\n' "$gpu" "$(date --iso-8601=seconds)" > "$out/status.txt"
    return 0
  fi
  printf 'failed exit_code=%s gpu=%s ended_at=%s\n' "$rc" "$gpu" "$(date --iso-8601=seconds)" > "$out/status.txt"
  return 1
}

worker_gpu0() {
  run_one jinan1_to_jinan2 0 "$J1_CKPT" "$J1_MODEL" jinan anon_3_4_jinan_real_2000.json 64 64 || true
  run_one jinan3_to_jinan1 0 "$J3_CKPT" "$J3_MODEL" jinan anon_3_4_jinan_real.json 64 512 || true
}
worker_gpu2() {
  run_one jinan1_to_jinan3 2 "$J1_CKPT" "$J1_MODEL" jinan anon_3_4_jinan_real_2500.json 64 64 || true
  run_one jinan3_to_jinan2 2 "$J3_CKPT" "$J3_MODEL" jinan anon_3_4_jinan_real_2000.json 64 512 || true
}
worker_gpu3() {
  run_one hangzhou2_to_hangzhou1 3 "$H2_CKPT" "$H2_MODEL" hangzhou anon_4_4_hangzhou_real.json 256 512 || true
}

worker_gpu0 & p0=$!
worker_gpu2 & p2=$!
worker_gpu3 & p3=$!
printf 'run_root=%s\nlauncher_pid=%s\nworker_pids=%s,%s,%s\nstarted_at=%s\n' "$RUN_ROOT" "$$" "$p0" "$p2" "$p3" "$(date --iso-8601=seconds)" > "$RUN_ROOT/launcher.status"
wait "$p0"; wait "$p2"; wait "$p3"

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.loads((root/'manifest.json').read_text())
metric_keys={'ATT':'test_avg_travel_time_over','AQL':'test_avg_queue_len_over','AWT':'test_avg_waiting_time_over','queued_vehicles':'test_queuing_vehicle_num_over'}
summary={'requested_experiment_count':8,'seed':2024,'experiments':[]}
for exp in manifest['experiments']:
    pair=exp['id']; base=root/'results'/pair/'seed_2024'; rp=base/'result.json'; sp=base/'status.txt'
    item={k:exp.get(k) for k in ('id','source','target','traffic_file')}; item['result_path']=str(rp)
    item['status']=sp.read_text().strip() if sp.exists() else 'missing'
    if rp.exists():
        payload=json.loads(rp.read_text()); results=payload.get('results',payload)
        item['status']='ok'
        item.update({name:results.get(key) for name,key in metric_keys.items()})
    summary['experiments'].append(item)
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (root/'status.tsv').open('w') as f:
    f.write('experiment\tsource\ttarget\ttraffic_file\tseed\tstatus\tATT\tAQL\tAWT\tresult\n')
    for x in summary['experiments']:
        f.write('\t'.join(str(x.get(k,'')) for k in ('id','source','target','traffic_file'))+f"\t2024\t{x['status']}\t{x.get('ATT','')}\t{x.get('AQL','')}\t{x.get('AWT','')}\t{x['result_path']}\n")
(root/'status.txt').write_text('completed\n')
PY
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUN_ROOT/launcher.status"
