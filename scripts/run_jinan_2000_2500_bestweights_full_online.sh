#!/usr/bin/env bash
# Full same-flow online evaluation for selected Jinan 2000/2500 checkpoints.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/jinan_2000_2500_bestweights_stage12_qwen3vl8b_qwshort_20260903_201533}"
RUN_ID="${RUN_ID:-jinan_2000_2500_bestweights_full_online_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/online_tests/$RUN_ID}"
EVAL_GPUS_RAW="${EVAL_GPUS:-0 1 2 3}"
EVAL_SEEDS_RAW="${EVAL_SEEDS:-2024 42 123}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-14400s}"
NOTIFY_INTERVAL_SEC="${NOTIFY_INTERVAL_SEC:-2400}"
FEISHU_SCRIPT="${FEISHU_SCRIPT:-$REPO_ROOT/scripts/feishu_push.py}"

read -r -a EVAL_GPUS <<< "$EVAL_GPUS_RAW"
read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_RAW"

QUEUE_TSV="$RUN_ROOT/job_queue.tsv"
STATUS_TSV="$RUN_ROOT/status.tsv"
SUMMARY_JSON="$RUN_ROOT/aggregate.json"
SUMMARY_CSV="$RUN_ROOT/aggregate.csv"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
MONITOR_DIR="$RUN_ROOT/monitor"
CURRENT_PHASE="$RUN_ROOT/current_phase.txt"

timestamp() { date --iso-8601=seconds; }

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_file() {
  [[ -s "$1" ]] || { log "Missing required file: $1"; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "Missing required directory: $1"; return 1; }
}

notify() {
  [[ -s "$FEISHU_SCRIPT" ]] || return 0
  timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" >/dev/null 2>&1 || true
}

latest_progress() {
  local total done failed running
  total="$(awk 'NR > 1 {n++} END {print n+0}' "$QUEUE_TSV" 2>/dev/null || printf 0)"
  done="$(awk -F'\t' 'NR > 1 && $1 == "done" {n++} END {print n+0}' "$QUEUE_TSV" 2>/dev/null || printf 0)"
  failed="$(awk -F'\t' 'NR > 1 && $1 == "failed" {n++} END {print n+0}' "$QUEUE_TSV" 2>/dev/null || printf 0)"
  running="$(awk -F'\t' 'NR > 1 && $1 == "running" {n++} END {print n+0}' "$QUEUE_TSV" 2>/dev/null || printf 0)"
  printf 'phase=%s total=%s done=%s running=%s failed=%s' \
    "$(cat "$CURRENT_PHASE" 2>/dev/null || printf unknown)" "$total" "$done" "$running" "$failed"
}

notify_loop() {
  while true; do
    sleep "$NOTIFY_INTERVAL_SEC"
    notify "Jinan 2000/2500 selected checkpoint full-online progress: $(latest_progress); run_root=$RUN_ROOT"
  done
}

notify_pid=""

finish() {
  local code="$?"
  if [[ -n "$notify_pid" ]]; then
    kill "$notify_pid" >/dev/null 2>&1 || true
  fi
  if (( code == 0 )); then
    printf 'state=completed\ncompleted_at=%s\nsummary_json=%s\nsummary_csv=%s\n' "$(timestamp)" "$SUMMARY_JSON" "$SUMMARY_CSV" > "$RUN_ROOT/campaign.status"
    notify "Jinan 2000/2500 selected checkpoint full-online completed: $(latest_progress); summary=$SUMMARY_JSON"
  else
    printf 'state=failed\nfailed_at=%s\nexit_status=%s\nsummary_json=%s\n' "$(timestamp)" "$code" "$SUMMARY_JSON" > "$RUN_ROOT/campaign.status"
    notify "Jinan 2000/2500 selected checkpoint full-online failed: exit=$code; $(latest_progress); run_root=$RUN_ROOT"
  fi
  return "$code"
}
trap finish EXIT

validate_gpus() {
  local rows gpu used
  mapfile -t rows < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  for gpu in "${EVAL_GPUS[@]}"; do
    used="$(printf '%s\n' "${rows[@]}" | awk -F, -v target="$gpu" '$1 + 0 == target {gsub(/ /, "", $2); print $2}')"
    [[ "$used" =~ ^[0-9]+$ ]] || { log "Could not read GPU $gpu memory usage"; return 1; }
    (( used < 1000 )) || { log "GPU $gpu is not idle: ${used} MiB used"; return 1; }
  done
}

write_manifest_and_queue() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$SOURCE_RUN" "$MODEL_PATH" "$RUN_COUNTS" "${EVAL_SEEDS[@]}" <<'PY'
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

run_root = Path(sys.argv[1])
source_run = Path(sys.argv[2])
model_path = sys.argv[3]
run_counts = int(sys.argv[4])
seeds = [int(x) for x in sys.argv[5:]]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

candidates = [
    {
        "name": "j2000_stage2_best",
        "flow": "anon_3_4_jinan_real_2000.json",
        "checkpoint": source_run / "tasks/jinan_real_2000/stage2/best.pt",
        "offline_epoch": 10,
        "offline_val_top1_cost_regret": 0.8632704604304198,
        "reason": "lowest offline Stage2 val_top1_cost_regret for jinan_real_2000",
    },
    {
        "name": "j2000_stage2_epoch15",
        "flow": "anon_3_4_jinan_real_2000.json",
        "checkpoint": source_run / "tasks/jinan_real_2000/stage2/epoch_15.pt",
        "offline_epoch": 15,
        "offline_val_top1_cost_regret": 0.942419842054898,
        "reason": "final converged checkpoint for robustness against offline selector noise",
    },
    {
        "name": "j2500_stage2_best",
        "flow": "anon_3_4_jinan_real_2500.json",
        "checkpoint": source_run / "tasks/jinan_real_2500/stage2/best.pt",
        "offline_epoch": 11,
        "offline_val_top1_cost_regret": 0.7473711119447317,
        "reason": "lowest offline Stage2 val_top1_cost_regret for jinan_real_2500",
    },
    {
        "name": "j2500_stage2_epoch15",
        "flow": "anon_3_4_jinan_real_2500.json",
        "checkpoint": source_run / "tasks/jinan_real_2500/stage2/epoch_15.pt",
        "offline_epoch": 15,
        "offline_val_top1_cost_regret": 0.8616554186058542,
        "reason": "final converged checkpoint for robustness against offline selector noise",
    },
]
for candidate in candidates:
    checkpoint = Path(candidate["checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)
    candidate["checkpoint"] = str(checkpoint)
    candidate["checkpoint_sha256"] = sha256(checkpoint)

manifest = {
    "schema": "jinan_2000_2500_selected_full_online_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "source_run": str(source_run),
    "model_path": model_path,
    "run_counts": run_counts,
    "num_rounds": 1,
    "prompt_mode": "queue_wait_short",
    "same_flow_evaluation": True,
    "eval_seeds": seeds,
    "candidates": candidates,
}
(run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
with (run_root / "job_queue.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["status", "name", "flow", "checkpoint", "seed", "offline_epoch", "offline_val_top1_cost_regret"])
    for candidate in candidates:
        for seed in seeds:
            writer.writerow([
                "pending",
                candidate["name"],
                candidate["flow"],
                candidate["checkpoint"],
                seed,
                candidate["offline_epoch"],
                candidate["offline_val_top1_cost_regret"],
            ])
PY
}

set_queue_status() {
  local name="$1" seed="$2" status="$3"
  (
    flock -x 9
    awk -F'\t' -v OFS='\t' -v name="$name" -v seed="$seed" -v status="$status" \
      'NR == 1 {print; next} $2 == name && $5 == seed {$1 = status} {print}' "$QUEUE_TSV" > "$QUEUE_TSV.tmp"
    mv "$QUEUE_TSV.tmp" "$QUEUE_TSV"
  ) 9>"$QUEUE_TSV.lock"
}

next_job() {
  (
    flock -x 9
    local picked
    picked="$(awk -F'\t' 'NR > 1 && $1 == "pending" {print NR "\t" $0; exit}' "$QUEUE_TSV")"
    if [[ -z "$picked" ]]; then
      exit 1
    fi
    local nr line
    nr="${picked%%$'\t'*}"
    line="${picked#*$'\t'}"
    awk -F'\t' -v OFS='\t' -v nr="$nr" 'NR == nr {$1 = "running"} {print}' "$QUEUE_TSV" > "$QUEUE_TSV.tmp"
    mv "$QUEUE_TSV.tmp" "$QUEUE_TSV"
    printf '%s\n' "$line"
  ) 9>"$QUEUE_TSV.lock"
}

validate_result() {
  local result="$1" checkpoint="$2" flow="$3" seed="$4"
  "$PYTHON_BIN" - "$result" "$checkpoint" "$flow" "$seed" <<'PY'
import json
import sys
from pathlib import Path

result, checkpoint, flow, seed = sys.argv[1:]
payload = json.loads(Path(result).read_text(encoding="utf-8"))
args = payload.get("args", {})
assert Path(args.get("projection_ckpt", "")).resolve() == Path(checkpoint).resolve()
assert args.get("traffic_file") == flow
assert int(args.get("seed")) == int(seed)
assert int(args.get("run_counts")) == 3600
assert payload.get("prompt_mode") == "queue_wait_short"
assert payload.get("results")
PY
}

append_status() {
  local row="$1"
  (
    flock -x 8
    printf '%s\n' "$row" >> "$STATUS_TSV"
  ) 8>"$STATUS_TSV.lock"
}

run_job() {
  local gpu="$1" name="$2" flow="$3" checkpoint="$4" seed="$5" offline_epoch="$6" offline_regret="$7"
  local out_dir="$RESULT_DIR/$name/seed_$seed"
  local log_path="$LOG_DIR/${name}_seed_${seed}_gpu_${gpu}.log"
  local result_json="$out_dir/result.json"
  local memo="online_tests/$RUN_ID/$name/seed_$seed"
  local status rc=0

  mkdir -p "$out_dir" "$LOG_DIR"
  printf 'online:%s:seed_%s\n' "$name" "$seed" > "$CURRENT_PHASE"
  append_status "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$(timestamp)" "$name" "$flow" "$seed" "$gpu" "$offline_epoch" "$offline_regret" "running" "$checkpoint" "$result_json" "$log_path")"
  log "START name=$name seed=$seed gpu=$gpu flow=$flow"

  if [[ -s "$result_json" ]] && validate_result "$result_json" "$checkpoint" "$flow" "$seed" >/dev/null 2>&1; then
    status="done"
    log "SKIP existing result name=$name seed=$seed"
  else
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      timeout --foreground "$EVAL_TIMEOUT" "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
      --memo "$memo" \
      --projection_ckpt "$checkpoint" \
      --llm_path "$MODEL_PATH" \
      --dataset jinan \
      --traffic_file "$flow" \
      --run_counts "$RUN_COUNTS" \
      --num_rounds 1 \
      --device cuda:0 \
      --seed "$seed" \
      --prompt_mode queue_wait_short \
      --seq_len 16 \
      --image_size 224 \
      --periodicity 16 \
      --input_dim 8 \
      --hidden_dim 64 \
      --prefix_tokens 8 \
      --prefix_mode prefix \
      --max_length 64 \
      --min_action_time 30 \
      --traffic_series_interval 2 \
      --inference_batch_size 64 \
      --bf16 \
      --deterministic \
      --disable_snapshots \
      --result_json "$result_json" \
      > "$log_path" 2>&1
    rc="$?"
    set -e
    if (( rc == 0 )) && [[ -s "$result_json" ]] && validate_result "$result_json" "$checkpoint" "$flow" "$seed"; then
      status="done"
    else
      status="failed"
    fi
  fi
  set_queue_status "$name" "$seed" "$status"
  append_status "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$(timestamp)" "$name" "$flow" "$seed" "$gpu" "$offline_epoch" "$offline_regret" "$status" "$checkpoint" "$result_json" "$log_path")"
  log "END name=$name seed=$seed gpu=$gpu status=$status"
  [[ "$status" == "done" ]]
}

worker() {
  local gpu="$1"
  local line status name flow checkpoint seed offline_epoch offline_regret
  while line="$(next_job)"; do
    IFS=$'\t' read -r status name flow checkpoint seed offline_epoch offline_regret <<< "$line"
    run_job "$gpu" "$name" "$flow" "$checkpoint" "$seed" "$offline_epoch" "$offline_regret" || true
  done
}

snapshot() {
  local label="$1" output="$2"
  {
    printf 'captured_at=%s\nlabel=%s\n' "$(timestamp)" "$label"
    ps -eo pid,ppid,sid,stat,etime,cmd | rg "$RUN_ID|run_projection_online.py" | head -n 120 || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    printf 'queue\n'
    sed -n '1,80p' "$QUEUE_TSV" 2>/dev/null || true
  } > "$output"
}

monitor_startup() {
  mkdir -p "$MONITOR_DIR"
  snapshot "start" "$MONITOR_DIR/start.txt"
  for minute in 1 2; do
    sleep 60
    snapshot "minute_$minute" "$MONITOR_DIR/minute_${minute}.txt"
    if awk -F'\t' 'NR > 1 && $1 == "failed" {found=1} END {exit !found}' "$QUEUE_TSV"; then
      log "A full-online job failed during startup monitoring."
      return 1
    fi
  done
}

aggregate() {
  "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
rows = []
errors = []
for candidate in manifest["candidates"]:
    for seed in manifest["eval_seeds"]:
        result_path = root / "results" / candidate["name"] / f"seed_{seed}" / "result.json"
        if not result_path.is_file():
            errors.append({"name": candidate["name"], "seed": seed, "error": "missing_result", "result_json": str(result_path)})
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row = {
            "name": candidate["name"],
            "flow": candidate["flow"],
            "seed": seed,
            "checkpoint": candidate["checkpoint"],
            "result_json": str(result_path),
            "offline_epoch": candidate["offline_epoch"],
            "offline_val_top1_cost_regret": candidate["offline_val_top1_cost_regret"],
        }
        row.update(payload.get("results", {}))
        rows.append(row)

metric_names = sorted(key for key in {k for row in rows for k in row} if key.startswith("test_"))
by_name = defaultdict(list)
for row in rows:
    by_name[row["name"]].append(row)
aggregate = {}
for name, group in sorted(by_name.items()):
    aggregate[name] = {
        "flow": group[0]["flow"],
        "checkpoint": group[0]["checkpoint"],
        "n": len(group),
        "offline_epoch": group[0]["offline_epoch"],
        "offline_val_top1_cost_regret": group[0]["offline_val_top1_cost_regret"],
        "metrics": {},
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in group if isinstance(row.get(metric), (int, float))]
        if values:
            aggregate[name]["metrics"][metric] = {
                "mean": statistics.fmean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }

payload = {
    "schema": "jinan_2000_2500_selected_full_online_aggregate_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "source_run": manifest["source_run"],
    "run_root": str(root),
    "same_flow_evaluation": True,
    "rows": rows,
    "aggregate": aggregate,
    "errors": errors,
}
(root / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
fieldnames = ["name", "flow", "seed", "offline_epoch", "offline_val_top1_cost_regret", *metric_names, "checkpoint", "result_json"]
with (root / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
if errors or len(rows) != len(manifest["candidates"]) * len(manifest["eval_seeds"]):
    raise SystemExit(1)
PY
}

main() {
  cd "$REPO_ROOT"
  mkdir -p "$RUN_ROOT" "$LOG_DIR" "$RESULT_DIR" "$MONITOR_DIR"
  printf 'state=running\nstarted_at=%s\nrun_id=%s\nrun_root=%s\nsource_run=%s\n' "$(timestamp)" "$RUN_ID" "$RUN_ROOT" "$SOURCE_RUN" > "$RUN_ROOT/campaign.status"
  printf 'preflight\n' > "$CURRENT_PHASE"
  require_file "$PYTHON_BIN"
  require_dir "$MODEL_PATH"
  require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_file "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real_2000.json"
  require_file "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real_2500.json"
  validate_gpus
  write_manifest_and_queue
  printf 'timestamp\tname\tflow\tseed\tgpu\toffline_epoch\toffline_val_top1_cost_regret\tstatus\tcheckpoint\tresult_json\tlog\n' > "$STATUS_TSV"
  log "Launching selected full-online suite at $RUN_ROOT"
  notify "Jinan 2000/2500 selected checkpoint full-online launched: candidates=4 seeds=${EVAL_SEEDS[*]} gpus=${EVAL_GPUS[*]} run_counts=$RUN_COUNTS run_root=$RUN_ROOT"
  notify_loop &
  notify_pid="$!"

  local pids=() gpu pid failures=0
  for gpu in "${EVAL_GPUS[@]}"; do
    worker "$gpu" > "$LOG_DIR/worker_gpu_${gpu}.log" 2>&1 &
    pid="$!"
    pids+=("$pid")
    log "worker gpu=$gpu pid=$pid"
  done

  monitor_startup
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  if awk -F'\t' 'NR > 1 && $1 == "failed" {found=1} END {exit !found}' "$QUEUE_TSV"; then
    failures=$((failures + 1))
  fi
  (( failures == 0 )) || return 1
  aggregate
  printf 'completed\n' > "$CURRENT_PHASE"
  log "Full-online suite complete: $RUN_ROOT"
}

main "$@"
