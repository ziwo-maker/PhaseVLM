#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/data/model/qwen-vl-8B}"
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_3_4_jinan_real.json}"
RUN_COUNTS="${RUN_COUNTS:-3600}"
EVAL_SEEDS="${EVAL_SEEDS:-2024}"
GPU_LIST="${GPU_LIST:-0 1 2}"
MAX_EPOCHS_PER_ARM="${MAX_EPOCHS_PER_ARM:-5}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_real_action_aux_selected_epochs_online_20260825_$(date +%H%M%S)}"
HYPERPARAM_ROOTS="${HYPERPARAM_ROOTS:-$REPO_ROOT/records/pipeline_runs/jinan_real_compact_action_aux_queue_short_20260825_113232:$REPO_ROOT/records/pipeline_runs/jinan_real_action_aux_weight_sweep_queue_short_20260825_121749:$REPO_ROOT/records/pipeline_runs/jinan_real_action_aux_fine_sweep_queue_short_20260825_130457}"

LOG_DIR="$OUT_ROOT/logs"
RESULT_DIR="$OUT_ROOT/results"
SELECTED_TSV="$OUT_ROOT/selected_epochs.tsv"
TASKS_TSV="$OUT_ROOT/tasks.tsv"
STATUS_TSV="$OUT_ROOT/status.tsv"
SUMMARY_JSON="$OUT_ROOT/summary.json"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

notify_feishu() {
  local title="$1"
  local body="$2"
  local template="${3:-blue}"
  local config
  local parsed
  local webhook

  for config in "${FEISHU_CONFIG:-}" "/root/.claude/feishu.json" "$HOME/.codex/feishu.json"; do
    [[ -n "$config" && -r "$config" ]] && break
  done
  [[ -r "$config" ]] || return 0

  parsed="$("$PYTHON" - "$config" <<'PY' 2>/dev/null || true
import json
import sys

try:
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
mode = str(cfg.get("mode", cfg.get("notify_mode", "on")) or "on").lower()
webhook = str(cfg.get("webhook_url", cfg.get("webhook", "")) or "")
if mode in {"off", "none", "disabled", "false", "0"} or not webhook:
    raise SystemExit(0)
print(webhook)
PY
)"
  webhook="$(printf '%s\n' "$parsed" | sed -n '1p')"
  [[ -n "$webhook" ]] || return 0

  "$PYTHON" - "$title" "$body" "$template" <<'PY' | \
    curl --silent --show-error --connect-timeout 5 --max-time 10 \
      --output /dev/null --request POST "$webhook" \
      --header 'Content-Type: application/json' --data @- || true
import json
import sys

_, title, body, template = sys.argv
print(json.dumps({
    "msg_type": "interactive",
    "card": {
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": [{"tag": "markdown", "content": body}],
    },
}, ensure_ascii=False))
PY
}

generate_selected_epochs() {
  "$PYTHON" - "$HYPERPARAM_ROOTS" "$SELECTED_TSV" "$MAX_EPOCHS_PER_ARM" <<'PY'
import csv
import pathlib
import re
import sys

roots = [pathlib.Path(item) for item in sys.argv[1].split(":") if item]
out_path = pathlib.Path(sys.argv[2])
max_epochs = int(sys.argv[3])
val_re = re.compile(r"\[epoch (\d+)\] val_loss=([0-9.]+) val_acc=([0-9.]+) val_f1=([0-9.]+)")

rows = []
for root in roots:
    if not root.exists():
        continue
    for arm in sorted(root.glob("compact*")):
        ckpt_dir = arm / "stage2" / "checkpoints"
        log_path = ckpt_dir / "train_progress.log"
        if not log_path.exists():
            continue
        vals = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = val_re.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            if ckpt_path.is_file() and ckpt_path.stat().st_size > 0:
                vals.append({
                    "epoch": epoch,
                    "val_loss": float(match.group(2)),
                    "val_acc": float(match.group(3)),
                    "val_f1": float(match.group(4)),
                    "ckpt_path": str(ckpt_path),
                })
        if not vals:
            continue

        reasons = {}
        def add(item, reason):
            reasons.setdefault(item["epoch"], set()).add(reason)

        for rank, item in enumerate(sorted(vals, key=lambda x: (-x["val_f1"], x["val_loss"], -x["val_acc"]))[:3], 1):
            add(item, f"top{rank}_val_f1")
        for rank, item in enumerate(sorted(vals, key=lambda x: (x["val_loss"], -x["val_f1"]))[:2], 1):
            add(item, f"top{rank}_low_val_loss")
        add(max(vals, key=lambda x: x["epoch"]), "last_epoch")

        val_by_epoch = {item["epoch"]: item for item in vals}
        priority = []
        for item in sorted(vals, key=lambda x: (-x["val_f1"], x["val_loss"], -x["val_acc"])):
            if item["epoch"] in reasons:
                priority.append(item["epoch"])
        for item in sorted(vals, key=lambda x: (x["val_loss"], -x["val_f1"])):
            if item["epoch"] in reasons and item["epoch"] not in priority:
                priority.append(item["epoch"])
        last_epoch = max(val_by_epoch)
        if last_epoch in reasons and last_epoch not in priority:
            priority.append(last_epoch)
        selected = sorted(priority[:max_epochs])

        if arm.name == "compact_clip":
            hyperparam = "0p0"
        elif arm.name.startswith("compact_action_aux"):
            hyperparam = arm.name.removeprefix("compact_action_aux")
            if hyperparam.isdigit():
                hyperparam = "0p" + (hyperparam.lstrip("0") or "0")
        else:
            hyperparam = arm.name.removeprefix("compact_")
        for epoch in selected:
            item = val_by_epoch[epoch]
            rows.append({
                "arm": arm.name,
                "hyperparam": hyperparam,
                "root": str(root),
                "epoch": epoch,
                "ckpt_name": f"epoch_{epoch}.pt",
                "val_loss": f'{item["val_loss"]:.6f}',
                "val_acc": f'{item["val_acc"]:.6f}',
                "val_f1": f'{item["val_f1"]:.6f}',
                "reasons": ",".join(sorted(reasons[epoch])),
                "ckpt_path": item["ckpt_path"],
            })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = ["arm", "hyperparam", "root", "epoch", "ckpt_name", "val_loss", "val_acc", "val_f1", "reasons", "ckpt_path"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

if not rows:
    raise SystemExit("No selected checkpoints found.")
print(f"Wrote {len(rows)} selected checkpoint rows to {out_path}")
PY
}

generate_tasks() {
  "$PYTHON" - "$SELECTED_TSV" "$TASKS_TSV" "$RESULT_DIR" "$EVAL_SEEDS" <<'PY'
import csv
import pathlib
import re
import sys

selected_path = pathlib.Path(sys.argv[1])
tasks_path = pathlib.Path(sys.argv[2])
result_dir = pathlib.Path(sys.argv[3])
seeds = [item.strip() for item in sys.argv[4].replace(",", " ").split() if item.strip()]

rows = []
with selected_path.open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        arm = row["arm"]
        epoch = row["epoch"]
        safe_arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", arm)
        for seed in seeds:
            run_id = f"{safe_arm}_epoch_{epoch}_seed_{seed}"
            run_dir = result_dir / safe_arm / f"epoch_{epoch}" / f"seed_{seed}"
            rows.append({
                **row,
                "seed": seed,
                "run_id": run_id,
                "result_json": str(run_dir / "result.json"),
                "worker_status": str(run_dir / "status.tsv"),
                "log_path": str(run_dir / "worker.log"),
            })

tasks_path.parent.mkdir(parents=True, exist_ok=True)
with tasks_path.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = [
        "arm", "hyperparam", "root", "epoch", "ckpt_name", "val_loss", "val_acc", "val_f1", "reasons",
        "ckpt_path", "seed", "run_id", "result_json", "worker_status", "log_path",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} tasks to {tasks_path}")
PY
}

summarize_results() {
  "$PYTHON" - "$TASKS_TSV" "$SUMMARY_JSON" "$SUMMARY_CSV" <<'PY'
import csv
import json
import pathlib
import sys

tasks_path = pathlib.Path(sys.argv[1])
summary_json = pathlib.Path(sys.argv[2])
summary_csv = pathlib.Path(sys.argv[3])

rows = []
if tasks_path.exists():
    with tasks_path.open(encoding="utf-8") as handle:
        for task in csv.DictReader(handle, delimiter="\t"):
            result_path = pathlib.Path(task["result_json"])
            row = {
                "arm": task["arm"],
                "hyperparam": task["hyperparam"],
                "epoch": task["epoch"],
                "seed": task["seed"],
                "val_loss": task["val_loss"],
                "val_acc": task["val_acc"],
                "val_f1": task["val_f1"],
                "reasons": task["reasons"],
                "result_json": str(result_path),
                "log_path": task["log_path"],
                "status": "missing",
            }
            if result_path.is_file() and result_path.stat().st_size > 0:
                try:
                    payload = json.load(open(result_path, encoding="utf-8"))
                    results = payload.get("results") or {}
                    metrics = results[-1] if isinstance(results, list) and results else results
                    if isinstance(metrics, dict):
                        row.update(metrics)
                        queue = metrics.get("test_avg_queue_len_over")
                        wait = metrics.get("test_avg_waiting_time_over")
                        travel = metrics.get("test_avg_travel_time_over")
                        if queue is not None and wait is not None and travel is not None:
                            row["score_sum"] = float(queue) + float(wait) + float(travel)
                    row["work_dir"] = payload.get("work_dir", "")
                    row["status"] = "done"
                except Exception as exc:
                    row["status"] = "parse_error"
                    row["error"] = str(exc)
            rows.append(row)

rows.sort(key=lambda row: (
    row.get("status") != "done",
    float(row.get("score_sum", "inf")) if row.get("score_sum", "") != "" else float("inf"),
    row["arm"],
    int(row["epoch"]),
    int(row["seed"]),
))

summary = {
    "tasks": len(rows),
    "done": sum(1 for row in rows if row.get("status") == "done"),
    "missing": sum(1 for row in rows if row.get("status") == "missing"),
    "rows": rows,
}
summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

fieldnames = [
    "status", "arm", "hyperparam", "epoch", "seed", "score_sum",
    "test_avg_queue_len_over", "test_avg_waiting_time_over", "test_avg_travel_time_over",
    "test_queuing_vehicle_num_over", "test_reward_over",
    "val_loss", "val_acc", "val_f1", "reasons", "result_json", "work_dir", "log_path", "error",
]
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

print(json.dumps({"tasks": summary["tasks"], "done": summary["done"], "missing": summary["missing"], "summary_csv": str(summary_csv)}, ensure_ascii=False))
PY
}

count_done() {
  if [[ ! -s "$TASKS_TSV" ]]; then
    printf '0'
    return 0
  fi
  "$PYTHON" - "$TASKS_TSV" <<'PY'
import csv
import pathlib
import sys

done = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        path = pathlib.Path(row["result_json"])
        done += int(path.is_file() and path.stat().st_size > 0)
print(done)
PY
}

heartbeat_pid=""
start_heartbeat() {
  (
    while true; do
      sleep 2400
      total_tasks=$(( $(wc -l < "$TASKS_TSV") - 1 ))
      completed_tasks="$(count_done)"
      summarize_results > "$LOG_DIR/progress_summary.log" 2>&1 || true
      notify_feishu \
        "Jinan Real selected-epoch online progress" \
        "Progress: ${completed_tasks}/${total_tasks}\nRun: $OUT_ROOT\nSummary: $SUMMARY_CSV" \
        "blue"
    done
  ) &
  heartbeat_pid="$!"
}

stop_heartbeat() {
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" >/dev/null 2>&1 || true
    wait "$heartbeat_pid" >/dev/null 2>&1 || true
  fi
}

on_exit() {
  local status="$?"
  local template="red"
  [[ "$status" -eq 0 ]] && template="green"
  stop_heartbeat
  summarize_results > "$LOG_DIR/summary.log" 2>&1 || true
  notify_feishu \
    "Jinan Real selected-epoch online finished" \
    "Exit status: $status\nRun: $OUT_ROOT\nSelected: $SELECTED_TSV\nSummary: $SUMMARY_CSV" \
    "$template"
  trap - EXIT
  exit "$status"
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$REPO_ROOT"

for required_path in "$PYTHON" "$MODEL_PATH" "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done

generate_selected_epochs
generate_tasks

{
  printf 'out_root=%s\n' "$OUT_ROOT"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repo_root=%s\n' "$REPO_ROOT"
  printf 'python=%s\n' "$PYTHON"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
  printf 'run_counts=%s\n' "$RUN_COUNTS"
  printf 'eval_seeds=%s\n' "$EVAL_SEEDS"
  printf 'gpu_list=%s\n' "$GPU_LIST"
  printf 'max_epochs_per_arm=%s\n' "$MAX_EPOCHS_PER_ARM"
  printf 'hyperparam_roots=%s\n' "$HYPERPARAM_ROOTS"
  printf 'selected_tsv=%s\n' "$SELECTED_TSV"
  printf 'tasks_tsv=%s\n' "$TASKS_TSV"
} | tee "$LOG_DIR/run_config.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  summarize_results | tee "$LOG_DIR/dry_run_summary.log"
  echo "DRY_RUN=1; no online evaluations launched."
  exit 0
fi

trap on_exit EXIT
start_heartbeat

printf 'run_id\tarm\tepoch\tseed\tgpu\tpid\tstatus\tstarted_at\tfinished_at\tresult_json\tlog\n' > "$STATUS_TSV"
read -r -a gpus <<< "$GPU_LIST"
if [[ "${#gpus[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty." >&2
  exit 1
fi

pids=()
status=0
slot=0
while IFS=$'\t' read -r arm hyperparam root epoch ckpt_name val_loss val_acc val_f1 reasons ckpt_path seed run_id result_json worker_status log_path; do
  [[ "$arm" == "arm" ]] && continue
  if [[ ! -s "$ckpt_path" ]]; then
    echo "Missing checkpoint: $ckpt_path" >&2
    status=1
    continue
  fi
  if [[ "$FORCE" != "1" && -s "$result_json" ]]; then
    printf '%s\t%s\t%s\t%s\t\t\tskipped\t%s\t%s\t%s\t%s\n' "$run_id" "$arm" "$epoch" "$seed" "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$STATUS_TSV"
    continue
  fi

  gpu="${gpus[$slot]}"
  mkdir -p "$(dirname "$result_json")"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    printf 'run_id\tarm\tepoch\tseed\tgpu\tstatus\tstarted_at\tfinished_at\tresult_json\tlog\n' > "$worker_status"
    printf '%s\t%s\t%s\t%s\t%s\trunning\t%s\t\t%s\t%s\n' "$run_id" "$arm" "$epoch" "$seed" "$gpu" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$worker_status"
    worker_exit=0
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "jinan_real_${run_id}" \
      --projection_ckpt "$ckpt_path" \
      --llm_path "$MODEL_PATH" \
      --dataset jinan \
      --traffic_file "$TRAFFIC_FILE" \
      --run_counts "$RUN_COUNTS" \
      --seed "$seed" \
      --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --bf16 \
      --result_json "$result_json" || worker_exit="$?"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t\t%s\t%s\t%s\n' "$run_id" "$arm" "$epoch" "$seed" "$gpu" "$([[ "$worker_exit" -eq 0 ]] && echo done || echo failed)" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$worker_status"
    exit "$worker_exit"
  ) > "$log_path" 2>&1 &
  pid="$!"
  pids+=("$pid")
  printf '%s\t%s\t%s\t%s\t%s\t%s\trunning\t%s\t\t%s\t%s\n' "$run_id" "$arm" "$epoch" "$seed" "$gpu" "$pid" "$(date --iso-8601=seconds)" "$result_json" "$log_path" >> "$STATUS_TSV"

  slot=$((slot + 1))
  if [[ "$slot" -ge "${#gpus[@]}" ]]; then
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    pids=()
    slot=0
    summarize_results > "$LOG_DIR/latest_summary.log" 2>&1 || true
  fi
done < "$TASKS_TSV"

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

summarize_results | tee "$LOG_DIR/summary.log"
exit "$status"
