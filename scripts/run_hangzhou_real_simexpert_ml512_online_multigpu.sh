#!/usr/bin/env bash
# Same-flow, multi-seed online evaluation for selected checkpoints from the
# Hangzhou-real SimExpert Stage-2 run.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON_BIN="${PYTHON_BIN:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
SOURCE_RUN="${SOURCE_RUN:-$REPO_ROOT/records/pipeline_runs/hangzhou_real_simexpert_h30_att1_aql1_awt1_spill0p2_stage12_ml512_ga16_20260908_004432}"
STAGE2_DIR="${STAGE2_DIR:-$SOURCE_RUN/stage2}"
RUN_ID="${RUN_ID:-hangzhou_real_simexpert_ml512_online_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_RUN/$RUN_ID}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEEDS_RAW="${SEEDS:-2024 42 123}"
CHECKPOINTS_RAW="${CHECKPOINTS:-epoch_3 epoch_6 epoch_9 epoch_12 epoch_15}"
FLOW="anon_4_4_hangzhou_real.json"
RUN_COUNTS="${RUN_COUNTS:-3600}"
CURRENT_BEST_AWT="${CURRENT_BEST_AWT:-44.94645940401502}"

require_file() { [[ -s "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }; }
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

run_task() {
  local checkpoint="$1" seed="$2" gpu="$3"
  local checkpoint_path="$STAGE2_DIR/${checkpoint}.pt"
  local task_dir="$OUTPUT_DIR/$checkpoint/seed_$seed"
  local status="$task_dir/status.txt"
  mkdir -p "$task_dir"
  if timeout --foreground 3600s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u projection_train/run_projection_online.py \
    --memo "$RUN_ID/${checkpoint}/seed_${seed}" --projection_ckpt "$checkpoint_path" --llm_path "$MODEL_PATH" \
    --dataset hangzhou --traffic_file "$FLOW" --run_counts "$RUN_COUNTS" --num_rounds 1 --seed "$seed" \
    --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 --prefix_tokens 32 --prefix_mode prefix --max_length 512 \
    --min_action_time 30 --traffic_series_interval 2 --inference_batch_size 64 --bf16 --deterministic --disable_snapshots \
    --max_phase_override_gap 0 --max_phase_override_left_wait_protect 0 --max_phase_override_max_wait_protect 0 \
    --max_phase_override_target_queue_floor 0 --max_phase_override_total_queue_floor 0 \
    --max_phase_override_queue_weight 1 --max_phase_override_wait_weight 0 --max_phase_override_p95_wait_weight 0 \
    --result_json "$task_dir/result.json" > "$task_dir/online.log" 2>&1; then
    printf 'ok completed_at=%s gpu=%s\n' "$(date --iso-8601=seconds)" "$gpu" > "$status"
  else
    local rc=$?
    printf 'failed exit_code=%s ended_at=%s gpu=%s\n' "$rc" "$(date --iso-8601=seconds)" "$gpu" > "$status"
    return "$rc"
  fi
}

write_summary_and_archive_if_needed() {
  OUTPUT_DIR="$OUTPUT_DIR" STAGE2_DIR="$STAGE2_DIR" SOURCE_RUN="$SOURCE_RUN" RUN_ID="$RUN_ID" FLOW="$FLOW" RUN_COUNTS="$RUN_COUNTS" \
  CHECKPOINTS_RAW="$CHECKPOINTS_RAW" SEEDS_RAW="$SEEDS_RAW" GPU_LIST="$GPU_LIST" CURRENT_BEST_AWT="$CURRENT_BEST_AWT" \
  REPO_ROOT="$REPO_ROOT" MODEL_PATH="$MODEL_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
stage2 = Path(os.environ["STAGE2_DIR"])
flow = os.environ["FLOW"]
metrics = {
    "reward": "test_reward_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
    "ATT": "test_avg_travel_time_over",
    "queued_vehicles": "test_queuing_vehicle_num_over",
    "p95_wait": "test_vehicle_wait_p95_over",
    "max_wait": "test_vehicle_wait_max_over",
}
by_checkpoint = {}
all_rows = []
for checkpoint in os.environ["CHECKPOINTS_RAW"].split():
    rows = []
    for path in sorted((root / checkpoint).glob("seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("results", payload)
        row = {"checkpoint": checkpoint, "seed": int(path.parent.name.removeprefix("seed_")), "result_path": str(path)}
        row.update({name: float(result[key]) for name, key in metrics.items() if result.get(key) is not None})
        rows.append(row)
        all_rows.append(row)
    aggregate = {}
    for metric in metrics:
        values = [row[metric] for row in rows if metric in row]
        if values:
            aggregate[metric] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "n": len(values)}
    by_checkpoint[checkpoint] = {"per_seed": rows, "aggregate": aggregate}

complete = all(len(item["per_seed"]) == len(os.environ["SEEDS_RAW"].split()) for item in by_checkpoint.values())
ranked = sorted(
    (item for item in by_checkpoint.values() if item["aggregate"].get("AWT")),
    key=lambda item: (item["aggregate"]["AWT"]["mean"], item["aggregate"].get("ATT", {}).get("mean", float("inf")), item["aggregate"].get("AQL", {}).get("mean", float("inf"))),
)
summary = {
    "schema": "hangzhou_real_simexpert_ml512_online_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scope": "same_flow",
    "flow": flow,
    "run_counts": int(os.environ["RUN_COUNTS"]),
    "max_length": 512,
    "checkpoints": os.environ["CHECKPOINTS_RAW"].split(),
    "seeds": [int(seed) for seed in os.environ["SEEDS_RAW"].split()],
    "gpus": os.environ["GPU_LIST"],
    "complete": complete,
    "by_checkpoint": by_checkpoint,
    "best_by_awt_mean": ranked[0] if ranked else None,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

# A single completed seed is sufficient for an archive decision under project policy.
winner = min((row for row in all_rows if "AWT" in row), key=lambda row: (row["AWT"], row.get("ATT", float("inf")), row.get("AQL", float("inf"))), default=None)
previous = float(os.environ["CURRENT_BEST_AWT"])
archive = {"archived": False, "previous_best_awt": previous, "candidate": winner}
if winner is not None and winner["AWT"] < previous:
    archive_root = Path(os.environ["REPO_ROOT"]) / "online_test_best_weights" / Path(flow).stem
    archive_root.mkdir(parents=True, exist_ok=True)
    source = stage2 / f"{winner['checkpoint']}.pt"
    destination = archive_root / f"simexpert_ml512_{winner['checkpoint']}_seed{winner['seed']}.pt"
    shutil.copy2(source, destination)
    index = archive_root / "README.md"
    index.write_text(json.dumps({
        "flow": flow, "metric": "test_avg_waiting_time_over", "previous_best": previous,
        "new_best": winner["AWT"], "source_checkpoint": str(source), "archived_weight": str(destination),
        "seed": winner["seed"], "scope": "same-flow full run_counts=3600, single-seed archive selection",
        "result_path": winner["result_path"], "summary_path": str(root / "summary.json"),
        "model_path": os.environ["MODEL_PATH"], "max_length": 512,
    }, indent=2) + "\n", encoding="utf-8")
    project_index = Path(os.environ["REPO_ROOT"]) / "online_test_best_weights" / "README.md"
    with project_index.open("a", encoding="utf-8") as handle:
        handle.write("\n### " + datetime.now().strftime("%Y-%m-%d") + " " + Path(flow).stem + " simexpert_ml512\n\n")
        handle.write(f"- Flow: `{flow}`\n- Metric: `test_avg_waiting_time_over`\n- Previous best: `{previous}`\n- New best: `{winner['AWT']}`\n")
        handle.write(f"- Source checkpoint: `{source}`\n- Archived weights: `{destination}`\n")
        handle.write(f"- Scope: same flow, full 3600-step online test, seed `{winner['seed']}`; multi-seed result summary: `{root / 'summary.json'}`.\n")
        handle.write(f"- Reproduce: run `{Path(os.environ['REPO_ROOT']) / 'scripts' / 'run_hangzhou_real_simexpert_ml512_online_multigpu.sh'}` with `CHECKPOINTS={winner['checkpoint']}` and `SEEDS={winner['seed']}`.\n")
    archive.update({"archived": True, "archive_path": str(destination)})
summary["archive"] = archive
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"complete": complete, "best_by_awt_mean": summary["best_by_awt_mean"], "archive": archive}, indent=2))
PY
}

main() {
  cd "$REPO_ROOT"
  require_file "$PYTHON_BIN"; require_file "$REPO_ROOT/projection_train/run_projection_online.py"
  require_dir "$MODEL_PATH"; require_dir "$STAGE2_DIR"
  require_file "$REPO_ROOT/data/Hangzhou/4_4/$FLOW"
  [[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to reuse existing OUTPUT_DIR: $OUTPUT_DIR" >&2; exit 2; }
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  read -r -a SEEDS <<< "$SEEDS_RAW"
  read -r -a CHECKPOINTS <<< "$CHECKPOINTS_RAW"
  (( ${#GPUS[@]} == 4 )) || { echo "GPU_LIST must contain exactly four GPUs" >&2; exit 2; }
  for checkpoint in "${CHECKPOINTS[@]}"; do require_file "$STAGE2_DIR/${checkpoint}.pt"; done
  mkdir -p "$OUTPUT_DIR"
  printf 'running started_at=%s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
  {
    printf 'source_run=%s\nstage2_dir=%s\nflow=%s\nrun_counts=%s\n' "$SOURCE_RUN" "$STAGE2_DIR" "$FLOW" "$RUN_COUNTS"
    printf 'checkpoints=%s\nseeds=%s\ngpus=%s\n' "$CHECKPOINTS_RAW" "$SEEDS_RAW" "$GPU_LIST"
    printf 'architecture=prompt_mode=queue_wait_short,seq_len=16,image_size=224,periodicity=16,input_dim=8,hidden_dim=256,prefix_tokens=32,prefix_mode=prefix,max_length=512,bf16,deterministic\n'
    printf 'max_phase_override=disabled_all_thresholds_zero\n'
  } > "$OUTPUT_DIR/manifest.txt"
  printf 'checkpoint\tseed\tstatus\n' > "$OUTPUT_DIR/status.tsv"

  local tasks=() checkpoint seed start offset index gpu
  for checkpoint in "${CHECKPOINTS[@]}"; do for seed in "${SEEDS[@]}"; do tasks+=("$checkpoint:$seed"); done; done
  for ((start=0; start<${#tasks[@]}; start+=${#GPUS[@]})); do
    local pids=() labels=()
    for ((offset=0; offset<${#GPUS[@]} && start+offset<${#tasks[@]}; offset++)); do
      index=$((start + offset)); checkpoint="${tasks[$index]%%:*}"; seed="${tasks[$index]##*:}"; gpu="${GPUS[$offset]}"
      log "launch checkpoint=$checkpoint seed=$seed gpu=$gpu"
      run_task "$checkpoint" "$seed" "$gpu" & pids+=("$!"); labels+=("$checkpoint:$seed")
    done
    for ((offset=0; offset<${#pids[@]}; offset++)); do
      wait "${pids[$offset]}" || true
      checkpoint="${labels[$offset]%%:*}"; seed="${labels[$offset]##*:}"
      printf '%s\t%s\t%s\n' "$checkpoint" "$seed" "$(tr '\n' ' ' < "$OUTPUT_DIR/$checkpoint/seed_$seed/status.txt")" >> "$OUTPUT_DIR/status.tsv"
    done
  done
  write_summary_and_archive_if_needed
  if rg -q $'\tfailed ' "$OUTPUT_DIR/status.tsv"; then
    printf 'completed_with_failures ended_at=%s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
    exit 1
  fi
  printf 'completed ended_at=%s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
}

main "$@"
