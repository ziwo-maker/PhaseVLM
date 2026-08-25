#!/usr/bin/env bash
# Evaluate saved offline-val_acc checkpoints from the online-selector campaign.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${JINAN_VALACC_COMPARATOR_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
RUN_ROOT="${1:-$REPO_ROOT/records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823}"
SOURCE_ROOT="${JINAN_VALACC_COMPARATOR_SOURCE_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2}"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_wait25_improvement_online_suite.sh"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
FLOW=anon_3_4_jinan_real.json
ARMS=(wait2_mixed_rank wait225_mixed_rank)
TRAINING_SEEDS=(2024 42 123)
EVAL_SEEDS=(2024 42 123)
LOG="$RUN_ROOT/campaign.log"
STATUS="$RUN_ROOT/campaign.status"
PROVENANCE="$RUN_ROOT/provenance.env"
LAST_NOTIFICATION_EPOCH=0
SELECTED_GPUS=()

timestamp() { date -Is; }

notify() {
    [[ -f "$FEISHU_SCRIPT" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" || true
}

discover_idle_gpus() {
    local gpu used
    SELECTED_GPUS=()
    while IFS=, read -r gpu used; do
        gpu="${gpu//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
        (( used < 500 )) && SELECTED_GPUS+=("$gpu")
    done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    ((${#SELECTED_GPUS[@]} >= 3)) || { echo "Need 3 idle GPUs; found ${#SELECTED_GPUS[@]}" >&2; return 1; }
}

completed_cells() {
    local count=0 path
    for path in "$RUN_ROOT"/arms/*/seed_*/status.txt; do
        [[ -f "$path" ]] || continue
        awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$path" && count=$((count + 1))
    done
    printf '%s' "$count"
}

latest_metric() {
    local result
    result="$(find "$RUN_ROOT/arms" -path "*/evaluation/${FLOW%.json}/seed_2024/result.json" -type f -print 2>/dev/null | sort | tail -n 1 || true)"
    [[ -s "$result" ]] || { printf 'latest_result=not_available'; return 0; }
    "$PYTHON_BIN" - "$result" <<'PY' 2>/dev/null || printf 'latest_result=unreadable'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
print("latest offline-val_acc seed2024 reward={:.3f} AQL={:.3f} AWT={:.3f} ATT={:.3f} P95={:.3f} Max={:.3f}".format(
    float(r["test_reward_over"]),
    float(r["test_avg_queue_len_over"]),
    float(r["test_avg_waiting_time_over"]),
    float(r["test_avg_travel_time_over"]),
    float(r["test_vehicle_wait_p95_over"]),
    float(r["test_vehicle_wait_max_over"]),
))
PY
}

maybe_notify_progress() {
    local now
    now="$(date +%s)"
    (( now - LAST_NOTIFICATION_EPOCH >= 2400 )) || return 0
    LAST_NOTIFICATION_EPOCH="$now"
    notify "Jinan offline-val_acc comparator progress: $(completed_cells)/6 cells complete; current=$1; $(latest_metric)."
}

checkpoint_for() {
    local arm="$1" seed="$2"
    printf '%s/arms/%s/seed_%s/checkpoints/best.offline_val_acc.pt' "$SOURCE_ROOT" "$arm" "$seed"
}

preflight() {
    [[ -x "$PYTHON_BIN" && -x "$ONLINE_SUITE" && -d "$SOURCE_ROOT" ]] || { echo "Missing comparator prerequisite" >&2; return 1; }
    local arm seed ckpt
    for arm in "${ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            ckpt="$(checkpoint_for "$arm" "$seed")"
            [[ -s "$ckpt" ]] || { echo "Missing offline-val_acc checkpoint: $ckpt" >&2; return 1; }
        done
    done
    discover_idle_gpus
}

write_provenance() {
    local git_head git_diff_hash path key
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    {
        printf 'schema=jinan_online_checkpoint_selection_valacc_comparator_v1\n'
        printf 'created_at=%s\nrun_root=%s\nsource_root=%s\n' "$(timestamp)" "$RUN_ROOT" "$SOURCE_ROOT"
        printf 'git_head=%s\ngit_diff_sha256=%s\n' "$git_head" "$git_diff_hash"
        printf 'training_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_only\n' "$FLOW" "$FLOW"
        printf 'checkpoint_selector=offline_val_acc_backup\n'
        printf 'arms=%s\ntraining_seeds=%s\neval_seeds=%s\n' "${ARMS[*]}" "${TRAINING_SEEDS[*]}" "${EVAL_SEEDS[*]}"
        for path in "$ONLINE_SUITE" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/scripts/run_jinan_online_checkpoint_selection_valacc_comparator.sh"; do
            key="${path#"$REPO_ROOT"/}"
            key="${key//\//_}"
            printf 'file_%s_sha256=%s\n' "$key" "$(sha256sum "$path" | awk '{print $1}')"
        done
    } > "$PROVENANCE"
}

run_cell() {
    local arm="$1" seed="$2" target ckpt status_path gpu_list
    target="$RUN_ROOT/arms/$arm/seed_$seed"
    ckpt="$(checkpoint_for "$arm" "$seed")"
    status_path="$target/status.txt"
    if [[ -f "$status_path" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$status_path"; then
        printf '[%s] SKIP arm=%s seed=%s already successful\n' "$(timestamp)" "$arm" "$seed"
        return 0
    fi
    [[ ! -e "$target" ]] || { echo "Existing incomplete target cannot be safely overwritten: $target" >&2; return 1; }
    discover_idle_gpus
    gpu_list="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]:0:3}")"
    mkdir -p "$target"
    printf 'started_at=%s\nstatus=running\narm=%s\ntraining_seed=%s\ncheckpoint=%s\ncheckpoint_selector=offline_val_acc_backup\n' \
        "$(timestamp)" "$arm" "$seed" "$ckpt" > "$status_path"
    printf '[%s] START arm=%s seed=%s checkpoint=%s eval_gpus=%s\n' "$(timestamp)" "$arm" "$seed" "$ckpt" "$gpu_list"
    notify "Jinan offline-val_acc comparator launched: arm=${arm}, train_seed=${seed}, gpus=${gpu_list}."
    WAIT25_IMPROVEMENT_EVAL_GPUS="$gpu_list" \
    WAIT25_IMPROVEMENT_HIDDEN_DIM=64 \
    WAIT25_IMPROVEMENT_PREFIX_TOKENS=8 \
        "$ONLINE_SUITE" "$RUN_ROOT" "$target" "$ckpt" "jinan_valacc_${arm}_train${seed}"
    printf 'completed_at=%s\nstatus=success\narm=%s\ntraining_seed=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\ncheckpoint_selector=offline_val_acc_backup\nonline_status=success\n' \
        "$(timestamp)" "$arm" "$seed" "$ckpt" "$(sha256sum "$ckpt" | awk '{print $1}')" > "$status_path"
    printf '[%s] DONE arm=%s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    maybe_notify_progress "$arm/seed_$seed"
}

aggregate_results() {
    timeout --foreground 60s "$PYTHON_BIN" - "$RUN_ROOT" "$SOURCE_ROOT" "$FLOW" <<'PY'
import csv, json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
flow = sys.argv[3]
metrics = [
    ("Reward", "test_reward_over", True),
    ("AQL", "test_avg_queue_len_over", False),
    ("AWT", "test_avg_waiting_time_over", False),
    ("ATT", "test_avg_travel_time_over", False),
    ("P95 wait", "test_vehicle_wait_p95_over", False),
    ("Max wait", "test_vehicle_wait_max_over", False),
]

def load_rows(base: Path, selector: str):
    rows = []
    incomplete = []
    for arm_dir in sorted((base / "arms").glob("*")):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        for seed_dir in sorted(arm_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            train_seed = int(seed_dir.name.removeprefix("seed_"))
            selected_name = ""
            if selector == "short_online":
                selection_path = seed_dir / "artifacts" / "online_checkpoint_selection.json"
                if selection_path.is_file():
                    try:
                        selected_name = json.loads(selection_path.read_text(encoding="utf-8")).get("selected", {}).get("name", "")
                    except Exception:
                        selected_name = "unreadable"
            for eval_seed in (2024, 42, 123):
                result = seed_dir / "evaluation" / flow.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
                base_row = {
                    "selector": selector,
                    "arm": arm,
                    "training_seed": train_seed,
                    "eval_seed": eval_seed,
                    "selected_checkpoint_name": selected_name,
                    "result_path": str(result),
                }
                if not result.is_file():
                    incomplete.append({**base_row, "reason": "missing_result"})
                    continue
                try:
                    payload = json.loads(result.read_text(encoding="utf-8"))
                    values = {key: float(payload["results"][key]) for _, key, _ in metrics}
                    rows.append({**base_row, **values})
                except Exception as exc:
                    incomplete.append({**base_row, "reason": type(exc).__name__})
    return rows, incomplete

val_rows, val_incomplete = load_rows(root, "offline_val_acc")
short_rows, short_incomplete = load_rows(source, "short_online")
rows = val_rows + short_rows
incomplete = val_incomplete + short_incomplete

fields = ["selector", "arm", "training_seed", "eval_seed", "selected_checkpoint_name", *[key for _, key, _ in metrics], "result_path"]
with (root / "combined_online_results.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with (root / "incomplete_runs.csv").open("w", encoding="utf-8", newline="") as f:
    fields_incomplete = ["selector", "arm", "training_seed", "eval_seed", "selected_checkpoint_name", "result_path", "reason"]
    writer = csv.DictWriter(f, fieldnames=fields_incomplete)
    writer.writeheader()
    writer.writerows(incomplete)

primary = [r for r in rows if r["eval_seed"] == 2024]
summary = {}
for selector in sorted({r["selector"] for r in primary}):
    summary[selector] = {}
    for arm in sorted({r["arm"] for r in primary if r["selector"] == selector}):
        selected = [r for r in primary if r["selector"] == selector and r["arm"] == arm]
        summary[selector][arm] = {}
        for _, key, _ in metrics:
            vals = [r[key] for r in selected]
            summary[selector][arm][key] = {
                "n": len(vals),
                "mean": statistics.fmean(vals) if vals else None,
                "std": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None),
                "values": vals,
            }

paired = {}
for arm in sorted({r["arm"] for r in primary}):
    paired[arm] = {}
    for label, key, higher in metrics:
        benefits = []
        for seed in (2024, 42, 123):
            short = [r for r in primary if r["selector"] == "short_online" and r["arm"] == arm and r["training_seed"] == seed]
            val = [r for r in primary if r["selector"] == "offline_val_acc" and r["arm"] == arm and r["training_seed"] == seed]
            if not short or not val:
                continue
            delta = short[0][key] - val[0][key]
            benefit = delta if higher else -delta
            benefits.append(benefit)
        paired[arm][key] = {
            "label": label,
            "n": len(benefits),
            "mean_benefit": statistics.fmean(benefits) if benefits else None,
            "std": statistics.stdev(benefits) if len(benefits) > 1 else (0.0 if benefits else None),
            "values": benefits,
            "short_online_wins": sum(v > 0 for v in benefits),
        }

aggregate = {
    "schema": "jinan_online_checkpoint_selection_valacc_comparator_v1",
    "flow": flow,
    "primary_eval_seed": 2024,
    "completed_offline_val_acc_rows": len(val_rows),
    "completed_short_online_rows": len(short_rows),
    "incomplete_rows": incomplete,
    "primary_by_selector_arm": summary,
    "paired_short_online_benefit_vs_offline_val_acc": paired,
}
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def fmt(stat):
    return "NA" if stat["mean"] is None else f"{stat['mean']:.3f} +/- {stat['std']:.3f}"

lines = [
    "# Jinan Offline-ValAcc Comparator Aggregate",
    "",
    f"Flow: `{flow}`. Primary eval seed: `2024`. Compares short-online selected checkpoints against saved offline-`val_acc` backups.",
    "",
    "| Selector | Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for selector in ("offline_val_acc", "short_online"):
    for arm in sorted(summary.get(selector, {})):
        stats = summary[selector][arm]
        lines.append(
            f"| `{selector}` | `{arm}` | {stats['test_reward_over']['n']} | {fmt(stats['test_reward_over'])} | {fmt(stats['test_avg_queue_len_over'])} | {fmt(stats['test_avg_waiting_time_over'])} | {fmt(stats['test_avg_travel_time_over'])} | {fmt(stats['test_vehicle_wait_p95_over'])} | {fmt(stats['test_vehicle_wait_max_over'])} |"
        )
lines += [
    "",
    "Paired benefits are short-online selector minus offline-`val_acc` selector for reward, and offline-`val_acc` minus short-online for lower-is-better metrics.",
    "",
    "| Arm | Metric | Mean benefit | Std | Short-online wins |",
    "|---|---|---:|---:|---:|",
]
for arm in sorted(paired):
    for label, key, _ in metrics:
        item = paired[arm][key]
        mean = "NA" if item["mean_benefit"] is None else f"{item['mean_benefit']:.3f}"
        std = "NA" if item["std"] is None else f"{item['std']:.3f}"
        lines.append(f"| `{arm}` | {label} | {mean} | {std} | {item['short_online_wins']} / {item['n']} |")
lines += ["", f"Completed offline-val_acc online rows: {len(val_rows)}. Incomplete rows: {len(incomplete)}."]
(root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"completed_offline_val_acc_rows": len(val_rows), "completed_short_online_rows": len(short_rows), "incomplete_rows": len(incomplete)}))
PY
}

on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\nsource_root=%s\n' "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" "$SOURCE_ROOT" > "$STATUS"
    notify "Jinan offline-val_acc comparator ${outcome} (exit=${code}): $(completed_cells)/6 cells complete. Run root: $RUN_ROOT. $(latest_metric)."
    trap - EXIT
    exit "$code"
}
trap 'exit 143' HUP INT TERM
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
exec >> "$LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\nsource_root=%s\n' "$(timestamp)" "$RUN_ROOT" "$SOURCE_ROOT" > "$STATUS"
preflight
write_provenance
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "Jinan offline-val_acc comparator launched: source=${SOURCE_ROOT}, run_root=${RUN_ROOT}, arms=${ARMS[*]}, train_seeds=${TRAINING_SEEDS[*]}."
for arm in "${ARMS[@]}"; do
    for seed in "${TRAINING_SEEDS[@]}"; do
        run_cell "$arm" "$seed"
    done
done
aggregate_results
