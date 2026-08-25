#!/usr/bin/env bash
# Re-run short-online checkpoint selection with held-out selector seed 42.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${JINAN_SELECTOR_SEED42_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
RUN_ROOT="${1:-$REPO_ROOT/records/pipeline_runs/jinan_online_checkpoint_selection_selector_seed42_20260823}"
SOURCE_ROOT="${JINAN_SELECTOR_SEED42_SOURCE_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2}"
VALACC_ROOT="${JINAN_SELECTOR_SEED42_VALACC_ROOT:-$REPO_ROOT/records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823}"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_wait25_improvement_online_suite.sh"
SELECT_SCRIPT="$REPO_ROOT/scripts/select_projection_checkpoint.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
FLOW=anon_3_4_jinan_real.json
ARMS=(wait2_mixed_rank wait225_mixed_rank)
TRAINING_SEEDS=(2024 42 123)
EVAL_SEEDS=(2024 42 123)
SELECTOR_SEED="${JINAN_SELECTOR_SEED42_SELECTOR_SEED:-42}"
SHORT_RUN_COUNTS="${JINAN_SELECTOR_SEED42_SHORT_RUN_COUNTS:-1200}"
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
print("latest selector-seed42 final seed2024 reward={:.3f} AQL={:.3f} AWT={:.3f} ATT={:.3f} P95={:.3f} Max={:.3f}".format(
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
    notify "Jinan selector-seed42 progress: $(completed_cells)/6 cells complete; current=$1; $(latest_metric)."
}

source_cell() {
    printf '%s/arms/%s/seed_%s' "$SOURCE_ROOT" "$1" "$2"
}

preflight() {
    [[ -x "$PYTHON_BIN" && -x "$ONLINE_SUITE" && -s "$SELECT_SCRIPT" && -d "$SOURCE_ROOT" ]] || { echo "Missing selector-seed42 prerequisite" >&2; return 1; }
    [[ -d "$VALACC_ROOT" ]] || { echo "Missing offline-val_acc comparator root: $VALACC_ROOT" >&2; return 1; }
    local arm seed cell
    for arm in "${ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            cell="$(source_cell "$arm" "$seed")"
            [[ -d "$cell/checkpoints" ]] || { echo "Missing source checkpoints: $cell/checkpoints" >&2; return 1; }
            compgen -G "$cell/checkpoints/epoch_*.pt" > /dev/null || { echo "Missing epoch checkpoints: $cell/checkpoints" >&2; return 1; }
        done
    done
    discover_idle_gpus
}

write_provenance() {
    local git_head git_diff_hash path key
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    {
        printf 'schema=jinan_online_checkpoint_selection_selector_seed42_v1\n'
        printf 'created_at=%s\nrun_root=%s\nsource_root=%s\nvalacc_root=%s\n' "$(timestamp)" "$RUN_ROOT" "$SOURCE_ROOT" "$VALACC_ROOT"
        printf 'git_head=%s\ngit_diff_sha256=%s\n' "$git_head" "$git_diff_hash"
        printf 'training_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_only\n' "$FLOW" "$FLOW"
        printf 'checkpoint_selector=short_online_reward\nselector_seed=%s\nshort_run_counts=%s\n' "$SELECTOR_SEED" "$SHORT_RUN_COUNTS"
        printf 'candidate_glob=checkpoints/epoch_*.pt\n'
        printf 'primary_final_eval_seed=2024\nheld_out_primary_seed=true\n'
        printf 'arms=%s\ntraining_seeds=%s\neval_seeds=%s\n' "${ARMS[*]}" "${TRAINING_SEEDS[*]}" "${EVAL_SEEDS[*]}"
        for path in "$SELECT_SCRIPT" "$ONLINE_SUITE" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/scripts/run_jinan_online_checkpoint_selection_selector_seed42_campaign.sh"; do
            key="${path#"$REPO_ROOT"/}"
            key="${key//\//_}"
            printf 'file_%s_sha256=%s\n' "$key" "$(sha256sum "$path" | awk '{print $1}')"
        done
    } > "$PROVENANCE"
}

short_eval_one() {
    local arm="$1" seed="$2" ckpt="$3" artifact_dir="$4" log_dir="$5" gpu="$6" stem result log
    stem="$(basename "$ckpt" .pt)"
    result="$artifact_dir/short_eval_seed${SELECTOR_SEED}_${stem}.json"
    log="$log_dir/short_eval_seed${SELECTOR_SEED}_${stem}.log"
    [[ -s "$result" ]] && return 0
    printf '[%s] SHORT_EVAL arm=%s seed=%s checkpoint=%s selector_seed=%s gpu=%s\n' "$(timestamp)" "$arm" "$seed" "$stem" "$SELECTOR_SEED" "$gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
        timeout --foreground 1800s "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
        --memo "jinan_selector_seed${SELECTOR_SEED}_${arm}_train${seed}_short_${stem}" \
        --projection_ckpt "$ckpt" \
        --llm_path /home/data/model/qwen-vl-8B \
        --dataset jinan \
        --traffic_file "$FLOW" \
        --run_counts "$SHORT_RUN_COUNTS" \
        --num_rounds 1 \
        --seed "$SELECTOR_SEED" \
        --prompt_mode queue_wait_short \
        --seq_len 16 \
        --image_size 224 \
        --periodicity 16 \
        --input_dim 8 \
        --hidden_dim 64 \
        --prefix_tokens 8 \
        --prefix_mode prefix \
        --max_length 64 \
        --inference_batch_size 64 \
        --bf16 \
        --deterministic \
        --disable_snapshots \
        --result_json "$result" > "$log" 2>&1
}

run_short_evals() {
    local arm="$1" seed="$2" source="$3" artifact_dir="$4" log_dir="$5" ckpts=() start slot idx pid gpu
    mapfile -t ckpts < <(find "$source/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' | sort)
    ((${#ckpts[@]} > 0)) || { echo "No epoch checkpoints found: $source/checkpoints" >&2; return 1; }
    discover_idle_gpus
    for ((start=0; start<${#ckpts[@]}; start+=3)); do
        local pids=()
        for slot in 0 1 2; do
            idx=$((start + slot))
            (( idx < ${#ckpts[@]} )) || continue
            gpu="${SELECTED_GPUS[$slot]}"
            short_eval_one "$arm" "$seed" "${ckpts[$idx]}" "$artifact_dir" "$log_dir" "$gpu" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        maybe_notify_progress "$arm/seed_$seed short_eval"
    done
}

selected_info() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
selected = data["selected"]
results = selected.get("results", {})
print("{}\t{}\t{}\t{}\t{}\t{}".format(
    selected.get("name", ""),
    selected["checkpoint_path"],
    selected.get("score", ""),
    results.get("test_reward_over", ""),
    results.get("test_avg_queue_len_over", ""),
    results.get("test_vehicle_wait_p95_over", ""),
))
PY
}

run_cell() {
    local arm="$1" seed="$2" target source target_ckpt status_path artifact_dir log_dir selection_json
    local selected_line selected_name selected_ckpt selected_score selected_reward selected_aql selected_p95 gpu_list
    target="$RUN_ROOT/arms/$arm/seed_$seed"
    source="$(source_cell "$arm" "$seed")"
    artifact_dir="$target/artifacts"
    log_dir="$target/logs"
    target_ckpt="$target/checkpoints/selected.selector_seed${SELECTOR_SEED}.pt"
    status_path="$target/status.txt"
    selection_json="$artifact_dir/selector_seed${SELECTOR_SEED}_online_checkpoint_selection.json"
    if [[ -f "$status_path" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$status_path"; then
        printf '[%s] SKIP arm=%s seed=%s already successful\n' "$(timestamp)" "$arm" "$seed"
        return 0
    fi
    [[ ! -e "$target" ]] || { echo "Existing incomplete target cannot be safely overwritten: $target" >&2; return 1; }
    mkdir -p "$target/checkpoints" "$artifact_dir" "$log_dir"
    printf 'started_at=%s\nstatus=running\narm=%s\ntraining_seed=%s\ncheckpoint_selector=short_online_reward\nselector_seed=%s\nsource_cell=%s\nshort_run_counts=%s\n' \
        "$(timestamp)" "$arm" "$seed" "$SELECTOR_SEED" "$source" "$SHORT_RUN_COUNTS" > "$status_path"

    run_short_evals "$arm" "$seed" "$source" "$artifact_dir" "$log_dir"
    printf '[%s] SELECT arm=%s seed=%s selector_seed=%s\n' "$(timestamp)" "$arm" "$seed" "$SELECTOR_SEED"
    timeout --foreground 300s "$PYTHON_BIN" "$SELECT_SCRIPT" \
        --stage_root "$target" \
        --ranking_metric reward \
        --eval_glob "artifacts/short_eval_seed${SELECTOR_SEED}_epoch_*.json" \
        --allow_external_ckpt \
        --fallback_to_unconstrained \
        --dry_run \
        --output_json "$selection_json" > "$log_dir/select_seed${SELECTOR_SEED}_online_best.log" 2>&1
    selected_line="$(selected_info "$selection_json")"
    IFS=$'\t' read -r selected_name selected_ckpt selected_score selected_reward selected_aql selected_p95 <<< "$selected_line"
    [[ -s "$selected_ckpt" ]] || { echo "Selected checkpoint missing: $selected_ckpt" >&2; return 1; }
    cp -p "$selected_ckpt" "$target_ckpt"
    printf 'selected_name=%s\nselected_score=%s\nselected_short_reward=%s\nselected_short_aql=%s\nselected_short_p95=%s\nselected_source_checkpoint=%s\nselected_source_sha256=%s\ncopied_checkpoint=%s\ncopied_checkpoint_sha256=%s\n' \
        "$selected_name" "$selected_score" "$selected_reward" "$selected_aql" "$selected_p95" "$selected_ckpt" "$(sha256sum "$selected_ckpt" | awk '{print $1}')" "$target_ckpt" "$(sha256sum "$target_ckpt" | awk '{print $1}')" > "$artifact_dir/selected_checkpoint.env"

    discover_idle_gpus
    gpu_list="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]:0:3}")"
    printf '[%s] START_FINAL arm=%s seed=%s selected=%s short_reward=%s eval_gpus=%s\n' "$(timestamp)" "$arm" "$seed" "$selected_name" "$selected_reward" "$gpu_list"
    notify "Jinan selector-seed${SELECTOR_SEED} final launched: arm=${arm}, train_seed=${seed}, selected=${selected_name}, short_reward=${selected_reward}, gpus=${gpu_list}."
    WAIT25_IMPROVEMENT_EVAL_GPUS="$gpu_list" \
    WAIT25_IMPROVEMENT_HIDDEN_DIM=64 \
    WAIT25_IMPROVEMENT_PREFIX_TOKENS=8 \
        "$ONLINE_SUITE" "$RUN_ROOT" "$target" "$target_ckpt" "jinan_selector_seed${SELECTOR_SEED}_${arm}_train${seed}"
    printf 'completed_at=%s\nstatus=success\narm=%s\ntraining_seed=%s\ncheckpoint_selector=short_online_reward\nselector_seed=%s\nselected_name=%s\nselected_score=%s\nselected_short_reward=%s\nselected_source_checkpoint=%s\nselected_source_sha256=%s\ncopied_checkpoint=%s\ncopied_checkpoint_sha256=%s\nonline_status=success\n' \
        "$(timestamp)" "$arm" "$seed" "$SELECTOR_SEED" "$selected_name" "$selected_score" "$selected_reward" "$selected_ckpt" "$(sha256sum "$selected_ckpt" | awk '{print $1}')" "$target_ckpt" "$(sha256sum "$target_ckpt" | awk '{print $1}')" > "$status_path"
    printf '[%s] DONE arm=%s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    maybe_notify_progress "$arm/seed_$seed"
}

aggregate_results() {
    timeout --foreground 60s "$PYTHON_BIN" - "$RUN_ROOT" "$SOURCE_ROOT" "$VALACC_ROOT" "$FLOW" "$SELECTOR_SEED" <<'PY'
import csv, json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
valacc = Path(sys.argv[3])
flow = sys.argv[4]
selector_seed = sys.argv[5]
eval_seeds = (2024, 42, 123)
metrics = [
    ("Reward", "test_reward_over", True),
    ("AQL", "test_avg_queue_len_over", False),
    ("AWT", "test_avg_waiting_time_over", False),
    ("ATT", "test_avg_travel_time_over", False),
    ("P95 wait", "test_vehicle_wait_p95_over", False),
    ("Max wait", "test_vehicle_wait_max_over", False),
]

def selection_name(seed_dir: Path, selector: str) -> str:
    if selector == "short_online_seed42":
        path = seed_dir / "artifacts" / f"selector_seed{selector_seed}_online_checkpoint_selection.json"
    elif selector == "short_online_seed2024":
        path = seed_dir / "artifacts" / "online_checkpoint_selection.json"
    else:
        return "offline_val_acc"
    if not path.is_file():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("selected", {}).get("name", ""))
    except Exception:
        return "unreadable"

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
            selected = selection_name(seed_dir, selector)
            for eval_seed in eval_seeds:
                result = seed_dir / "evaluation" / flow.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
                base_row = {
                    "selector": selector,
                    "arm": arm,
                    "training_seed": train_seed,
                    "eval_seed": eval_seed,
                    "selected_checkpoint_name": selected,
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

seed42_rows, seed42_incomplete = load_rows(root, "short_online_seed42")
seed2024_rows, seed2024_incomplete = load_rows(source, "short_online_seed2024")
val_rows, val_incomplete = load_rows(valacc, "offline_val_acc")
rows = seed42_rows + seed2024_rows + val_rows
incomplete = seed42_incomplete + seed2024_incomplete + val_incomplete

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

primary = [row for row in rows if row["eval_seed"] == 2024]
summary = {}
for selector in ("offline_val_acc", "short_online_seed2024", "short_online_seed42"):
    summary[selector] = {}
    for arm in sorted({r["arm"] for r in primary if r["selector"] == selector}):
        selected_rows = [r for r in primary if r["selector"] == selector and r["arm"] == arm]
        summary[selector][arm] = {}
        for _, key, _ in metrics:
            values = [r[key] for r in selected_rows]
            summary[selector][arm][key] = {
                "n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                "values": values,
            }

def paired_benefits(candidate_selector: str, baseline_selector: str):
    out = {}
    for arm in sorted({r["arm"] for r in primary}):
        out[arm] = {}
        for label, key, higher in metrics:
            benefits = []
            for seed in (2024, 42, 123):
                cand = [r for r in primary if r["selector"] == candidate_selector and r["arm"] == arm and r["training_seed"] == seed]
                base = [r for r in primary if r["selector"] == baseline_selector and r["arm"] == arm and r["training_seed"] == seed]
                if not cand or not base:
                    continue
                delta = cand[0][key] - base[0][key]
                benefits.append(delta if higher else -delta)
            out[arm][key] = {
                "label": label,
                "n": len(benefits),
                "mean_benefit": statistics.fmean(benefits) if benefits else None,
                "std": statistics.stdev(benefits) if len(benefits) > 1 else (0.0 if benefits else None),
                "values": benefits,
                "wins": sum(value > 0 for value in benefits),
            }
    return out

aggregate = {
    "schema": "jinan_online_checkpoint_selection_selector_seed42_aggregate_v1",
    "flow": flow,
    "selector_seed": int(selector_seed),
    "primary_eval_seed": 2024,
    "completed_seed42_rows": len(seed42_rows),
    "completed_seed2024_rows": len(seed2024_rows),
    "completed_offline_val_acc_rows": len(val_rows),
    "incomplete_rows": incomplete,
    "primary_by_selector_arm": summary,
    "paired_seed42_benefit_vs_seed2024_selector": paired_benefits("short_online_seed42", "short_online_seed2024"),
    "paired_seed42_benefit_vs_offline_val_acc": paired_benefits("short_online_seed42", "offline_val_acc"),
}
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def fmt(stat):
    return "NA" if stat["mean"] is None else f"{stat['mean']:.3f} +/- {stat['std']:.3f}"

lines = [
    "# Jinan Selector-Seed42 Online Checkpoint Selection Aggregate",
    "",
    f"Flow: `{flow}`. Selector seed: `{selector_seed}`. Primary final eval seed: `2024`.",
    "",
    "| Selector | Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for selector in ("offline_val_acc", "short_online_seed2024", "short_online_seed42"):
    for arm in sorted(summary.get(selector, {})):
        stats = summary[selector][arm]
        lines.append(
            f"| `{selector}` | `{arm}` | {stats['test_reward_over']['n']} | {fmt(stats['test_reward_over'])} | {fmt(stats['test_avg_queue_len_over'])} | {fmt(stats['test_avg_waiting_time_over'])} | {fmt(stats['test_avg_travel_time_over'])} | {fmt(stats['test_vehicle_wait_p95_over'])} | {fmt(stats['test_vehicle_wait_max_over'])} |"
        )
for title, paired in (
    ("Selector-seed42 benefits vs selector-seed2024", aggregate["paired_seed42_benefit_vs_seed2024_selector"]),
    ("Selector-seed42 benefits vs offline-val_acc", aggregate["paired_seed42_benefit_vs_offline_val_acc"]),
):
    lines += [
        "",
        title + ". Benefits are seed42 minus baseline for reward, and baseline minus seed42 for lower-is-better metrics.",
        "",
        "| Arm | Metric | Mean benefit | Std | Seed42 wins |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in sorted(paired):
        for label, key, _ in metrics:
            item = paired[arm][key]
            mean = "NA" if item["mean_benefit"] is None else f"{item['mean_benefit']:.3f}"
            std = "NA" if item["std"] is None else f"{item['std']:.3f}"
            lines.append(f"| `{arm}` | {label} | {mean} | {std} | {item['wins']} / {item['n']} |")
lines += ["", f"Completed selector-seed42 rows: {len(seed42_rows)}. Incomplete rows: {len(incomplete)}."]
(root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"completed_seed42_rows": len(seed42_rows), "completed_seed2024_rows": len(seed2024_rows), "completed_offline_val_acc_rows": len(val_rows), "incomplete_rows": len(incomplete)}))
PY
}

on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\nsource_root=%s\nvalacc_root=%s\nselector_seed=%s\n' "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" "$SOURCE_ROOT" "$VALACC_ROOT" "$SELECTOR_SEED" > "$STATUS"
    notify "Jinan selector-seed${SELECTOR_SEED} campaign ${outcome} (exit=${code}): $(completed_cells)/6 cells complete. Run root: $RUN_ROOT. $(latest_metric)."
    trap - EXIT
    exit "$code"
}
trap 'exit 143' HUP INT TERM
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
exec >> "$LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\nsource_root=%s\nvalacc_root=%s\nselector_seed=%s\n' "$(timestamp)" "$RUN_ROOT" "$SOURCE_ROOT" "$VALACC_ROOT" "$SELECTOR_SEED" > "$STATUS"
preflight
write_provenance
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "Jinan selector-seed${SELECTOR_SEED} campaign launched: source=${SOURCE_ROOT}, run_root=${RUN_ROOT}, short_run_counts=${SHORT_RUN_COUNTS}, arms=${ARMS[*]}, train_seeds=${TRAINING_SEEDS[*]}."
for arm in "${ARMS[@]}"; do
    for seed in "${TRAINING_SEEDS[@]}"; do
        run_cell "$arm" "$seed"
    done
done
aggregate_results
