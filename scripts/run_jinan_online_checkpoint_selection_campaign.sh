#!/usr/bin/env bash
# Run Jinan same-flow online checkpoint selection for wait2/wait225.
set -Eeuo pipefail

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_20260822}"
REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${JINAN_ONLINE_CKPT_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN_BIN="${JINAN_ONLINE_CKPT_TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH=/home/data/model/qwen-vl-8B
DATA_PATH="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl"
DATA_SUMMARY="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.summary.json"
EXPECTED_DATA_SHA256=793df872a9f81a095161303f8655c6fdd9c9794fc675ff0feea841435cdf2dae
ALIGN_CKPT="${JINAN_ONLINE_CKPT_ALIGN_CKPT:-$REPO_ROOT/records/pipeline_runs/jinan_direct_urgency_five_tracks_20260822_recovered_cohort_r3/baseline_threeflow/stage1/best.pt}"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_wait25_improvement_online_suite.sh"
SELECT_SCRIPT="$REPO_ROOT/scripts/select_projection_checkpoint.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
PROVENANCE="$RUN_ROOT/provenance.env"
FLOW=anon_3_4_jinan_real.json
SHORT_RUN_COUNTS="${JINAN_ONLINE_CKPT_SHORT_RUN_COUNTS:-1200}"
SHORT_EVAL_SEED="${JINAN_ONLINE_CKPT_SHORT_EVAL_SEED:-2024}"
ARMS=(wait2_mixed_rank wait225_mixed_rank)
TRAINING_SEEDS=(2024 42 123)
EVAL_SEEDS=(2024 42 123)
GLOBAL_EFFECTIVE_BATCH=96
PER_RANK_BATCH=4
HIDDEN_DIM=64
PREFIX_TOKENS=8
LAST_NOTIFICATION_EPOCH=0
SELECTED_GPUS=()

timestamp() { date -Is; }

notify() {
    [[ -f "$FEISHU_SCRIPT" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" || true
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
print("latest seed2024 reward={:.3f} AQL={:.3f} AWT={:.3f} ATT={:.3f} P95={:.3f}".format(
    float(r.get("test_reward_over")),
    float(r.get("test_avg_queue_len_over")),
    float(r.get("test_avg_waiting_time_over")),
    float(r.get("test_avg_travel_time_over")),
    float(r.get("test_vehicle_wait_p95_over")),
))
PY
}

maybe_notify_progress() {
    local now
    now="$(date +%s)"
    (( now - LAST_NOTIFICATION_EPOCH >= 2400 )) || return 0
    LAST_NOTIFICATION_EPOCH="$now"
    notify "Jinan online checkpoint selection progress: $(completed_cells)/6 cells complete; current=$1; $(latest_metric)."
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
    ((${#SELECTED_GPUS[@]} > 0)) || { echo "No idle GPU is available" >&2; return 1; }
}

validate_data() {
    timeout --foreground 60s "$PYTHON_BIN" - "$DATA_PATH" "$DATA_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
data = Path(sys.argv[1])
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert summary["rows"] == 22860
assert summary["terminal_labels"] == 1440
assert summary["full_candidate_metric_rows"] == 1440
rows = terminals = metrics = 0
for line in data.read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    rows += 1
    if "expert_action" not in item:
        continue
    terminals += 1
    metrics += int(set(item.get("candidate_action_metrics", {})) == {"ETWT", "NTST", "ELWL", "NLSL"})
assert rows == 22860 and terminals == 1440 and metrics == 1440
print({"rows": rows, "terminal_labels": terminals, "candidate_metric_rows": metrics})
PY
}

preflight() {
    [[ -x "$PYTHON_BIN" && -x "$TORCHRUN_BIN" ]] || { echo "Missing Python or torchrun" >&2; return 1; }
    [[ -d "$MODEL_PATH" && -s "$DATA_PATH" && -s "$DATA_SUMMARY" && -s "$ALIGN_CKPT" ]] || { echo "Missing model/data/stage1 prerequisite" >&2; return 1; }
    [[ -x "$ONLINE_SUITE" && -s "$SELECT_SCRIPT" && -s "$REPO_ROOT/data/Jinan/3_4/$FLOW" ]] || { echo "Missing online prerequisite" >&2; return 1; }
    [[ "$(sha256sum "$DATA_PATH" | awk '{print $1}')" == "$EXPECTED_DATA_SHA256" ]] || { echo "Direct dataset hash mismatch" >&2; return 1; }
    validate_data
    discover_idle_gpus
}

write_provenance() {
    local path key git_head git_diff_hash
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    {
        printf 'schema=jinan_online_checkpoint_selection_campaign_v1\n'
        printf 'created_at=%s\nrun_root=%s\ngit_head=%s\ngit_diff_sha256=%s\n' "$(timestamp)" "$RUN_ROOT" "$git_head" "$git_diff_hash"
        printf 'training_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_only\nshort_eval_seed=%s\nshort_run_counts=%s\nfinal_eval_seeds=%s\n' "$FLOW" "$FLOW" "$SHORT_EVAL_SEED" "$SHORT_RUN_COUNTS" "${EVAL_SEEDS[*]}"
        printf 'arms=%s\ntraining_seeds=%s\nselection=rereank_epoch_checkpoints_by_short_online_reward\n' "${ARMS[*]}" "${TRAINING_SEEDS[*]}"
        printf 'source_data=%s\nsource_data_sha256=%s\nsource_summary=%s\nsource_summary_sha256=%s\n' "$DATA_PATH" "$(sha256sum "$DATA_PATH" | awk '{print $1}')" "$DATA_SUMMARY" "$(sha256sum "$DATA_SUMMARY" | awk '{print $1}')"
        printf 'stage1_checkpoint=%s\nstage1_checkpoint_sha256=%s\n' "$ALIGN_CKPT" "$(sha256sum "$ALIGN_CKPT" | awk '{print $1}')"
        for path in "$SELECT_SCRIPT" "$ONLINE_SUITE" "$REPO_ROOT/projection_train/train_projection_bc.py" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/projection_train/cost_ranking.py" "$REPO_ROOT/projection_train/grouped_dataset.py" "$REPO_ROOT/data_split.py"; do
            key="${path#"$REPO_ROOT"/}"
            key="${key//\//_}"
            printf 'file_%s_sha256=%s\n' "$key" "$(sha256sum "$path" | awk '{print $1}')"
        done
    } > "$PROVENANCE"
}

target_args_for_arm() {
    case "$1" in
        wait2_mixed_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        wait225_mixed_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.25 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        *)
            echo "Unknown arm: $1" >&2
            return 2
            ;;
    esac
}

run_training() {
    local arm="$1" seed="$2" ckpt_dir="$3" log_dir="$4" gpu_list gpu_count grad_accum
    discover_idle_gpus
    gpu_count="${#SELECTED_GPUS[@]}"
    (( GLOBAL_EFFECTIVE_BATCH % (gpu_count * PER_RANK_BATCH) == 0 )) || { echo "Cannot preserve effective batch on ${gpu_count} GPUs" >&2; return 1; }
    grad_accum="$(( GLOBAL_EFFECTIVE_BATCH / (gpu_count * PER_RANK_BATCH) ))"
    gpu_list="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]}")"
    target_args_for_arm "$arm"
    printf '[%s] TRAIN arm=%s seed=%s gpus=%s grad_accum=%s\n' "$(timestamp)" "$arm" "$seed" "$gpu_list" "$grad_accum"
    env CUDA_VISIBLE_DEVICES="$gpu_list" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        timeout --foreground 7200s "$TORCHRUN_BIN" --standalone --nproc_per_node="$gpu_count" \
        projection_train/train_projection_bc.py \
        --data_path "$DATA_PATH" \
        --llm_path "$MODEL_PATH" \
        --align_ckpt "$ALIGN_CKPT" \
        --output_dir "$ckpt_dir" \
        --seq_len 16 \
        --image_size 224 \
        --periodicity 16 \
        --input_dim 8 \
        --hidden_dim "$HIDDEN_DIM" \
        --prefix_tokens "$PREFIX_TOKENS" \
        --prefix_mode prefix \
        --max_length 64 \
        --prompt_mode queue_wait_short \
        --batch_size "$PER_RANK_BATCH" \
        --grad_accum_steps "$grad_accum" \
        --epochs 15 \
        --lr 1e-4 \
        --min_lr 1e-5 \
        --warmup_ratio 0.1 \
        --lr_schedule cosine \
        --weight_decay 1e-4 \
        --max_grad_norm 1.0 \
        --val_ratio 0.1 \
        --split_strategy blocked \
        --validation_purge_samples 24 \
        --num_workers 4 \
        --seed "$seed" \
        --bf16 \
        --disable_class_balance \
        --log_interval 20 \
        --save_every 1 \
        --checkpoint_slim \
        --best_metric val_acc \
        --best_mode max \
        --ddp_timeout_seconds 1800 \
        "${TARGET_ARGS[@]}" > "$log_dir/train.log" 2>&1
}

short_eval_one() {
    local arm="$1" seed="$2" ckpt="$3" artifact_dir="$4" log_dir="$5" gpu="$6" stem result log
    stem="$(basename "$ckpt" .pt)"
    result="$artifact_dir/short_eval_${stem}.json"
    log="$log_dir/short_eval_${stem}.log"
    [[ -s "$result" ]] && return 0
    printf '[%s] SHORT_EVAL arm=%s seed=%s checkpoint=%s gpu=%s\n' "$(timestamp)" "$arm" "$seed" "$stem" "$gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
        timeout --foreground 1800s "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
        --memo "jinan_online_ckpt_${arm}_train${seed}_short_${stem}" \
        --projection_ckpt "$ckpt" \
        --llm_path "$MODEL_PATH" \
        --dataset jinan \
        --traffic_file "$FLOW" \
        --run_counts "$SHORT_RUN_COUNTS" \
        --num_rounds 1 \
        --seed "$SHORT_EVAL_SEED" \
        --prompt_mode queue_wait_short \
        --seq_len 16 \
        --image_size 224 \
        --periodicity 16 \
        --input_dim 8 \
        --hidden_dim "$HIDDEN_DIM" \
        --prefix_tokens "$PREFIX_TOKENS" \
        --prefix_mode prefix \
        --max_length 64 \
        --inference_batch_size 64 \
        --bf16 \
        --deterministic \
        --disable_snapshots \
        --result_json "$result" > "$log" 2>&1
}

run_short_selection() {
    local arm="$1" seed="$2" stage_root="$3" ckpt_dir="$4" artifact_dir="$5" log_dir="$6" ckpt gpu
    discover_idle_gpus
    gpu="${SELECTED_GPUS[0]}"
    for ckpt in "$ckpt_dir"/epoch_*.pt "$ckpt_dir"/best.pt "$ckpt_dir"/last.pt; do
        [[ -f "$ckpt" ]] || continue
        short_eval_one "$arm" "$seed" "$ckpt" "$artifact_dir" "$log_dir" "$gpu"
        maybe_notify_progress "$arm/seed_$seed short_eval"
    done
    printf '[%s] SELECT arm=%s seed=%s\n' "$(timestamp)" "$arm" "$seed"
    timeout --foreground 300s "$PYTHON_BIN" "$SELECT_SCRIPT" \
        --stage_root "$stage_root" \
        --ranking_metric reward \
        --eval_glob "artifacts/short_eval_*.json" \
        --promote_to_best \
        --fallback_to_unconstrained \
        --best_backup_name best.offline_val_acc.pt \
        --output_json "$artifact_dir/online_checkpoint_selection.json" > "$log_dir/select_online_best.log" 2>&1
}

run_final_online() {
    local arm="$1" seed="$2" stage_root="$3" ckpt_dir="$4"
    discover_idle_gpus
    if ((${#SELECTED_GPUS[@]} < 3)); then
        echo "Need 3 idle GPUs for parallel final online seeds; got ${#SELECTED_GPUS[@]}" >&2
        return 1
    fi
    WAIT25_IMPROVEMENT_EVAL_GPUS="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]:0:3}")" \
    WAIT25_IMPROVEMENT_HIDDEN_DIM="$HIDDEN_DIM" \
    WAIT25_IMPROVEMENT_PREFIX_TOKENS="$PREFIX_TOKENS" \
        "$ONLINE_SUITE" "$RUN_ROOT" "$stage_root" "$ckpt_dir/best.pt" "jinan_online_ckpt_${arm}_train${seed}_selected"
}

write_cell_status() {
    local status="$1" arm="$2" seed="$3" stage_root="$4" ckpt_dir="$5" status_path="$6"
    if [[ "$status" == success ]]; then
        printf 'completed_at=%s\nstatus=success\narm=%s\ntraining_seed=%s\ntraining_flow=%s\nevaluation_flow=%s\nshort_eval_seed=%s\nshort_run_counts=%s\nselected_checkpoint=%s\nselected_checkpoint_sha256=%s\nselection_report=%s\nonline_status=success\n' \
            "$(timestamp)" "$arm" "$seed" "$FLOW" "$FLOW" "$SHORT_EVAL_SEED" "$SHORT_RUN_COUNTS" "$ckpt_dir/best.pt" "$(sha256sum "$ckpt_dir/best.pt" | awk '{print $1}')" "$stage_root/artifacts/online_checkpoint_selection.json" > "$status_path"
    else
        printf 'completed_at=%s\nstatus=failed\narm=%s\ntraining_seed=%s\ntraining_flow=%s\nevaluation_flow=%s\n' \
            "$(timestamp)" "$arm" "$seed" "$FLOW" "$FLOW" > "$status_path"
    fi
}

run_cell() {
    local arm="$1" seed="$2" stage_root ckpt_dir log_dir artifact_dir status_path
    stage_root="$RUN_ROOT/arms/$arm/seed_$seed"
    ckpt_dir="$stage_root/checkpoints"
    log_dir="$stage_root/logs"
    artifact_dir="$stage_root/artifacts"
    status_path="$stage_root/status.txt"
    if [[ -f "$status_path" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$status_path"; then
        printf '[%s] SKIP arm=%s seed=%s already successful\n' "$(timestamp)" "$arm" "$seed"
        return 0
    fi
    [[ ! -e "$stage_root" ]] || { echo "Existing incomplete cell cannot be safely overwritten: $stage_root" >&2; return 1; }
    mkdir -p "$ckpt_dir" "$log_dir" "$artifact_dir"
    printf 'started_at=%s\nstatus=running\narm=%s\ntraining_seed=%s\ntraining_flow=%s\nevaluation_flow=%s\nshort_eval_seed=%s\nshort_run_counts=%s\n' \
        "$(timestamp)" "$arm" "$seed" "$FLOW" "$FLOW" "$SHORT_EVAL_SEED" "$SHORT_RUN_COUNTS" > "$status_path"
    notify "Jinan online checkpoint selection cell launched: arm=${arm}, train_seed=${seed}, short_eval_seed=${SHORT_EVAL_SEED}, run_root=${stage_root}."
    run_training "$arm" "$seed" "$ckpt_dir" "$log_dir"
    [[ -s "$ckpt_dir/best.pt" ]] || { echo "Training finished without best.pt: $ckpt_dir" >&2; return 1; }
    run_short_selection "$arm" "$seed" "$stage_root" "$ckpt_dir" "$artifact_dir" "$log_dir"
    [[ -s "$artifact_dir/online_checkpoint_selection.json" ]] || { echo "Missing selection report: $artifact_dir" >&2; return 1; }
    run_final_online "$arm" "$seed" "$stage_root" "$ckpt_dir"
    write_cell_status success "$arm" "$seed" "$stage_root" "$ckpt_dir" "$status_path"
    notify "Jinan online checkpoint selection cell complete: arm=${arm}, train_seed=${seed}; $(latest_metric)."
}

aggregate_results() {
    timeout --foreground 60s "$PYTHON_BIN" - "$RUN_ROOT" "$FLOW" <<'PY'
import csv, json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
flow = sys.argv[2]
metrics = [
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
]
rows = []
incomplete = []
for arm_dir in sorted((root / "arms").glob("*")):
    if not arm_dir.is_dir():
        continue
    for seed_dir in sorted(arm_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        arm = arm_dir.name
        train_seed = seed_dir.name.removeprefix("seed_")
        selection_path = seed_dir / "artifacts" / "online_checkpoint_selection.json"
        selected = ""
        if selection_path.is_file():
            try:
                selected = json.loads(selection_path.read_text(encoding="utf-8")).get("selected", {}).get("name", "")
            except Exception:
                selected = "unreadable"
        for eval_seed in (2024, 42, 123):
            result = seed_dir / "evaluation" / flow.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
            base = {"arm": arm, "training_seed": train_seed, "eval_seed": eval_seed, "selected_short_checkpoint": selected, "result_path": str(result)}
            if not result.is_file():
                incomplete.append({**base, "reason": "missing_result"})
                continue
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
                values = {metric: float(payload["results"][metric]) for metric in metrics}
                rows.append({**base, **values})
            except Exception as exc:
                incomplete.append({**base, "reason": type(exc).__name__})

with (root / "online_results.csv").open("w", encoding="utf-8", newline="") as f:
    fields = ["arm", "training_seed", "eval_seed", "selected_short_checkpoint", *metrics, "result_path"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with (root / "incomplete_runs.csv").open("w", encoding="utf-8", newline="") as f:
    fields = ["arm", "training_seed", "eval_seed", "selected_short_checkpoint", "result_path", "reason"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(incomplete)

primary = [row for row in rows if int(row["eval_seed"]) == 2024]
summary = {}
for arm in sorted({row["arm"] for row in primary}):
    summary[arm] = {}
    selected_rows = [row for row in primary if row["arm"] == arm]
    for metric in metrics:
        values = [row[metric] for row in selected_rows]
        summary[arm][metric] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "values": values,
        }
out = {"schema": "jinan_online_checkpoint_selection_aggregate_v1", "flow": flow, "primary_eval_seed": 2024, "completed_rows": len(rows), "incomplete_rows": incomplete, "primary_by_arm": summary}
(root / "aggregate.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Jinan Online Checkpoint Selection Aggregate",
    "",
    f"Flow: `{flow}`. Primary eval seed: `2024`. Short online selector ranks epoch checkpoints by reward.",
    "",
    "| Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
labels = {
    "test_reward_over": "Reward",
    "test_avg_queue_len_over": "AQL",
    "test_avg_waiting_time_over": "AWT",
    "test_avg_travel_time_over": "ATT",
    "test_vehicle_wait_p95_over": "P95",
    "test_vehicle_wait_max_over": "Max",
}
for arm, stats in summary.items():
    def fmt(metric):
        item = stats[metric]
        return "NA" if item["mean"] is None else f"{item['mean']:.3f} +/- {item['std']:.3f}"
    lines.append(
        f"| `{arm}` | {stats['test_reward_over']['n']} | {fmt('test_reward_over')} | {fmt('test_avg_queue_len_over')} | {fmt('test_avg_waiting_time_over')} | {fmt('test_avg_travel_time_over')} | {fmt('test_vehicle_wait_p95_over')} | {fmt('test_vehicle_wait_max_over')} |"
    )
lines += ["", f"Completed online rows: {len(rows)}. Incomplete rows: {len(incomplete)}."]
(root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"completed_rows": len(rows), "incomplete_rows": len(incomplete)}))
PY
}

on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\n' "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
    notify "Jinan online checkpoint selection campaign ${outcome} (exit=${code}): $(completed_cells)/6 cells complete. Run root: $RUN_ROOT. $(latest_metric)."
    trap - EXIT
    exit "$code"
}
trap 'exit 143' HUP INT TERM
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
exec >> "$CAMPAIGN_LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\n' "$(timestamp)" "$RUN_ROOT" > "$CAMPAIGN_STATUS"
preflight
write_provenance
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "Jinan online checkpoint selection campaign launched: arms=${ARMS[*]}, train_seeds=${TRAINING_SEEDS[*]}, short_eval_seed=${SHORT_EVAL_SEED}, short_run_counts=${SHORT_RUN_COUNTS}, run_root=${RUN_ROOT}."
for arm in "${ARMS[@]}"; do
    for seed in "${TRAINING_SEEDS[@]}"; do
        run_cell "$arm" "$seed"
    done
done
aggregate_results
