#!/usr/bin/env bash
# Train and evaluate a frozen SimExpert reference on the Jinan real flow only.
set -Eeuo pipefail

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/jinan_real_frozen_simexpert_reference_20260822_sameflow}"
REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${FROZEN_SIMEXPERT_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /home/sysu/anaconda3/envs/hnr5090/bin/python ]]; then
        PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
if [[ -n "${FROZEN_SIMEXPERT_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$FROZEN_SIMEXPERT_LD_PRELOAD"
fi

FROZEN_DATA="$REPO_ROOT/records/sim_expert_validation_search/anon_3_4_jinan_real/best/traffic_prompt_series_2s.jsonl"
FROZEN_CONFIG="$REPO_ROOT/records/sim_expert_validation_search/anon_3_4_jinan_real/best_config.json"
EXPECTED_DATA_SHA256=d82e6deb12f444d94a44d282e23d5f217b3146c01022a5952885c741aa94c3f3
EXPECTED_CONFIG_SHA256=96274d54caed4e8e98e56a4216a063679ac88f4dfa6780a35e9697868a481b9e
STAGE1_CKPT="$REPO_ROOT/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
ARM_SCRIPT="$REPO_ROOT/scripts/launch_jinan_frozen_simexpert_reference_arm.sh"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_frozen_simexpert_online_suite.sh"
AGGREGATE_SCRIPT="$REPO_ROOT/scripts/aggregate_jinan_real_frozen_simexpert_reference.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
PROVENANCE_PATH="$RUN_ROOT/provenance.env"
TRAINING_FLOW=anon_3_4_jinan_real.json
EVALUATION_FLOW=anon_3_4_jinan_real.json
ARM=frozen_simexpert_mixed_rank
TRAINING_SEEDS=(2024 42 123)
LAST_NOTIFICATION_EPOCH=0
AVAILABLE_GPUS=()

timestamp() { date -Is; }

notify() {
    local message="$1"
    [[ -f "$FEISHU_SCRIPT" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$message" || true
}

discover_idle_gpus() {
    local gpu used
    AVAILABLE_GPUS=()
    while IFS=, read -r gpu used; do
        gpu="${gpu//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
        if (( used < 500 )); then
            AVAILABLE_GPUS+=("$gpu")
        fi
    done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    ((${#AVAILABLE_GPUS[@]} > 0)) || { echo "No idle GPU is visible to this runtime" >&2; return 1; }
    FROZEN_SIMEXPERT_GPUS="$(IFS=,; printf '%s' "${AVAILABLE_GPUS[*]}")"
    export FROZEN_SIMEXPERT_GPUS
    if [[ -z "${FROZEN_SIMEXPERT_EVAL_GPU:-}" ]]; then
        FROZEN_SIMEXPERT_EVAL_GPU="${AVAILABLE_GPUS[0]}"
    fi
    [[ " ${AVAILABLE_GPUS[*]} " =~ " ${FROZEN_SIMEXPERT_EVAL_GPU} " ]] || {
        echo "FROZEN_SIMEXPERT_EVAL_GPU=${FROZEN_SIMEXPERT_EVAL_GPU} is not idle" >&2
        return 1
    }
    export FROZEN_SIMEXPERT_EVAL_GPU
}

completed_count() {
    local count=0 status
    for status in "$RUN_ROOT"/arms/"$ARM"/seed_*/status.txt; do
        [[ -f "$status" ]] || continue
        if awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$status"; then
            count=$((count + 1))
        fi
    done
    printf '%s' "$count"
}

maybe_progress_notification() {
    local phase="$1"
    local now
    now="$(date +%s)"
    if (( now - LAST_NOTIFICATION_EPOCH < 2400 )); then
        return 0
    fi
    LAST_NOTIFICATION_EPOCH="$now"
    notify "jinan_real frozen SimExpert campaign progress: $(completed_count)/3 models complete; current=${phase}."
}

validate_frozen_source() {
    timeout --foreground 120s "$PYTHON_BIN" - "$FROZEN_DATA" "$FROZEN_CONFIG" "$REPO_ROOT" <<'PY'
import collections
import json
from pathlib import Path
import sys

data_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
repo_root = Path(sys.argv[3]).resolve()
phases = {"ETWT", "NTST", "ELWL", "NLSL"}
config = json.loads(config_path.read_text(encoding="utf-8"))
best = config["best"]
assert best["horizon_s"] == 60
assert best["expert_att_weight"] == 1.0
assert best["expert_aql_weight"] == 1.0
assert best["expert_awt_weight"] == 1.0
assert best["expert_spillback_weight"] == 0.0

rows = 0
terminal_rows = 0
label_counts = collections.Counter()
for line_number, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), start=1):
    record = json.loads(line)
    if "snapshot_path" not in record:
        assert "expert_action" not in record
        assert "candidate_action_metrics" not in record
        rows += 1
        continue
    action = record["expert_action"]
    assert action in phases, (line_number, action)
    metrics = record["candidate_action_metrics"]
    assert set(metrics) == phases, (line_number, sorted(metrics))
    for phase, values in metrics.items():
        assert {"att", "aql", "awt", "spillback"}.issubset(values), (line_number, phase)
        assert values["weight_att"] == 1.0
        assert values["weight_aql"] == 1.0
        assert values["weight_awt"] == 1.0
        assert values["weight_spillback"] == 0.0
    snapshot = (repo_root / record["snapshot_path"]).resolve()
    assert snapshot.is_relative_to(repo_root)
    assert snapshot.is_file(), snapshot
    terminal_rows += 1
    label_counts[action] += 1
    rows += 1

assert rows == 22860, rows
assert terminal_rows == 1440, terminal_rows
assert label_counts == {"ETWT": 555, "NTST": 507, "ELWL": 197, "NLSL": 181}, label_counts
print(json.dumps({"rows": rows, "terminal_labels": terminal_rows, "label_counts": label_counts}, sort_keys=True))
PY
}

preflight() {
    [[ -s "$FROZEN_DATA" ]] || { echo "Missing frozen SimExpert JSONL: $FROZEN_DATA" >&2; return 1; }
    [[ -s "$FROZEN_CONFIG" ]] || { echo "Missing frozen SimExpert config: $FROZEN_CONFIG" >&2; return 1; }
    [[ "$(sha256sum "$FROZEN_DATA" | awk '{print $1}')" == "$EXPECTED_DATA_SHA256" ]] || { echo "Frozen JSONL hash mismatch" >&2; return 1; }
    [[ "$(sha256sum "$FROZEN_CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] || { echo "Frozen config hash mismatch" >&2; return 1; }
    [[ -s "$STAGE1_CKPT" ]] || { echo "Missing Jinan Stage 1 checkpoint: $STAGE1_CKPT" >&2; return 1; }
    [[ -d /home/data/model/qwen-vl-8B ]] || { echo "Missing model: /home/data/model/qwen-vl-8B" >&2; return 1; }
    [[ -s "$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json" ]] || { echo "Missing Jinan roadnet" >&2; return 1; }
    [[ -s "$REPO_ROOT/data/Jinan/3_4/$TRAINING_FLOW" ]] || { echo "Missing Jinan real traffic file" >&2; return 1; }
    [[ "$TRAINING_FLOW" == "$EVALUATION_FLOW" ]] || { echo "Training and evaluation flow must match" >&2; return 1; }
    validate_frozen_source
    discover_idle_gpus
    printf 'Detected idle GPUs: %s; fixed online evaluation GPU: %s; flow=%s\n' \
        "$FROZEN_SIMEXPERT_GPUS" "$FROZEN_SIMEXPERT_EVAL_GPU" "$TRAINING_FLOW"
}

write_provenance() {
    local path relative git_head git_diff_hash snapshot_path
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    snapshot_path="$(timeout --foreground 30s "$PYTHON_BIN" - "$FROZEN_DATA" <<'PY'
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    print(json.loads(line)["snapshot_path"])
    break
PY
)"
    [[ -n "$snapshot_path" && -s "$REPO_ROOT/$snapshot_path" ]] || { echo "Cannot resolve frozen snapshot" >&2; return 1; }
    {
        printf 'created_at=%s\n' "$(timestamp)"
        printf 'git_head=%s\ngit_diff_sha256=%s\n' "$git_head" "$git_diff_hash"
        printf 'frozen_source_jsonl=%s\nfrozen_source_jsonl_sha256=%s\n' "$FROZEN_DATA" "$(sha256sum "$FROZEN_DATA" | awk '{print $1}')"
        printf 'frozen_source_config=%s\nfrozen_source_config_sha256=%s\n' "$FROZEN_CONFIG" "$(sha256sum "$FROZEN_CONFIG" | awk '{print $1}')"
        printf 'frozen_snapshot_example=%s\nfrozen_snapshot_example_sha256=%s\n' "$snapshot_path" "$(sha256sum "$REPO_ROOT/$snapshot_path" | awk '{print $1}')"
        printf 'frozen_label_semantics=SimExpert rollout teacher; composite minmax cost with ATT=1,AQL=1,AWT=1,spillback=0\n'
        printf 'training_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_only\n' "$TRAINING_FLOW" "$EVALUATION_FLOW"
        printf 'stage1_checkpoint=%s\nstage1_checkpoint_sha256=%s\n' "$STAGE1_CKPT" "$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')"
        printf 'online_eval_gpu=%s\nonline_eval_seed_execution=serial_single_gpu\n' "$FROZEN_SIMEXPERT_EVAL_GPU"
        for path in \
            "$ARM_SCRIPT" \
            "$ONLINE_SUITE" \
            "$AGGREGATE_SCRIPT" \
            "$REPO_ROOT/projection_train/train_projection_bc.py" \
            "$REPO_ROOT/projection_train/run_projection_online.py" \
            "$REPO_ROOT/projection_train/cost_ranking.py" \
            "$REPO_ROOT/projection_train/grouped_dataset.py" \
            "$REPO_ROOT/data_split.py"; do
            relative="${path#"$REPO_ROOT"/}"
            relative="${relative//\//_}"
            printf 'file_%s_sha256=%s\n' "$relative" "$(sha256sum "$path" | awk '{print $1}')"
        done
    } > "$PROVENANCE_PATH"
    printf '[%s] wrote provenance manifest: %s\n' "$(timestamp)" "$PROVENANCE_PATH"
}

run_arm() {
    local seed="$1"
    local arm_root="$RUN_ROOT/arms/$ARM/seed_$seed"
    if [[ -f "$arm_root/status.txt" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$arm_root/status.txt"; then
        printf '[%s] arm already complete: %s seed=%s\n' "$(timestamp)" "$ARM" "$seed"
        return 0
    fi
    [[ ! -e "$arm_root" ]] || { echo "Refusing to overwrite incomplete arm: $arm_root" >&2; return 1; }
    printf '[%s] launching Stage 2 arm=%s seed=%s flow=%s\n' "$(timestamp)" "$ARM" "$seed" "$TRAINING_FLOW"
    setsid bash "$ARM_SCRIPT" "$RUN_ROOT" "$ARM" "$seed" &
    local child_pid=$!
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep 60
        maybe_progress_notification "${ARM}/seed_${seed}"
    done
    wait "$child_pid"
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\nrun_root=%s\ntraining_flow=%s\nevaluation_flow=%s\n' \
            "$(timestamp)" "$RUN_ROOT" "$TRAINING_FLOW" "$EVALUATION_FLOW" > "$CAMPAIGN_STATUS"
        notify "jinan_real frozen SimExpert reference campaign completed: 3 Stage-2 models and 9 same-flow online results. Aggregate: $RUN_ROOT/AGGREGATE.md"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\nrun_root=%s\ntraining_flow=%s\nevaluation_flow=%s\n' \
            "$(timestamp)" "$code" "$RUN_ROOT" "$TRAINING_FLOW" "$EVALUATION_FLOW" > "$CAMPAIGN_STATUS"
        notify "jinan_real frozen SimExpert reference campaign stopped with exit_code=${code}; inspect $CAMPAIGN_LOG"
    fi
    trap - EXIT
    exit "$code"
}
trap 'exit 143' HUP INT TERM
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists; refusing to overwrite: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
exec >> "$CAMPAIGN_LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\ntraining_flow=%s\nevaluation_flow=%s\n' \
    "$(timestamp)" "$RUN_ROOT" "$TRAINING_FLOW" "$EVALUATION_FLOW" > "$CAMPAIGN_STATUS"
LAST_NOTIFICATION_EPOCH="$(date +%s)"
preflight
write_provenance
notify "jinan_real frozen SimExpert reference campaign launched: frozen teacher, train_seeds=2024/42/123, training_flow=${TRAINING_FLOW}, evaluation_flow=${EVALUATION_FLOW}, training_gpus=${FROZEN_SIMEXPERT_GPUS}, online_gpu=${FROZEN_SIMEXPERT_EVAL_GPU}."
for seed in "${TRAINING_SEEDS[@]}"; do
    run_arm "$seed"
done
timeout --foreground 120s "$PYTHON_BIN" "$AGGREGATE_SCRIPT" --run-root "$RUN_ROOT"
