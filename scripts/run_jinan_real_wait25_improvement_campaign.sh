#!/usr/bin/env bash
# Run the controlled Jinan-real wait25 local-refinement campaign only.
set -Eeuo pipefail

RUN_ROOT="${1:-/home/code/LLMSTSC/records/pipeline_runs/jinan_real_wait25_improvement_20260822}"
REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN="${WAIT25_IMPROVEMENT_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
DATA_PATH="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.jsonl"
DATA_SUMMARY="$REPO_ROOT/records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/data/train_1440_direct_phase_urgency_v1.summary.json"
EXPECTED_DATA_SHA256=793df872a9f81a095161303f8655c6fdd9c9794fc675ff0feea841435cdf2dae
STAGE1_CKPT="$REPO_ROOT/records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt"
ARM_SCRIPT="$REPO_ROOT/scripts/launch_jinan_wait25_improvement_arm.sh"
ONLINE_SUITE="$REPO_ROOT/scripts/run_jinan_real_wait25_improvement_online_suite.sh"
AGGREGATE_SCRIPT="$REPO_ROOT/scripts/aggregate_jinan_real_wait25_improvement.py"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
PROVENANCE="$RUN_ROOT/provenance.env"
TRAINING_FLOW=anon_3_4_jinan_real.json
EVALUATION_FLOW=anon_3_4_jinan_real.json
ARMS=(wait225_mixed_rank wait25_conservative_rank wait25_strong_rank)
TRAINING_SEEDS=(2024 42 123)
AVAILABLE_GPUS=()
LAST_NOTIFICATION_EPOCH=0

timestamp() { date -Is; }
notify() { [[ -f "$FEISHU_SCRIPT" ]] && timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" || true; }

discover_idle_gpus() {
    local gpu used
    while IFS=, read -r gpu used; do
        gpu="${gpu//[[:space:]]/}"; used="${used//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] && (( used < 500 )) && AVAILABLE_GPUS+=("$gpu")
    done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    ((${#AVAILABLE_GPUS[@]} > 0)) || { echo "No idle GPU is visible" >&2; return 1; }
    WAIT25_IMPROVEMENT_GPUS="$(IFS=,; printf '%s' "${AVAILABLE_GPUS[*]}")"; export WAIT25_IMPROVEMENT_GPUS
    WAIT25_IMPROVEMENT_EVAL_GPU="${WAIT25_IMPROVEMENT_EVAL_GPU:-${AVAILABLE_GPUS[0]}}"; export WAIT25_IMPROVEMENT_EVAL_GPU
    [[ " ${AVAILABLE_GPUS[*]} " =~ " ${WAIT25_IMPROVEMENT_EVAL_GPU} " ]] || { echo "Configured evaluation GPU is not idle" >&2; return 1; }
}

completed_count() {
    local count=0 path
    for path in "$RUN_ROOT"/arms/*/seed_*/status.txt; do
        [[ -f "$path" ]] || continue
        awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$path" && count=$((count + 1))
    done
    printf '%s' "$count"
}

maybe_notify_progress() {
    local now; now="$(date +%s)"; (( now - LAST_NOTIFICATION_EPOCH >= 2400 )) || return 0
    LAST_NOTIFICATION_EPOCH="$now"; notify "jinan_real wait25 refinement campaign progress: $(completed_count)/9 models complete; current=$1."
}

validate_data() {
    timeout --foreground 60s "$PYTHON_BIN" - "$DATA_PATH" "$DATA_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
data = Path(sys.argv[1]); summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert summary["rows"] == 22860 and summary["terminal_labels"] == 1440 and summary["full_candidate_metric_rows"] == 1440
rows = terminals = contexts = metrics = 0
for line in data.read_text(encoding="utf-8").splitlines():
    item = json.loads(line); rows += 1
    if "expert_action" not in item: continue
    terminals += 1; contexts += int(bool(item.get("prompt_context"))); metrics += int(set(item.get("candidate_action_metrics", {})) == {"ETWT", "NTST", "ELWL", "NLSL"})
assert rows == 22860 and terminals == 1440 and contexts == 1440 and metrics == 1440
print({"rows": rows, "terminal_labels": terminals, "nonempty_prompt_context": contexts, "candidate_metric_rows": metrics})
PY
}

preflight() {
    [[ -x "$PYTHON_BIN" && -s "$DATA_PATH" && -s "$DATA_SUMMARY" && -s "$STAGE1_CKPT" ]] || { echo "Missing training prerequisite" >&2; return 1; }
    [[ "$(sha256sum "$DATA_PATH" | awk '{print $1}')" == "$EXPECTED_DATA_SHA256" ]] || { echo "Direct dataset hash mismatch" >&2; return 1; }
    [[ -d /home/data/model/qwen-vl-8B && -s "$REPO_ROOT/data/Jinan/3_4/roadnet_3_4.json" && -s "$REPO_ROOT/data/Jinan/3_4/$TRAINING_FLOW" ]] || { echo "Missing Jinan-real online prerequisite" >&2; return 1; }
    [[ "$TRAINING_FLOW" == "$EVALUATION_FLOW" ]] || { echo "Flow protocol must be same-flow only" >&2; return 1; }
    validate_data; discover_idle_gpus
}

write_provenance() {
    local path key git_head git_diff_hash
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    {
        printf 'created_at=%s\ngit_head=%s\ngit_diff_sha256=%s\n' "$(timestamp)" "$git_head" "$git_diff_hash"
        printf 'source_data=%s\nsource_data_sha256=%s\nsource_summary=%s\nsource_summary_sha256=%s\n' "$DATA_PATH" "$(sha256sum "$DATA_PATH" | awk '{print $1}')" "$DATA_SUMMARY" "$(sha256sum "$DATA_SUMMARY" | awk '{print $1}')"
        printf 'stage1_checkpoint=%s\nstage1_checkpoint_sha256=%s\ntraining_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_only\nonline_eval_gpu=%s\nonline_eval_seed_execution=serial_single_gpu\n' "$STAGE1_CKPT" "$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')" "$TRAINING_FLOW" "$EVALUATION_FLOW" "$WAIT25_IMPROVEMENT_EVAL_GPU"
        for path in "$ARM_SCRIPT" "$ONLINE_SUITE" "$AGGREGATE_SCRIPT" "$REPO_ROOT/projection_train/train_projection_bc.py" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/projection_train/cost_ranking.py" "$REPO_ROOT/projection_train/grouped_dataset.py" "$REPO_ROOT/data_split.py"; do
            key="${path#"$REPO_ROOT"/}"; key="${key//\//_}"; printf 'file_%s_sha256=%s\n' "$key" "$(sha256sum "$path" | awk '{print $1}')"
        done
    } > "$PROVENANCE"
}

run_arm() {
    local arm="$1" seed="$2" root="$RUN_ROOT/arms/$arm/seed_$seed" pid
    [[ ! -e "$root" ]] || { echo "Refusing to overwrite existing arm: $root" >&2; return 1; }
    printf '[%s] launching arm=%s seed=%s flow=%s\n' "$(timestamp)" "$arm" "$seed" "$TRAINING_FLOW"
    setsid bash "$ARM_SCRIPT" "$RUN_ROOT" "$arm" "$seed" & pid=$!
    while kill -0 "$pid" 2>/dev/null; do sleep 60; maybe_notify_progress "$arm/seed_$seed"; done
    wait "$pid"
}

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 ]]; then
        printf 'completed_at=%s\nstatus=success\nrun_root=%s\ntraining_flow=%s\nevaluation_flow=%s\n' "$(timestamp)" "$RUN_ROOT" "$TRAINING_FLOW" "$EVALUATION_FLOW" > "$CAMPAIGN_STATUS"
        notify "jinan_real wait25 refinement campaign completed: 9 Stage-2 models and 27 same-flow online results. Aggregate: $RUN_ROOT/AGGREGATE.md"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\nrun_root=%s\ntraining_flow=%s\nevaluation_flow=%s\n' "$(timestamp)" "$code" "$RUN_ROOT" "$TRAINING_FLOW" "$EVALUATION_FLOW" > "$CAMPAIGN_STATUS"
        notify "jinan_real wait25 refinement campaign stopped with exit_code=${code}; inspect $CAMPAIGN_LOG"
    fi
    trap - EXIT; exit "$code"
}
trap 'exit 143' HUP INT TERM
trap on_exit EXIT

[[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"; exec >> "$CAMPAIGN_LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\ntraining_flow=%s\nevaluation_flow=%s\n' "$(timestamp)" "$RUN_ROOT" "$TRAINING_FLOW" "$EVALUATION_FLOW" > "$CAMPAIGN_STATUS"
LAST_NOTIFICATION_EPOCH="$(date +%s)"; preflight; write_provenance
notify "jinan_real wait25 refinement campaign launched: arms=${ARMS[*]}, train_seeds=2024/42/123, flow=${TRAINING_FLOW}, training_gpus=${WAIT25_IMPROVEMENT_GPUS}, online_gpu=${WAIT25_IMPROVEMENT_EVAL_GPU}."
for arm in "${ARMS[@]}"; do for seed in "${TRAINING_SEEDS[@]}"; do run_arm "$arm" "$seed"; done; done
timeout --foreground 120s "$PYTHON_BIN" "$AGGREGATE_SCRIPT" --run-root "$RUN_ROOT"
