#!/usr/bin/env bash
# Run the Jinan-2000 direct-urgency evidence suite with a three-flow Stage-1 initializer.
set -Eeuo pipefail
shopt -s nullglob

REPO_ROOT=/home/code/LLMSTSC
RUN_ROOT="${1:-$REPO_ROOT/records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${JINAN2000_PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN_BIN="${JINAN2000_TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${JINAN2000_MODEL_PATH:-/home/data/model/qwen-vl-8B}"
EXTERNAL_STAGE1_CKPT="${JINAN2000_STAGE1_CKPT:-}"
FEISHU_SCRIPT="$REPO_ROOT/scripts/feishu_push.py"
BUILDER="$REPO_ROOT/scripts/build_direct_urgency_stage2_dataset.py"
SELECT_SCRIPT="$REPO_ROOT/scripts/select_projection_checkpoint.py"
FLOW=anon_3_4_jinan_real_2000.json
FLOW_STEM="${FLOW%.json}"
RAW_STAGE2_DATA="$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
DIRECT_DATA="$RUN_ROOT/data/train_1440_jinan2000_direct_phase_urgency_v1.jsonl"
DIRECT_SUMMARY="$RUN_ROOT/data/train_1440_jinan2000_direct_phase_urgency_v1.summary.json"
STAGE1_DIR="$RUN_ROOT/stage1_three_flow"
CAMPAIGN_LOG="$RUN_ROOT/campaign.log"
CAMPAIGN_STATUS="$RUN_ROOT/campaign.status"
PROVENANCE="$RUN_ROOT/provenance.env"
STARTUP_MONITOR="$RUN_ROOT/startup_monitor.log"
QUEUE_DIR="$RUN_ROOT/queues"

STAGE1_DATA=(
    "$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
    "$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
    "$REPO_ROOT/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl"
)

DEFAULT_STAGE2_ARMS=(
    hard_bc
    equal_mixed_rank
    wait2_mixed_rank
    wait25_mixed_rank
    wait225_mixed_rank
    wait25_conservative_rank
    wait25_strong_rank
)
DEFAULT_SELECTOR_ARMS=(wait2_mixed_rank wait225_mixed_rank)
DEFAULT_TRAINING_SEEDS=(2024 42 123)
DEFAULT_EVAL_SEEDS=(2024 42 123)

STAGE1_HIDDEN_DIM="${JINAN2000_STAGE1_HIDDEN_DIM:-64}"
STAGE1_PREFIX_TOKENS="${JINAN2000_STAGE1_PREFIX_TOKENS:-8}"
STAGE1_BATCH_SIZE="${JINAN2000_STAGE1_BATCH_SIZE:-4}"
STAGE1_GRAD_ACCUM="${JINAN2000_STAGE1_GRAD_ACCUM:-2}"
STAGE1_TIMEOUT="${JINAN2000_STAGE1_TIMEOUT:-86400s}"

HIDDEN_DIM="${JINAN2000_HIDDEN_DIM:-64}"
PREFIX_TOKENS="${JINAN2000_PREFIX_TOKENS:-8}"
PER_RANK_BATCH="${JINAN2000_PER_RANK_BATCH:-4}"
STAGE2_GLOBAL_EFFECTIVE_BATCH="${JINAN2000_STAGE2_GLOBAL_EFFECTIVE_BATCH:-96}"
STAGE2_GRAD_ACCUM="${JINAN2000_STAGE2_GRAD_ACCUM:-$((STAGE2_GLOBAL_EFFECTIVE_BATCH / PER_RANK_BATCH))}"
STAGE2_TIMEOUT="${JINAN2000_STAGE2_TIMEOUT:-43200s}"
SHORT_RUN_COUNTS="${JINAN2000_SHORT_RUN_COUNTS:-1200}"
FINAL_RUN_COUNTS="${JINAN2000_FINAL_RUN_COUNTS:-3600}"
SHORT_EVAL_TIMEOUT="${JINAN2000_SHORT_EVAL_TIMEOUT:-3600s}"
FINAL_EVAL_TIMEOUT="${JINAN2000_FINAL_EVAL_TIMEOUT:-7200s}"
RUN_SELECTOR_SEED42="${JINAN2000_RUN_SELECTOR_SEED42:-1}"
TAIL_P95_WEIGHT="${JINAN2000_TAILRISK_P95_WEIGHT:-0.10}"
TAIL_MAX_WEIGHT="${JINAN2000_TAILRISK_MAX_WEIGHT:-0.02}"
REWARD_TOP_K="${JINAN2000_CONSTRAINED_REWARD_TOP_K:-3}"
TAIL_COST_P95_WEIGHT="${JINAN2000_CONSTRAINED_TAIL_COST_P95_WEIGHT:-1.0}"
TAIL_COST_MAX_WEIGHT="${JINAN2000_CONSTRAINED_TAIL_COST_MAX_WEIGHT:-0.05}"
IDLE_MEMORY_THRESHOLD_MB="${JINAN2000_IDLE_MEMORY_THRESHOLD_MB:-500}"
LAST_NOTIFICATION_EPOCH=0

STAGE2_ARMS=()
SELECTOR_ARMS=()
TRAINING_SEEDS=()
EVAL_SEEDS=()
ALL_GPUS=()
TARGET_ARGS=()

timestamp() { date -Is; }

parse_words() {
    local env_name="$1"
    local -n output="$2"
    shift 2
    if [[ -n "${!env_name:-}" ]]; then
        read -r -a output <<< "${!env_name}"
    else
        output=("$@")
    fi
}

join_by_comma() {
    local IFS=,
    printf '%s' "$*"
}

notify() {
    [[ -f "$FEISHU_SCRIPT" && -x "$PYTHON_BIN" ]] || return 0
    timeout --foreground 15s "$PYTHON_BIN" "$FEISHU_SCRIPT" "$1" || true
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

file_sha() {
    local path="$1"
    if [[ -s "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'missing'
    fi
}

log_has_errors() {
    local path="$1"
    [[ -f "$path" ]] || return 1
    timeout --foreground 10s rg -q 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|Killed|killed|segmentation fault' "$path"
}

completed_stage2_cells() {
    local count=0 path
    for path in "$RUN_ROOT"/arms/*/seed_*/status.txt; do
        [[ -f "$path" ]] || continue
        if awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$path"; then
            count=$((count + 1))
        fi
    done
    printf '%s' "$count"
}

completed_online_rows() {
    find "$RUN_ROOT" -path "*/evaluation/$FLOW_STEM/seed_*/result.json" -type f -print 2>/dev/null | wc -l | tr -dc '0-9'
}

latest_metric() {
    local result
    result="$(find "$RUN_ROOT" -path "*/evaluation/$FLOW_STEM/seed_2024/result.json" -type f -print 2>/dev/null | sort | tail -n 1 || true)"
    [[ -s "$result" ]] || { printf 'latest_result=not_available'; return 0; }
    "$PYTHON_BIN" - "$result" <<'PY' 2>/dev/null || printf 'latest_result=unreadable'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
r = payload["results"]
print("latest seed2024 reward={:.3f} AQL={:.3f} AWT={:.3f} ATT={:.3f} P95={:.3f} Max={:.3f}".format(
    float(r.get("test_reward_over")),
    float(r.get("test_avg_queue_len_over")),
    float(r.get("test_avg_waiting_time_over")),
    float(r.get("test_avg_travel_time_over")),
    float(r.get("test_vehicle_wait_p95_over")),
    float(r.get("test_vehicle_wait_max_over")),
))
PY
}

maybe_notify_progress() {
    local label="$1" now
    now="$(date +%s)"
    (( now - LAST_NOTIFICATION_EPOCH >= 2400 )) || return 0
    LAST_NOTIFICATION_EPOCH="$now"
    notify "Jinan-2000 direct-urgency progress: phase=${label}; stage2=$(completed_stage2_cells); online_rows=$(completed_online_rows); $(latest_metric); run_root=${RUN_ROOT}"
}

kill_process_group() {
    local pid="$1"
    kill -- -"$pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- -"$pid" 2>/dev/null || true
}

snapshot_process() {
    local label="$1" pid="$2" log_file="$3" gpu_desc="$4"
    {
        printf '\n[%s] %s pid=%s gpu=%s\n' "$(timestamp)" "$label" "$pid" "$gpu_desc"
        timeout --foreground 10s ps -eo pid,ppid,pgid,sid,stat,etime,cmd | awk -v wanted="$pid" 'NR==1 || $1==wanted || $2==wanted || $3==wanted' | head -n 140 || true
        [[ -f "$log_file" ]] && timeout --foreground 10s tail -n 100 "$log_file" || true
        [[ -f "$log_file" ]] && timeout --foreground 10s rg -n 'Traceback|RuntimeError|ERROR|Exception|NaN|Inf|out of memory|Killed|killed|segmentation fault' "$log_file" | tail -n 60 || true
        timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    } >> "$STARTUP_MONITOR" 2>&1
}

run_logged_process() {
    local label="$1" log_file="$2" gpu_desc="$3" pid minute rc
    shift 3
    mkdir -p "$(dirname "$log_file")"
    log "START ${label} gpu=${gpu_desc}"
    setsid "$@" > "$log_file" 2>&1 &
    pid="$!"
    snapshot_process "${label}_startup_t0" "$pid" "$log_file" "$gpu_desc"
    if log_has_errors "$log_file"; then
        kill_process_group "$pid"
        wait "$pid" || true
        log "FAILED ${label} due to startup log error"
        return 1
    fi
    for minute in 1 2 3 4 5; do
        sleep 60
        snapshot_process "${label}_startup_t${minute}m" "$pid" "$log_file" "$gpu_desc"
        if ! kill -0 "$pid" 2>/dev/null; then
            set +e
            wait "$pid"
            rc="$?"
            set -e
            if [[ "$rc" -eq 0 ]]; then
                log "DONE ${label}"
            else
                log "FAILED ${label} exit=${rc}"
            fi
            return "$rc"
        fi
        if log_has_errors "$log_file"; then
            kill_process_group "$pid"
            wait "$pid" || true
            log "FAILED ${label} due to startup log error"
            return 1
        fi
    done
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        maybe_notify_progress "$label"
        if log_has_errors "$log_file"; then
            kill_process_group "$pid"
            wait "$pid" || true
            log "FAILED ${label} due to log error"
            return 1
        fi
    done
    set +e
    wait "$pid"
    rc="$?"
    set -e
    if [[ "$rc" -eq 0 ]]; then
        log "DONE ${label}"
    else
        log "FAILED ${label} exit=${rc}"
    fi
    return "$rc"
}

discover_gpus() {
    local raw gpu used trimmed seen_key
    ALL_GPUS=()
    if [[ -n "${JINAN2000_ALL_GPUS:-}" ]]; then
        IFS=',' read -r -a ALL_GPUS <<< "$JINAN2000_ALL_GPUS"
    else
        while IFS=, read -r gpu used; do
            gpu="${gpu//[[:space:]]/}"
            used="${used//[[:space:]]/}"
            [[ "$gpu" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
            (( used < IDLE_MEMORY_THRESHOLD_MB )) && ALL_GPUS+=("$gpu")
        done < <(timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    fi
    ((${#ALL_GPUS[@]} > 0)) || { echo "No usable GPU found" >&2; return 1; }
    declare -A seen=()
    for raw in "${ALL_GPUS[@]}"; do
        trimmed="${raw//[[:space:]]/}"
        [[ "$trimmed" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $trimmed" >&2; return 1; }
        seen_key="$trimmed"
        [[ -z "${seen[$seen_key]:-}" ]] || { echo "Duplicate GPU id: $trimmed" >&2; return 1; }
        seen["$seen_key"]=1
    done
}

target_args_for_arm() {
    local arm="$1"
    case "$arm" in
        hard_bc)
            TARGET_ARGS=(--target_mode hard --ranking_loss_weight 0 --base_policy_kl_weight 0)
            ;;
        equal_mixed_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 1 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        wait2_mixed_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        wait25_mixed_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.5 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        wait225_mixed_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.30 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.25 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        wait25_conservative_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.15 --soft_label_temperature 0.35 --ranking_loss_weight 0.02 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.5 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        wait25_strong_rank)
            TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.45 --soft_label_temperature 0.35 --ranking_loss_weight 0.08 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_att_weight 0 --cost_aql_weight 1 --cost_awt_weight 2.5 --cost_spillback_weight 0 --cost_tail_wait_weight 0)
            ;;
        *)
            echo "Unknown arm: $arm" >&2
            return 2
            ;;
    esac
}

preflight() {
    local path arm selector_arm found
    [[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; return 1; }
    [[ -x "$PYTHON_BIN" && -x "$TORCHRUN_BIN" ]] || { echo "Missing Python or torchrun" >&2; return 1; }
    [[ -d "$MODEL_PATH" && -s "$BUILDER" && -s "$SELECT_SCRIPT" ]] || { echo "Missing model or helper script" >&2; return 1; }
    [[ -s "$RAW_STAGE2_DATA" && -s "$REPO_ROOT/data/Jinan/3_4/$FLOW" ]] || { echo "Missing Jinan-2000 data" >&2; return 1; }
    [[ -z "$EXTERNAL_STAGE1_CKPT" || -s "$EXTERNAL_STAGE1_CKPT" ]] || { echo "Missing external Stage-1 checkpoint: $EXTERNAL_STAGE1_CKPT" >&2; return 1; }
    command -v flock >/dev/null || { echo "Missing flock" >&2; return 1; }
    command -v rg >/dev/null || { echo "Missing rg" >&2; return 1; }
    command -v nvidia-smi >/dev/null || { echo "Missing nvidia-smi" >&2; return 1; }
    for path in "${STAGE1_DATA[@]}"; do
        [[ -s "$path" ]] || { echo "Missing Stage-1 data: $path" >&2; return 1; }
    done
    for arm in "${STAGE2_ARMS[@]}"; do
        target_args_for_arm "$arm" >/dev/null
    done
    for selector_arm in "${SELECTOR_ARMS[@]}"; do
        found=0
        for arm in "${STAGE2_ARMS[@]}"; do
            [[ "$selector_arm" == "$arm" ]] && found=1
        done
        (( found )) || { echo "Selector arm is not in Stage-2 arms: $selector_arm" >&2; return 1; }
    done
    (( STAGE2_GLOBAL_EFFECTIVE_BATCH % PER_RANK_BATCH == 0 )) || { echo "Invalid Stage-2 effective batch" >&2; return 1; }
    discover_gpus
}

write_provenance() {
    local git_head git_diff_hash path key
    git_head="$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    git_diff_hash="$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    {
        printf 'schema=jinan2000_stage2_direct_urgency_all_gpu_campaign_v1\n'
        printf 'created_at=%s\nrun_root=%s\n' "$(timestamp)" "$RUN_ROOT"
        printf 'git_head=%s\ngit_diff_sha256=%s\n' "$git_head" "$git_diff_hash"
        printf 'training_flow=%s\nevaluation_flow=%s\nflow_protocol=same_flow_jinan2000\n' "$FLOW" "$FLOW"
        printf 'stage1_data_mode=three_flow\n'
        printf 'stage1_data=%s\n' "${STAGE1_DATA[*]}"
        printf 'stage1_data_sha256_bundle=%s\n' "$(for path in "${STAGE1_DATA[@]}"; do sha256sum "$path"; done | sha256sum | awk '{print $1}')"
        if [[ -n "$EXTERNAL_STAGE1_CKPT" ]]; then
            printf 'external_stage1_checkpoint=%s\nexternal_stage1_checkpoint_sha256=%s\n' "$EXTERNAL_STAGE1_CKPT" "$(file_sha "$EXTERNAL_STAGE1_CKPT")"
        fi
        printf 'raw_stage2_data=%s\nraw_stage2_data_sha256=%s\n' "$RAW_STAGE2_DATA" "$(file_sha "$RAW_STAGE2_DATA")"
        printf 'direct_stage2_data=%s\ndirect_stage2_summary=%s\n' "$DIRECT_DATA" "$DIRECT_SUMMARY"
        printf 'stage1_checkpoint=%s\n' "$STAGE1_DIR/best.pt"
        printf 'stage2_arms=%s\nselector_arms=%s\ntraining_seeds=%s\neval_seeds=%s\n' "${STAGE2_ARMS[*]}" "${SELECTOR_ARMS[*]}" "${TRAINING_SEEDS[*]}" "${EVAL_SEEDS[*]}"
        printf 'gpus=%s\nstage2_execution=one_training_worker_per_gpu\nshort_eval_execution=one_eval_worker_per_gpu\nfinal_eval_execution=one_eval_worker_per_gpu\n' "$(join_by_comma "${ALL_GPUS[@]}")"
        printf 'short_run_counts=%s\nfinal_run_counts=%s\nrun_selector_seed42=%s\n' "$SHORT_RUN_COUNTS" "$FINAL_RUN_COUNTS" "$RUN_SELECTOR_SEED42"
        for path in "$BUILDER" "$SELECT_SCRIPT" "$REPO_ROOT/train_align.py" "$REPO_ROOT/projection_train/train_projection_bc.py" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/projection_train/cost_ranking.py" "$REPO_ROOT/projection_train/grouped_dataset.py" "$REPO_ROOT/data_split.py" "$0"; do
            [[ -e "$path" ]] || continue
            key="${path#"$REPO_ROOT"/}"
            key="${key//\//_}"
            printf 'file_%s_sha256=%s\n' "$key" "$(file_sha "$path")"
        done
    } > "$PROVENANCE"
}

build_direct_data() {
    mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/logs"
    log "BUILD direct urgency data for ${FLOW}"
    timeout --foreground 600s "$PYTHON_BIN" "$BUILDER" \
        --input "$RAW_STAGE2_DATA" \
        --output "$DIRECT_DATA" \
        --summary "$DIRECT_SUMMARY" > "$RUN_ROOT/logs/build_direct_data.log" 2>&1
    timeout --foreground 60s "$PYTHON_BIN" - "$DIRECT_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["rows"] == 22860, summary
assert summary["terminal_labels"] == 1440, summary
assert summary["full_candidate_metric_rows"] == 1440, summary
print({"rows": summary["rows"], "terminal_labels": summary["terminal_labels"]})
PY
    printf 'direct_stage2_data_sha256=%s\n' "$(file_sha "$DIRECT_DATA")" >> "$PROVENANCE"
    printf 'direct_stage2_summary_sha256=%s\n' "$(file_sha "$DIRECT_SUMMARY")" >> "$PROVENANCE"
}

run_stage1() {
    local gpu_list gpu_count log_file
    gpu_list="$(join_by_comma "${ALL_GPUS[@]}")"
    gpu_count="${#ALL_GPUS[@]}"
    log_file="$STAGE1_DIR/logs/train.log"
    mkdir -p "$STAGE1_DIR/logs"
    if [[ -n "$EXTERNAL_STAGE1_CKPT" ]]; then
        log "REUSE external Stage-1 checkpoint ${EXTERNAL_STAGE1_CKPT}"
        cp -p "$EXTERNAL_STAGE1_CKPT" "$STAGE1_DIR/best.pt"
        printf 'completed_at=%s\nstatus=success\nbest_checkpoint=%s\nbest_checkpoint_sha256=%s\nstage1_data_mode=three_flow\nexternal_stage1_checkpoint=%s\nexternal_stage1_checkpoint_sha256=%s\ngpus=%s\n' \
            "$(timestamp)" "$STAGE1_DIR/best.pt" "$(file_sha "$STAGE1_DIR/best.pt")" "$EXTERNAL_STAGE1_CKPT" "$(file_sha "$EXTERNAL_STAGE1_CKPT")" "$gpu_list" > "$STAGE1_DIR/status.env"
        printf 'stage1_checkpoint_sha256=%s\n' "$(file_sha "$STAGE1_DIR/best.pt")" >> "$PROVENANCE"
        return 0
    fi
    printf 'started_at=%s\nstatus=running\nstage1_data_mode=three_flow\ngpus=%s\n' "$(timestamp)" "$gpu_list" > "$STAGE1_DIR/status.env"
    run_logged_process "stage1_three_flow" "$log_file" "$gpu_list" \
        env CUDA_VISIBLE_DEVICES="$gpu_list" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        timeout --foreground "$STAGE1_TIMEOUT" "$TORCHRUN_BIN" --standalone --nproc_per_node="$gpu_count" "$REPO_ROOT/train_align.py" \
            --data_path "${STAGE1_DATA[@]}" \
            --model_path "$MODEL_PATH" \
            --output_dir "$STAGE1_DIR" \
            --decision_only \
            --seq_len 16 \
            --image_size 224 \
            --periodicity 16 \
            --input_dim 8 \
            --hidden_dim "$STAGE1_HIDDEN_DIM" \
            --prefix_tokens "$STAGE1_PREFIX_TOKENS" \
            --prefix_mode prefix \
            --temperature 0.05 \
            --batch_size "$STAGE1_BATCH_SIZE" \
            --grad_accum_steps "$STAGE1_GRAD_ACCUM" \
            --lr 1e-4 \
            --min_lr 1e-5 \
            --warmup_ratio 0.1 \
            --lr_schedule cosine \
            --epochs 15 \
            --num_workers 4 \
            --val_ratio 0.1 \
            --eval_max_samples 1000 \
            --log_every_steps 20 \
            --save_every_epochs 5 \
            --seed 2024 \
            --bf16
    [[ -s "$STAGE1_DIR/best.pt" ]] || { echo "Stage 1 completed without best.pt" >&2; return 1; }
    printf 'completed_at=%s\nstatus=success\nbest_checkpoint=%s\nbest_checkpoint_sha256=%s\nstage1_data_mode=three_flow\ngpus=%s\n' \
        "$(timestamp)" "$STAGE1_DIR/best.pt" "$(file_sha "$STAGE1_DIR/best.pt")" "$gpu_list" > "$STAGE1_DIR/status.env"
    printf 'stage1_checkpoint_sha256=%s\n' "$(file_sha "$STAGE1_DIR/best.pt")" >> "$PROVENANCE"
}

run_stage2_cell() {
    local arm="$1" seed="$2" gpu="$3" stage_root ckpt_dir log_dir status_path label
    stage_root="$RUN_ROOT/arms/$arm/seed_$seed"
    ckpt_dir="$stage_root/checkpoints"
    log_dir="$stage_root/logs"
    status_path="$stage_root/status.txt"
    label="stage2_${arm}_seed_${seed}"
    if [[ -f "$status_path" ]] && awk -F= '$1=="status" && $2=="success" { found=1 } END { exit !found }' "$status_path"; then
        log "SKIP ${label}"
        return 0
    fi
    [[ ! -e "$stage_root" ]] || { echo "Existing incomplete Stage-2 cell: $stage_root" >&2; return 1; }
    mkdir -p "$ckpt_dir" "$log_dir"
    target_args_for_arm "$arm"
    printf 'started_at=%s\nstatus=running\narm=%s\ntraining_seed=%s\ntraining_flow=%s\ndata=%s\nalign_ckpt=%s\ngpu=%s\n' \
        "$(timestamp)" "$arm" "$seed" "$FLOW" "$DIRECT_DATA" "$STAGE1_DIR/best.pt" "$gpu" > "$status_path"
    run_logged_process "$label" "$log_dir/train.log" "$gpu" \
        env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        timeout --foreground "$STAGE2_TIMEOUT" "$TORCHRUN_BIN" --standalone --nproc_per_node=1 "$REPO_ROOT/projection_train/train_projection_bc.py" \
            --data_path "$DIRECT_DATA" \
            --llm_path "$MODEL_PATH" \
            --align_ckpt "$STAGE1_DIR/best.pt" \
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
            --grad_accum_steps "$STAGE2_GRAD_ACCUM" \
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
            "${TARGET_ARGS[@]}"
    [[ -s "$ckpt_dir/best.pt" ]] || { echo "Stage 2 completed without best.pt: $stage_root" >&2; return 1; }
    printf 'completed_at=%s\nstatus=success\narm=%s\ntraining_seed=%s\ntraining_flow=%s\ndata=%s\ndata_sha256=%s\nalign_ckpt=%s\nalign_ckpt_sha256=%s\nstage2_best=%s\nstage2_best_sha256=%s\ngpu=%s\n' \
        "$(timestamp)" "$arm" "$seed" "$FLOW" "$DIRECT_DATA" "$(file_sha "$DIRECT_DATA")" "$STAGE1_DIR/best.pt" "$(file_sha "$STAGE1_DIR/best.pt")" "$ckpt_dir/best.pt" "$(file_sha "$ckpt_dir/best.pt")" "$gpu" > "$status_path"
}

next_job() {
    local queue="$1" lock="$2" line
    {
        flock 9
        line="$(sed -n '1p' "$queue" || true)"
        if [[ -n "$line" ]]; then
            sed -i '1d' "$queue"
        fi
        printf '%s' "$line"
    } 9>"$lock"
}

stage2_worker() {
    local gpu="$1" queue="$2" lock="$3" line arm seed
    while true; do
        line="$(next_job "$queue" "$lock")"
        [[ -n "$line" ]] || break
        IFS=$'\t' read -r arm seed <<< "$line"
        run_stage2_cell "$arm" "$seed" "$gpu"
    done
}

run_stage2_queue() {
    local queue="$QUEUE_DIR/stage2_jobs.tsv" lock="$QUEUE_DIR/stage2_jobs.lock" gpu pid rc fail=0
    mkdir -p "$QUEUE_DIR"
    : > "$queue"
    for arm in "${STAGE2_ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            printf '%s\t%s\n' "$arm" "$seed" >> "$queue"
        done
    done
    log "STAGE2 queue started on GPUs $(join_by_comma "${ALL_GPUS[@]}")"
    pids=()
    for gpu in "${ALL_GPUS[@]}"; do
        stage2_worker "$gpu" "$queue" "$lock" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done
    (( fail == 0 )) || return 1
}

run_short_eval_one() {
    local arm="$1" seed="$2" selector_seed="$3" ckpt="$4" gpu="$5"
    local stage_root artifact_dir log_dir stem result log_file label
    stage_root="$RUN_ROOT/arms/$arm/seed_$seed"
    artifact_dir="$stage_root/artifacts/selector_seed_${selector_seed}"
    log_dir="$stage_root/logs/short_eval/selector_seed_${selector_seed}"
    stem="$(basename "$ckpt" .pt)"
    result="$artifact_dir/short_eval_${stem}.json"
    log_file="$log_dir/short_eval_${stem}.log"
    label="short_${arm}_train_${seed}_selector_${selector_seed}_${stem}"
    [[ -s "$result" ]] && return 0
    mkdir -p "$artifact_dir" "$log_dir"
    run_logged_process "$label" "$log_file" "$gpu" \
        env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
        timeout --foreground "$SHORT_EVAL_TIMEOUT" "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
            --memo "jinan2000_short_${arm}_train${seed}_selector${selector_seed}_${stem}" \
            --projection_ckpt "$ckpt" \
            --llm_path "$MODEL_PATH" \
            --dataset jinan \
            --traffic_file "$FLOW" \
            --run_counts "$SHORT_RUN_COUNTS" \
            --num_rounds 1 \
            --seed "$selector_seed" \
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
            --result_json "$result"
    [[ -s "$result" ]] || { echo "Missing short eval result: $result" >&2; return 1; }
}

short_eval_worker() {
    local gpu="$1" queue="$2" lock="$3" line arm seed selector_seed ckpt
    while true; do
        line="$(next_job "$queue" "$lock")"
        [[ -n "$line" ]] || break
        IFS=$'\t' read -r arm seed selector_seed ckpt <<< "$line"
        run_short_eval_one "$arm" "$seed" "$selector_seed" "$ckpt" "$gpu"
    done
}

run_short_eval_queue() {
    local queue="$QUEUE_DIR/short_eval_jobs.tsv" lock="$QUEUE_DIR/short_eval_jobs.lock" arm seed selector_seed ckpt gpu pid fail=0
    mkdir -p "$QUEUE_DIR"
    : > "$queue"
    selector_seeds=(2024)
    if [[ "$RUN_SELECTOR_SEED42" == "1" ]]; then
        selector_seeds+=(42)
    fi
    for arm in "${SELECTOR_ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            for selector_seed in "${selector_seeds[@]}"; do
                for ckpt in "$RUN_ROOT/arms/$arm/seed_$seed/checkpoints"/epoch_*.pt "$RUN_ROOT/arms/$arm/seed_$seed/checkpoints"/best.pt "$RUN_ROOT/arms/$arm/seed_$seed/checkpoints"/last.pt; do
                    [[ -f "$ckpt" ]] || continue
                    printf '%s\t%s\t%s\t%s\n' "$arm" "$seed" "$selector_seed" "$ckpt" >> "$queue"
                done
            done
        done
    done
    if [[ ! -s "$queue" ]]; then
        log "SHORT_EVAL queue empty"
        return 0
    fi
    log "SHORT_EVAL queue started on GPUs $(join_by_comma "${ALL_GPUS[@]}")"
    pids=()
    for gpu in "${ALL_GPUS[@]}"; do
        short_eval_worker "$gpu" "$queue" "$lock" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done
    (( fail == 0 )) || return 1
}

selected_field() {
    local json_path="$1" field="$2"
    "$PYTHON_BIN" - "$json_path" "$field" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
selected = data.get("selected") or {}
value = selected.get(sys.argv[2], "")
print(value)
PY
}

prepare_selector_checkpoint() {
    local arm="$1" seed="$2" selector="$3" selector_seed="$4"
    local stage_root target_dir ckpt_dir artifact_dir selection_json selected_ckpt selected_name copied_ckpt log_dir eval_glob
    stage_root="$RUN_ROOT/arms/$arm/seed_$seed"
    target_dir="$stage_root/selectors/$selector"
    ckpt_dir="$target_dir/checkpoints"
    artifact_dir="$target_dir/artifacts"
    log_dir="$target_dir/logs"
    mkdir -p "$ckpt_dir" "$artifact_dir" "$log_dir"
    copied_ckpt="$ckpt_dir/selected.${selector}.pt"
    if [[ "$selector" == "offline_val_acc" ]]; then
        selected_ckpt="$stage_root/checkpoints/best.pt"
        selected_name="best"
        [[ -s "$selected_ckpt" ]] || { echo "Missing offline-val_acc checkpoint: $selected_ckpt" >&2; return 1; }
        cp -p "$selected_ckpt" "$copied_ckpt"
        printf 'selector=%s\nselector_seed=offline\nselected_name=%s\nselected_source_checkpoint=%s\nselected_source_sha256=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
            "$selector" "$selected_name" "$selected_ckpt" "$(file_sha "$selected_ckpt")" "$copied_ckpt" "$(file_sha "$copied_ckpt")" > "$target_dir/selected_checkpoint.env"
        return 0
    fi

    case "$selector" in
        short_online)
            selection_json="$artifact_dir/short_online_checkpoint_selection.json"
            timeout --foreground 300s "$PYTHON_BIN" "$SELECT_SCRIPT" \
                --stage_root "$stage_root" \
                --ranking_metric reward \
                --eval_glob "artifacts/selector_seed_${selector_seed}/short_eval_*.json" \
                --fallback_to_unconstrained \
                --dry_run \
                --output_json "$selection_json" > "$log_dir/select_short_online.log" 2>&1
            ;;
        tail_risk)
            selection_json="$artifact_dir/tail_risk_checkpoint_selection.json"
            timeout --foreground 300s "$PYTHON_BIN" "$SELECT_SCRIPT" \
                --stage_root "$stage_root" \
                --ranking_metric tail_risk \
                --tail_p95_weight "$TAIL_P95_WEIGHT" \
                --tail_max_weight "$TAIL_MAX_WEIGHT" \
                --eval_glob "artifacts/selector_seed_${selector_seed}/short_eval_*.json" \
                --fallback_to_unconstrained \
                --dry_run \
                --output_json "$selection_json" > "$log_dir/select_tail_risk.log" 2>&1
            ;;
        constrained)
            selection_json="$artifact_dir/constrained_checkpoint_selection.json"
            timeout --foreground 300s "$PYTHON_BIN" "$SELECT_SCRIPT" \
                --stage_root "$stage_root" \
                --ranking_metric reward_topk_tail \
                --reward_top_k "$REWARD_TOP_K" \
                --tail_cost_p95_weight "$TAIL_COST_P95_WEIGHT" \
                --tail_cost_max_weight "$TAIL_COST_MAX_WEIGHT" \
                --eval_glob "artifacts/selector_seed_${selector_seed}/short_eval_epoch_*.json" \
                --fallback_to_unconstrained \
                --dry_run \
                --output_json "$selection_json" > "$log_dir/select_constrained.log" 2>&1
            ;;
        short_online_seed42)
            selection_json="$artifact_dir/short_online_seed42_checkpoint_selection.json"
            timeout --foreground 300s "$PYTHON_BIN" "$SELECT_SCRIPT" \
                --stage_root "$stage_root" \
                --ranking_metric reward \
                --eval_glob "artifacts/selector_seed_${selector_seed}/short_eval_*.json" \
                --fallback_to_unconstrained \
                --dry_run \
                --output_json "$selection_json" > "$log_dir/select_short_online_seed42.log" 2>&1
            ;;
        *)
            echo "Unknown selector: $selector" >&2
            return 2
            ;;
    esac
    selected_ckpt="$(selected_field "$selection_json" checkpoint_path)"
    selected_name="$(selected_field "$selection_json" name)"
    [[ -s "$selected_ckpt" ]] || { echo "Selected checkpoint missing: $selected_ckpt" >&2; return 1; }
    cp -p "$selected_ckpt" "$copied_ckpt"
    printf 'selector=%s\nselector_seed=%s\nselection_json=%s\nselected_name=%s\nselected_source_checkpoint=%s\nselected_source_sha256=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
        "$selector" "$selector_seed" "$selection_json" "$selected_name" "$selected_ckpt" "$(file_sha "$selected_ckpt")" "$copied_ckpt" "$(file_sha "$copied_ckpt")" > "$target_dir/selected_checkpoint.env"
}

prepare_selectors() {
    local arm seed
    for arm in "${STAGE2_ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            prepare_selector_checkpoint "$arm" "$seed" offline_val_acc offline
        done
    done
    for arm in "${SELECTOR_ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            prepare_selector_checkpoint "$arm" "$seed" short_online 2024
            prepare_selector_checkpoint "$arm" "$seed" tail_risk 2024
            prepare_selector_checkpoint "$arm" "$seed" constrained 2024
            if [[ "$RUN_SELECTOR_SEED42" == "1" ]]; then
                prepare_selector_checkpoint "$arm" "$seed" short_online_seed42 42
            fi
        done
    done
}

validate_online_result() {
    local result="$1" checkpoint="$2" seed="$3"
    "$PYTHON_BIN" - "$result" "$checkpoint" "$REPO_ROOT/projection_train/run_projection_online.py" "$FLOW" "$seed" "$FINAL_RUN_COUNTS" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

result = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
evaluator = Path(sys.argv[3])
flow = sys.argv[4]
seed = int(sys.argv[5])
run_counts = int(sys.argv[6])

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = json.loads(result.read_text(encoding="utf-8"))
args = payload["args"]
hashes = payload["artifact_hashes"]
det = payload["determinism"]
assert args["traffic_file"] == flow
assert int(args["seed"]) == seed
assert int(args["run_counts"]) == run_counts
assert Path(args["projection_ckpt"]).resolve() == checkpoint.resolve()
assert hashes["projection_ckpt_sha256"] == digest(checkpoint)
assert hashes["online_evaluator_sha256"] == digest(evaluator)
assert args["deterministic"] is True
assert det["requested"] is True
assert det["torch_deterministic_algorithms"] is True
assert det["cudnn_deterministic"] is True
assert det["cudnn_benchmark"] is False
assert det["allow_tf32"] is False
assert all(float(args.get(name, 0.0) or 0.0) == 0.0 for name in (
    "max_phase_override_gap",
    "max_phase_override_left_wait_protect",
    "max_phase_override_target_queue_floor",
    "max_phase_override_total_queue_floor",
))
assert isinstance(payload["results"], dict)
assert all(math.isfinite(float(value)) for value in payload["results"].values())
PY
}

run_final_eval_one() {
    local arm="$1" seed="$2" selector="$3" checkpoint="$4" eval_seed="$5" gpu="$6"
    local target_dir result log_file label
    target_dir="$RUN_ROOT/arms/$arm/seed_$seed/selectors/$selector"
    result="$target_dir/evaluation/$FLOW_STEM/seed_${eval_seed}/result.json"
    log_file="$target_dir/logs/online/$FLOW_STEM/seed_${eval_seed}.log"
    label="final_${selector}_${arm}_train_${seed}_eval_${eval_seed}"
    [[ -s "$result" ]] && validate_online_result "$result" "$checkpoint" "$eval_seed" && return 0
    mkdir -p "$(dirname "$result")" "$(dirname "$log_file")"
    run_logged_process "$label" "$log_file" "$gpu" \
        env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
        timeout --foreground "$FINAL_EVAL_TIMEOUT" "$PYTHON_BIN" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
            --memo "jinan2000_${selector}_${arm}_train${seed}_eval${eval_seed}" \
            --projection_ckpt "$checkpoint" \
            --llm_path "$MODEL_PATH" \
            --dataset jinan \
            --traffic_file "$FLOW" \
            --run_counts "$FINAL_RUN_COUNTS" \
            --num_rounds 1 \
            --seed "$eval_seed" \
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
            --result_json "$result"
    validate_online_result "$result" "$checkpoint" "$eval_seed"
}

final_eval_worker() {
    local gpu="$1" queue="$2" lock="$3" line arm seed selector checkpoint eval_seed
    while true; do
        line="$(next_job "$queue" "$lock")"
        [[ -n "$line" ]] || break
        IFS=$'\t' read -r arm seed selector checkpoint eval_seed <<< "$line"
        run_final_eval_one "$arm" "$seed" "$selector" "$checkpoint" "$eval_seed" "$gpu"
    done
}

run_final_eval_queue() {
    local queue="$QUEUE_DIR/final_eval_jobs.tsv" lock="$QUEUE_DIR/final_eval_jobs.lock" arm seed selector_dir selector checkpoint eval_seed gpu pid fail=0
    mkdir -p "$QUEUE_DIR"
    : > "$queue"
    for arm in "${STAGE2_ARMS[@]}"; do
        for seed in "${TRAINING_SEEDS[@]}"; do
            for selector_dir in "$RUN_ROOT/arms/$arm/seed_$seed/selectors"/*; do
                [[ -d "$selector_dir" ]] || continue
                selector="$(basename "$selector_dir")"
                checkpoint="$selector_dir/checkpoints/selected.${selector}.pt"
                [[ -s "$checkpoint" ]] || continue
                for eval_seed in "${EVAL_SEEDS[@]}"; do
                    printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "$seed" "$selector" "$checkpoint" "$eval_seed" >> "$queue"
                done
            done
        done
    done
    [[ -s "$queue" ]] || { echo "Final evaluation queue is empty" >&2; return 1; }
    log "FINAL_EVAL queue started on GPUs $(join_by_comma "${ALL_GPUS[@]}")"
    pids=()
    for gpu in "${ALL_GPUS[@]}"; do
        final_eval_worker "$gpu" "$queue" "$lock" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done
    (( fail == 0 )) || return 1
}

aggregate_results() {
    timeout --foreground 120s "$PYTHON_BIN" - "$RUN_ROOT" "$FLOW" "${EVAL_SEEDS[*]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
flow = sys.argv[2]
eval_seeds = [int(item) for item in sys.argv[3].split()]
flow_stem = flow.removesuffix(".json")
metrics = [
    ("Reward", "test_reward_over"),
    ("AQL", "test_avg_queue_len_over"),
    ("AWT", "test_avg_waiting_time_over"),
    ("ATT", "test_avg_travel_time_over"),
    ("P95 wait", "test_vehicle_wait_p95_over"),
    ("Max wait", "test_vehicle_wait_max_over"),
]

def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out

rows = []
incomplete = []
for arm_dir in sorted((root / "arms").glob("*")):
    if not arm_dir.is_dir():
        continue
    arm = arm_dir.name
    for seed_dir in sorted(arm_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        training_seed = seed_dir.name.removeprefix("seed_")
        for selector_dir in sorted((seed_dir / "selectors").glob("*")):
            if not selector_dir.is_dir():
                continue
            selector = selector_dir.name
            selected = read_env(selector_dir / "selected_checkpoint.env")
            checkpoint = selected.get("checkpoint", "")
            selected_name = selected.get("selected_name", "")
            selector_seed = selected.get("selector_seed", "")
            for eval_dir in sorted((selector_dir / "evaluation" / flow_stem).glob("seed_*")):
                if not eval_dir.is_dir():
                    continue
                eval_seed = eval_dir.name.removeprefix("seed_")
                result_path = eval_dir / "result.json"
                base = {
                    "arm": arm,
                    "training_seed": training_seed,
                    "selector": selector,
                    "selector_seed": selector_seed,
                    "selected_checkpoint_name": selected_name,
                    "checkpoint": checkpoint,
                    "eval_seed": eval_seed,
                    "flow": flow,
                    "result_path": str(result_path),
                }
                if not result_path.is_file():
                    incomplete.append({**base, "reason": "missing_result"})
                    continue
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    values = {key: float(payload["results"][key]) for _, key in metrics}
                    rows.append({**base, **values})
                except Exception as exc:
                    incomplete.append({**base, "reason": type(exc).__name__})

expected = []
for selector_dir in sorted(root.glob("arms/*/seed_*/selectors/*")):
    if not selector_dir.is_dir():
        continue
    arm = selector_dir.parents[2].name
    training_seed = selector_dir.parents[1].name.removeprefix("seed_")
    selector = selector_dir.name
    selected = read_env(selector_dir / "selected_checkpoint.env")
    for eval_seed in eval_seeds:
        result_path = selector_dir / "evaluation" / flow_stem / f"seed_{eval_seed}" / "result.json"
        if not result_path.is_file():
            incomplete.append({
                "arm": arm,
                "training_seed": training_seed,
                "selector": selector,
                "selector_seed": selected.get("selector_seed", ""),
                "selected_checkpoint_name": selected.get("selected_name", ""),
                "checkpoint": selected.get("checkpoint", ""),
                "eval_seed": str(eval_seed),
                "flow": flow,
                "result_path": str(result_path),
                "reason": "missing_result",
            })

fields = [
    "arm",
    "training_seed",
    "selector",
    "selector_seed",
    "selected_checkpoint_name",
    "checkpoint",
    "eval_seed",
    "flow",
    *[key for _, key in metrics],
    "result_path",
]
with (root / "online_results.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

incomplete_fields = [
    "arm",
    "training_seed",
    "selector",
    "selector_seed",
    "selected_checkpoint_name",
    "checkpoint",
    "eval_seed",
    "flow",
    "result_path",
    "reason",
]
with (root / "incomplete_runs.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=incomplete_fields)
    writer.writeheader()
    writer.writerows(incomplete)

primary = [row for row in rows if str(row["eval_seed"]) == "2024"]
summary: dict[str, dict[str, dict[str, object]]] = {}
for selector in sorted({row["selector"] for row in primary}):
    summary[selector] = {}
    for arm in sorted({row["arm"] for row in primary if row["selector"] == selector}):
        selected_rows = [row for row in primary if row["selector"] == selector and row["arm"] == arm]
        summary[selector][arm] = {}
        for _, key in metrics:
            values = [float(row[key]) for row in selected_rows]
            summary[selector][arm][key] = {
                "n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                "values": values,
            }

output = {
    "schema": "jinan2000_stage2_direct_urgency_all_gpu_aggregate_v1",
    "flow": flow,
    "primary_eval_seed": 2024,
    "completed_rows": len(rows),
    "incomplete_rows": incomplete,
    "primary_by_selector_arm": summary,
}
(root / "aggregate.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def fmt(item: dict[str, object]) -> str:
    mean = item["mean"]
    if mean is None:
        return "NA"
    return f"{float(mean):.3f} +/- {float(item['std']):.3f}"

lines = [
    "# Jinan-2000 Direct-Urgency All-GPU Aggregate",
    "",
    f"Flow: `{flow}`. Primary eval seed: `2024`.",
    "",
    "| Selector | Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for selector in sorted(summary):
    for arm in sorted(summary[selector]):
        stats = summary[selector][arm]
        lines.append(
            f"| `{selector}` | `{arm}` | {stats['test_reward_over']['n']} | "
            f"{fmt(stats['test_reward_over'])} | {fmt(stats['test_avg_queue_len_over'])} | "
            f"{fmt(stats['test_avg_waiting_time_over'])} | {fmt(stats['test_avg_travel_time_over'])} | "
            f"{fmt(stats['test_vehicle_wait_p95_over'])} | {fmt(stats['test_vehicle_wait_max_over'])} |"
        )
lines += ["", f"Completed online rows: {len(rows)}. Incomplete rows: {len(incomplete)}."]
(root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"completed_rows": len(rows), "incomplete_rows": len(incomplete)}))
PY
}

on_error() {
    local code="$1" line="$2"
    printf '[%s] ERROR line=%s exit=%s\n' "$(timestamp)" "$line" "$code" >> "${CAMPAIGN_LOG:-/tmp/jinan2000_direct_urgency.error.log}" 2>/dev/null || true
}

on_exit() {
    local code="$?" outcome
    if [[ "$code" -eq 0 ]]; then
        outcome=success
    else
        outcome=failed
    fi
    printf 'completed_at=%s\nstatus=%s\nexit_code=%s\nrun_root=%s\nstage2_cells=%s\nonline_rows=%s\n' \
        "$(timestamp)" "$outcome" "$code" "$RUN_ROOT" "$(completed_stage2_cells)" "$(completed_online_rows)" > "$CAMPAIGN_STATUS"
    notify "Jinan-2000 direct-urgency all-GPU campaign ${outcome}: exit=${code}; stage2=$(completed_stage2_cells); online_rows=$(completed_online_rows); $(latest_metric); run_root=${RUN_ROOT}"
    trap - EXIT
    exit "$code"
}

trap 'exit 143' HUP INT TERM
trap 'on_error "$?" "$LINENO"' ERR
trap on_exit EXIT

parse_words JINAN2000_STAGE2_ARMS STAGE2_ARMS "${DEFAULT_STAGE2_ARMS[@]}"
parse_words JINAN2000_SELECTOR_ARMS SELECTOR_ARMS "${DEFAULT_SELECTOR_ARMS[@]}"
parse_words JINAN2000_TRAINING_SEEDS TRAINING_SEEDS "${DEFAULT_TRAINING_SEEDS[@]}"
parse_words JINAN2000_EVAL_SEEDS EVAL_SEEDS "${DEFAULT_EVAL_SEEDS[@]}"

preflight
mkdir -p "$RUN_ROOT" "$QUEUE_DIR"
exec >> "$CAMPAIGN_LOG" 2>&1
printf 'started_at=%s\nstatus=running\nrun_root=%s\nflow=%s\n' "$(timestamp)" "$RUN_ROOT" "$FLOW" > "$CAMPAIGN_STATUS"
write_provenance
LAST_NOTIFICATION_EPOCH="$(date +%s)"
notify "Jinan-2000 direct-urgency all-GPU campaign launched: Stage1=three_flow, Stage2 flow=${FLOW}, arms=${STAGE2_ARMS[*]}, selector_arms=${SELECTOR_ARMS[*]}, GPUs=$(join_by_comma "${ALL_GPUS[@]}"), run_root=${RUN_ROOT}"

if [[ "${JINAN2000_STOP_AFTER_PREFLIGHT:-0}" == "1" ]]; then
    log "STOP after preflight"
    exit 0
fi

build_direct_data
run_stage1
run_stage2_queue
run_short_eval_queue
prepare_selectors
run_final_eval_queue
aggregate_results
