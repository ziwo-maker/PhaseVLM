#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_all_sim_expert_datasets.sh [--dry-run] [--run-id RUN_ID] [--datasets "hangzhou"]

Default behavior:
  Runs run_sim_expert_action.py once per traffic file.
  Each completed traffic file immediately writes its own cleaned/backfilled jsonl.

Common launch:
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  export RUN_ID
  nohup bash scripts/run_all_sim_expert_datasets.sh > "nohup_sim_expert_all_${RUN_ID}.log" 2>&1 &

Useful environment overrides:
  CUDA_VISIBLE_DEVICES=0,1
  LLM_MODEL=gemma-3-27B
  LLM_PATH=/home/data/model/gemma-3-27B/
  TENSOR_PARALLEL_SIZE=2
  BACKFILL_BATCH_SIZE=4
  CLEAN_SEQ_LEN=16
  OUTPUT_ROOT=data/recorded/generated_datasets/${RUN_ID}
  LOG_ROOT=records/dataset_generation/${RUN_ID}
EOF
}

DRY_RUN="${DRY_RUN:-0}"
RUN_DATASETS="${RUN_DATASETS:-jinan hangzhou}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --run-id)
            if [[ $# -lt 2 ]]; then
                echo "--run-id requires a value" >&2
                exit 2
            fi
            RUN_ID="$2"
            shift 2
            ;;
        --datasets)
            if [[ $# -lt 2 ]]; then
                echo "--datasets requires a value" >&2
                exit 2
            fi
            RUN_DATASETS="${2//,/ }"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LLM_MODEL="${LLM_MODEL:-gemma-3-27B}"
LLM_PATH="${LLM_PATH:-/home/data/model/gemma-3-27B/}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
BACKFILL_BATCH_SIZE="${BACKFILL_BATCH_SIZE:-4}"
CLEAN_SEQ_LEN="${CLEAN_SEQ_LEN:-16}"
MEMO="${MEMO:-SimExpertDataset_${RUN_ID}}"
PROJ_NAME="${PROJ_NAME:-SimExpertDataset_${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/recorded/generated_datasets/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-records/dataset_generation/${RUN_ID}}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

EVENT_LOG="${LOG_ROOT}/run_events.log"
MANIFEST="${LOG_ROOT}/run_manifest.tsv"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

log_event() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${EVENT_LOG}"
}

declare -A DATASET_TRAFFIC_FILES
# DATASET_TRAFFIC_FILES[jinan]="anon_3_4_jinan_real.json anon_3_4_jinan_real_2000.json anon_3_4_jinan_real_2500.json anon_3_4_jinan_synthetic_24000_60min.json anon_3_4_jinan_synthetic_24h_6000.json"
DATASET_TRAFFIC_FILES[hangzhou]="anon_4_4_hangzhou_real.json anon_4_4_hangzhou_real_5816.json anon_4_4_hangzhou_synthetic_24000_60min.json"

printf 'run_id\tdataset\ttraffic_file\tstart_time\tend_time\telapsed_sec\tstatus\tlog_file\tclean_output\tbackfill_output\n' > "${MANIFEST}"

log_event "run_id=${RUN_ID}"
log_event "repo_root=${REPO_ROOT}"
log_event "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
log_event "llm_model=${LLM_MODEL}"
log_event "llm_path=${LLM_PATH}"
log_event "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
log_event "output_root=${OUTPUT_ROOT}"
log_event "log_root=${LOG_ROOT}"
log_event "dry_run=${DRY_RUN}"

EXIT_CODE=0
TOTAL_START_TS=$(date +%s)

run_dataset() {
    local dataset="$1"

    if [[ -z "${DATASET_TRAFFIC_FILES[$dataset]+defined}" ]]; then
        log_event "skip unknown dataset=${dataset}"
        EXIT_CODE=1
        return 0
    fi

    local traffic_files_text="${DATASET_TRAFFIC_FILES[$dataset]}"
    local -a traffic_files
    read -r -a traffic_files <<< "${traffic_files_text}"

    local all_clean_output="${OUTPUT_ROOT}/align_train.cleaned.${dataset}.all.${RUN_ID}.jsonl"
    local all_backfill_output="${OUTPUT_ROOT}/align_train.cleaned.llm_caption.${dataset}.all.${RUN_ID}.jsonl"

    : > "${all_clean_output}"
    : > "${all_backfill_output}"

    log_event "start dataset=${dataset} traffic_files=${traffic_files_text}"
    log_event "all_clean_output=${all_clean_output}"
    log_event "all_backfill_output=${all_backfill_output}"

    local traffic_file
    for traffic_file in "${traffic_files[@]}"; do
        local traffic_base="${traffic_file##*/}"
        local traffic_stem="${traffic_base%.json}"
        local clean_output="${OUTPUT_ROOT}/align_train.cleaned.${dataset}.${traffic_stem}.${RUN_ID}.jsonl"
        local backfill_output="${OUTPUT_ROOT}/align_train.cleaned.llm_caption.${dataset}.${traffic_stem}.${RUN_ID}.jsonl"
        local log_file="${LOG_ROOT}/nohup_sim_expert.${dataset}.${traffic_stem}.${RUN_ID}.log"

        local start_time
        local end_time
        local start_ts
        local end_ts
        local elapsed_sec
        local status
        local -a cmd

        start_time=$(timestamp)
        start_ts=$(date +%s)

        cmd=(
            python run_sim_expert_action.py
            --dataset "${dataset}"
            --traffic_file "${traffic_file}"
            --llm_model "${LLM_MODEL}"
            --llm_path "${LLM_PATH}"
            --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
            --backfill_batch_size "${BACKFILL_BATCH_SIZE}"
            --clean_seq_len "${CLEAN_SEQ_LEN}"
            --clean_output "${clean_output}"
            --backfill_output "${backfill_output}"
            --memo "${MEMO}_${dataset}_${traffic_stem}"
            --proj_name "${PROJ_NAME}_${dataset}_${traffic_stem}"
        )

        log_event "start dataset=${dataset} traffic_file=${traffic_file}"
        log_event "log_file=${log_file}"
        log_event "clean_output=${clean_output}"
        log_event "backfill_output=${backfill_output}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            {
                printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
                printf '%q ' "${cmd[@]}"
                printf '\n'
            } > "${log_file}"
            status="DRY_RUN"
        else
            env \
                CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
                PYTHONUNBUFFERED=1 \
                PYTHONIOENCODING=UTF-8 \
                LANG="${LANG:-C.UTF-8}" \
                LC_ALL="${LC_ALL:-C.UTF-8}" \
                TERM=dumb \
                NO_COLOR=1 \
                FORCE_COLOR=0 \
                VLLM_LOGGING_COLOR=0 \
                TQDM_DISABLE=1 \
                HF_HUB_DISABLE_PROGRESS_BARS=1 \
                "${cmd[@]}" > "${log_file}" 2>&1
            local exit_status=$?
            if [[ ${exit_status} -eq 0 ]]; then
                status="SUCCESS"
                if [[ -s "${clean_output}" ]]; then
                    cat "${clean_output}" >> "${all_clean_output}"
                fi
                if [[ -s "${backfill_output}" ]]; then
                    cat "${backfill_output}" >> "${all_backfill_output}"
                fi
            else
                status="FAILED_${exit_status}"
                EXIT_CODE=1
            fi
        fi

        end_time=$(timestamp)
        end_ts=$(date +%s)
        elapsed_sec=$((end_ts - start_ts))

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${RUN_ID}" \
            "${dataset}" \
            "${traffic_file}" \
            "${start_time}" \
            "${end_time}" \
            "${elapsed_sec}" \
            "${status}" \
            "${log_file}" \
            "${clean_output}" \
            "${backfill_output}" >> "${MANIFEST}"

        log_event "finish dataset=${dataset} traffic_file=${traffic_file} status=${status} elapsed_sec=${elapsed_sec}"
    done

    log_event "finish dataset=${dataset} all_clean_output=${all_clean_output} all_backfill_output=${all_backfill_output}"
}

read -r -a SELECTED_DATASETS <<< "${RUN_DATASETS}"
for dataset in "${SELECTED_DATASETS[@]}"; do
    run_dataset "${dataset}"
done

TOTAL_END_TS=$(date +%s)
log_event "total_elapsed_sec=$((TOTAL_END_TS - TOTAL_START_TS))"
log_event "manifest=${MANIFEST}"

exit "${EXIT_CODE}"
