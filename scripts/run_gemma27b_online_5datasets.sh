#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}" || exit 1

RUN_ID="${RUN_ID:-gemma27b_online_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/records/online_tests/${RUN_ID}}"
LOG_DIR="${BASE_DIR}/logs"
SUMMARY_TSV="${BASE_DIR}/summary.tsv"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LLM_MODEL="${LLM_MODEL:-gemma-3-27B}"
LLM_PATH="${LLM_PATH:-/home/data/model/gemma-3-27B/}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NEW_MAX_TOKENS="${NEW_MAX_TOKENS:-1024}"
PROJ_NAME="${PROJ_NAME:-Gemma27BOnline_${RUN_ID}}"

mkdir -p "${LOG_DIR}"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

log_event() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

write_summary_header() {
    printf 'run_id\tdataset\ttraffic_file\tstart_time\tend_time\telapsed_sec\tstatus\tlog_file\n' > "${SUMMARY_TSV}"
}

run_one() {
    local dataset="$1"
    local traffic_file="$2"
    local traffic_stem="${traffic_file%.json}"
    local log_file="${LOG_DIR}/${dataset}_${traffic_stem}.log"
    local start_time
    local end_time
    local start_ts
    local end_ts
    local elapsed_sec
    local status

    start_time=$(timestamp)
    start_ts=$(date +%s)

    log_event "start dataset=${dataset} traffic_file=${traffic_file} log=${log_file}"

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
    WANDB_MODE="${WANDB_MODE:-offline}" \
    WANDB_SILENT="${WANDB_SILENT:-true}" \
    VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
    "${PYTHON_BIN}" run_open_LLM_with_vllm.py \
        --memo "online_tests/${RUN_ID}/runs/${dataset}_${traffic_stem}" \
        --llm_model "${LLM_MODEL}" \
        --llm_path "${LLM_PATH}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --new_max_tokens "${NEW_MAX_TOKENS}" \
        --proj_name "${PROJ_NAME}" \
        --dataset "${dataset}" \
        --traffic_file "${traffic_file}" \
        > "${log_file}" 2>&1

    local exit_status=$?
    if [[ "${exit_status}" -eq 0 ]]; then
        status="SUCCESS"
    else
        status="FAILED_${exit_status}"
    fi

    end_time=$(timestamp)
    end_ts=$(date +%s)
    elapsed_sec=$((end_ts - start_ts))

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${RUN_ID}" \
        "${dataset}" \
        "${traffic_file}" \
        "${start_time}" \
        "${end_time}" \
        "${elapsed_sec}" \
        "${status}" \
        "${log_file}" >> "${SUMMARY_TSV}"

    log_event "finish dataset=${dataset} traffic_file=${traffic_file} status=${status} elapsed_sec=${elapsed_sec}"
    return "${exit_status}"
}

main() {
    write_summary_header

    log_event "run_id=${RUN_ID}"
    log_event "base_dir=${BASE_DIR}"
    log_event "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    log_event "llm_model=${LLM_MODEL}"
    log_event "llm_path=${LLM_PATH}"
    log_event "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"

    local exit_code=0

    run_one jinan anon_3_4_jinan_real.json || exit_code=1
    run_one jinan anon_3_4_jinan_real_2000.json || exit_code=1
    run_one jinan anon_3_4_jinan_real_2500.json || exit_code=1
    run_one hangzhou anon_4_4_hangzhou_real.json || exit_code=1
    run_one hangzhou anon_4_4_hangzhou_real_5816.json || exit_code=1

    log_event "summary_tsv=${SUMMARY_TSV}"
    log_event "done exit_code=${exit_code}"
    return "${exit_code}"
}

main "$@"
