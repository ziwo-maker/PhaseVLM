#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/LLMSTSC}"
PY_BIN="${PY_BIN:-/root/miniconda3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_TAG="${RUN_TAG:-stage3_gap_online_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/records/pipeline_runs/${RUN_TAG}}"

LLM_PATH="${LLM_PATH:-/gemini/pretrain2/qwen-vl-8B}"
JINAN_CKPT="${JINAN_CKPT:-${REPO_ROOT}/records/pipeline_runs/exp_jinan/stage3_jinan_real_eval5_updates10_20260521_024721/checkpoints/best.pt}"
HANGZHOU_CKPT="${HANGZHOU_CKPT:-${REPO_ROOT}/records/pipeline_runs/exp_hangzhou_5816/stage3_from_stage2best_nosnapshot_mb8ga4_u10_eval5_save1_20260522_070729/checkpoints/best.pt}"

mkdir -p "${OUT_ROOT}/logs"
STATUS_TSV="${OUT_ROOT}/status.tsv"
SUMMARY_JSON="${OUT_ROOT}/summary.json"
printf 'started_at\tcity\ttraffic_file\tgap\tstatus\tresult_json\tlog\n' > "${STATUS_TSV}"

cd "${REPO_ROOT}"

run_one() {
  local city="$1"
  local dataset="$2"
  local traffic_file="$3"
  local ckpt="$4"
  local gap="$5"
  local max_length="$6"
  local name="${city}_gap${gap}_${traffic_file%.json}"
  local root="${OUT_ROOT}/${name}"
  local log_path="${root}/logs/test.log"
  local result_json="${root}/artifacts/test_result.json"

  mkdir -p "${root}/logs" "${root}/artifacts"
  printf '[%s] start city=%s traffic=%s gap=%s ckpt=%s\n' \
    "$(date '+%F %T')" "${city}" "${traffic_file}" "${gap}" "${ckpt}" | tee -a "${OUT_ROOT}/logs/master.log"

  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY_BIN}" projection_train/run_projection_online.py \
      --memo "pipeline_runs/${RUN_TAG}/${name}/test_runs" \
      --projection_ckpt "${ckpt}" \
      --dataset "${dataset}" \
      --traffic_file "${traffic_file}" \
      --device auto \
      --seed 2024 \
      --num_rounds 1 \
      --proj_name LLM-TSCS-extreme \
      --run_counts 3600 \
      --result_json "${result_json}" \
      --max_phase_override_gap "${gap}" \
      --max_phase_override_left_wait_protect 45 \
      --llm_path "${LLM_PATH}" \
      --seq_len 16 \
      --image_size 224 \
      --periodicity 16 \
      --input_dim 8 \
      --hidden_dim 64 \
      --prefix_tokens 8 \
      --prefix_mode prefix \
      --max_length "${max_length}" \
      --prompt_mode image_only_short \
      --bf16 \
      > "${log_path}" 2>&1
  local status="$?"
  set -e

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%F %T')" "${city}" "${traffic_file}" "${gap}" "${status}" "${result_json}" "${log_path}" \
    >> "${STATUS_TSV}"
  printf '[%s] end city=%s traffic=%s gap=%s status=%s\n' \
    "$(date '+%F %T')" "${city}" "${traffic_file}" "${gap}" "${status}" | tee -a "${OUT_ROOT}/logs/master.log"
}

run_one "jinan" "jinan" "anon_3_4_jinan_real.json" "${JINAN_CKPT}" 2 128
run_one "jinan" "jinan" "anon_3_4_jinan_real_2000.json" "${JINAN_CKPT}" 3 128
run_one "jinan" "jinan" "anon_3_4_jinan_real_2500.json" "${JINAN_CKPT}" 4 128
run_one "jinan" "jinan" "anon_3_4_jinan_synthetic_24000_60min.json" "${JINAN_CKPT}" 5 128
run_one "jinan" "jinan" "anon_3_4_jinan_synthetic_24h_6000.json" "${JINAN_CKPT}" 6 128

run_one "hangzhou" "hangzhou" "anon_4_4_hangzhou_real_5816.json" "${HANGZHOU_CKPT}" 7 64
run_one "hangzhou" "hangzhou" "anon_4_4_hangzhou_real.json" "${HANGZHOU_CKPT}" 8 64
run_one "hangzhou" "hangzhou" "anon_4_4_hangzhou_synthetic_24000_60min.json" "${HANGZHOU_CKPT}" 9 64

"${PY_BIN}" - "${OUT_ROOT}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import os
import sys

out_root, summary_json = sys.argv[1], sys.argv[2]
status_path = os.path.join(out_root, "status.tsv")
rows = []
with open(status_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        result = {}
        if row["status"] == "0" and os.path.exists(row["result_json"]):
            with open(row["result_json"], encoding="utf-8") as result_handle:
                payload = json.load(result_handle)
            result = payload.get("results", payload)
        row["results"] = result
        rows.append(row)

with open(summary_json, "w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=2, ensure_ascii=False)
print(summary_json)
PY

printf '[%s] all experiments finished; summary=%s\n' "$(date '+%F %T')" "${SUMMARY_JSON}" | tee -a "${OUT_ROOT}/logs/master.log"
