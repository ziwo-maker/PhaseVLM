#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${SIMEXPERT_RUN_ID:-hangzhou_real_simexpert_small_sweep_20260907}"
OUTPUT_DIR="${REPO_ROOT}/records/sim_expert_validation_search/${RUN_ID}"
PYTHON_BIN="${SIMEXPERT_PYTHON:-/home/data/sysu/anaconda3/envs/hnr5090/bin/python}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -c 'import cityflow; assert hasattr(cityflow, "Engine"), "cityflow.Engine unavailable"'

exec "${PYTHON_BIN}" scripts/run_sim_expert_horizon_weight_search.py \
    --dataset hangzhou \
    --traffic_files anon_4_4_hangzhou_real.json \
    --horizons 30,60 \
    --weight_sets "1,1,1,0.2;1.5,1,1,0.2;1,1.5,1,0.2;1,1,1.5,0.2;1,1,1,0.4" \
    --selection_att_weight 1.0 \
    --selection_aql_weight 1.0 \
    --selection_awt_weight 1.0 \
    --run_id "${RUN_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_workers 4 \
    --worker_cuda_devices 0,1,2,3 \
    --cuda_visible_devices 0 \
    --keep_candidate_logs \
    --extra_args \
        --disable_replay \
        --skip_state_action_log \
        --skip_env_batch_log \
        --memory_limit_gb 24 \
        --memory_check_interval_steps 5
