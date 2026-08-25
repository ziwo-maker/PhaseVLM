#!/usr/bin/env bash
set -Eeuo pipefail

# Train one additional seed for the paper's primary DUCPD-vs-hard-BC comparison.
# Each arm deliberately reserves all three local GPUs for the same DDP job.

RUN_ROOT="${1:?Usage: $0 RUN_ROOT ARM SEED}"
ARM="${2:?Usage: $0 RUN_ROOT ARM SEED}"
SEED="${3:?Usage: $0 RUN_ROOT ARM SEED}"

REPO_ROOT=/home/code/LLMSTSC
PYTHON_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/python
TORCHRUN_BIN=/home/sysu/anaconda3/envs/hnr5090/bin/torchrun
MODEL_PATH=/home/data/model/qwen-vl-8B
SOURCE_RUN_ROOT="$REPO_ROOT/records/pipeline_runs/hangzhou_real_direct_urgency_20260820"
DATA_PATH="$SOURCE_RUN_ROOT/data/train_5816_direct_phase_urgency_v1.jsonl"
ALIGN_CKPT="$REPO_ROOT/records/pipeline_runs/hangzhou_pt32_hd256_20260813/stage1/best.pt"
ARM_ROOT="$RUN_ROOT/arms/$ARM/seed_$SEED"
STAGE2_DIR="$ARM_ROOT/stage2"
LOG_DIR="$ARM_ROOT/logs"
STATUS_PATH="$ARM_ROOT/status.txt"
CONFIG_PATH="$ARM_ROOT/config.json"
PREFIX_TOKENS="${PREFIX_TOKENS:-32}"

[[ "$PREFIX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "PREFIX_TOKENS must be a positive integer: $PREFIX_TOKENS" >&2; exit 2; }

case "$ARM" in
    hard_bc)
        TARGET_ARGS=(--target_mode hard --ranking_loss_weight 0 --base_policy_kl_weight 0)
        ;;
    wait2_mixed_rank)
        TARGET_ARGS=(--target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0 --cost_aql_weight 1 --cost_awt_weight 2 --cost_att_weight 0 --cost_spillback_weight 0)
        ;;
    *)
        echo "Unsupported robustness arm: $ARM" >&2
        exit 2
        ;;
esac

[[ -s "$DATA_PATH" ]] || { echo "Missing direct-urgency data: $DATA_PATH" >&2; exit 1; }
[[ -s "$ALIGN_CKPT" ]] || { echo "Missing Stage 1 checkpoint: $ALIGN_CKPT" >&2; exit 1; }
[[ ! -e "$ARM_ROOT" ]] || { echo "Run directory already exists: $ARM_ROOT" >&2; exit 1; }
for gpu in 0 1 2; do
    used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -dc '0-9')"
    [[ -n "$used_memory" && "$used_memory" -lt 500 ]] || { echo "GPU $gpu is busy: ${used_memory:-unknown} MiB" >&2; exit 1; }
done

mkdir -p "$STAGE2_DIR" "$LOG_DIR"
"$PYTHON_BIN" - "$CONFIG_PATH" "$ARM" "$SEED" "$DATA_PATH" "$ALIGN_CKPT" "$PREFIX_TOKENS" <<'PY'
import json
import sys
from datetime import datetime, timezone

config_path, arm, seed, data_path, align_ckpt, prefix_tokens = sys.argv[1:]
payload = {
    "schema": "hangzhou_ducpd_robustness_train_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "arm": arm,
    "training_seed": int(seed),
    "data_path": data_path,
    "align_ckpt": align_ckpt,
    "ddp_world_size": 3,
    "cuda_visible_devices": "0,1,2",
    "architecture": {"seq_len": 16, "image_size": 224, "hidden_dim": 256, "prefix_tokens": int(prefix_tokens)},
    "optimization": {"epochs": 15, "batch_size_per_rank": 4, "grad_accum_steps": 8, "lr": 1e-4, "seed_split": "stratified"},
    "checkpoint_selection": {"metric": "val_acc", "mode": "max", "evaluation_not_used": True},
    "supervision": {
        "schema": "direct_phase_urgency_v1",
        "semantics": "observed_phase_queue_wait_no_rollout",
        "counterfactual_rollouts": False,
        "hard_rule_label": True,
    },
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

on_exit() {
    local code="$?"
    if [[ "$code" -eq 0 && -s "$STAGE2_DIR/best.pt" ]]; then
        printf 'completed_at=%s\nstatus=success\nstage2_best=%s\n' "$(date -Is)" "$STAGE2_DIR/best.pt" > "$STATUS_PATH"
    else
        printf 'completed_at=%s\nstatus=failed\nexit_code=%s\n' "$(date -Is)" "$code" > "$STATUS_PATH"
    fi
    exit "$code"
}
trap on_exit EXIT

{
    printf 'started_at=%s\nstatus=running\narm=%s\nseed=%s\n' "$(date -Is)" "$ARM" "$SEED"
    printf 'data=%s\nalign_ckpt=%s\nconfig=%s\n' "$DATA_PATH" "$ALIGN_CKPT" "$CONFIG_PATH"
    printf 'counterfactual_rollouts=false\ncheckpoint_selection=val_acc_max\n'
} > "$STATUS_PATH"

exec > "$LOG_DIR/pipeline.log" 2>&1
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$TORCHRUN_BIN" --standalone --nproc_per_node=3 projection_train/train_projection_bc.py \
    --data_path "$DATA_PATH" \
    --llm_path "$MODEL_PATH" --align_ckpt "$ALIGN_CKPT" --output_dir "$STAGE2_DIR" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
    --prefix_tokens "$PREFIX_TOKENS" --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
    --batch_size 4 --grad_accum_steps 8 --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy stratified \
    --num_workers 4 --seed "$SEED" --bf16 --disable_class_balance \
    --log_interval 20 --save_every 1 --checkpoint_slim --best_metric val_acc --best_mode max \
    "${TARGET_ARGS[@]}" --ddp_timeout_seconds 1800
