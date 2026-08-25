#!/usr/bin/env bash
# Autonomous ablation pipeline — no human intervention needed
# Ablation B: Pure BC on qwctx → Ablation C: BC+cost ranking on qwctx → iterate
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PIPELINE_STATE="/home/code/LLMSTSC/.aris/ablation_pipeline_state.txt"
PYTHON_BIN=/home/lijn99/miniconda3/envs/lab/bin/python
MODEL_PATH=/home/data/model/qwen-vl-8B
QWCTX_DATA=/mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857/raw/traffic_prompt_series_2s.jsonl
QWCTX_STAGE1=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_double_qwctx_20260731_040000/stage1/best.pt
BASELINE_SUM=2822

log_msg() { echo "$(date +%H:%M:%S) | $*"; }

# --- State management ---
get_state() {
    if [[ -f "$PIPELINE_STATE" ]]; then
        cat "$PIPELINE_STATE"
    else
        echo "IDLE"
    fi
}
set_state() { echo "$1" > "$PIPELINE_STATE"; }

# --- Check if training is done ---
training_done() {
    local run_root="$1"
    [[ -f "$run_root/completion.status" ]] && grep -q "status=success" "$run_root/completion.status"
}

# --- Run online test ---
run_online_test() {
    local run_root="$1"
    local ckpt="$run_root/stage2/checkpoints/best.pt"
    local test_dir="$run_root/online_double"
    local result_json="$test_dir/result.json"

    if [[ -f "$result_json" ]]; then
        log_msg "Online test already done: $result_json"
        return 0
    fi

    log_msg "Launching online test for $run_root"
    mkdir -p "$test_dir"
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 timeout 2400 "$PYTHON_BIN" projection_train/run_projection_online.py \
        --memo "pipeline_runs/$(basename "$run_root")/online_double" \
        --projection_ckpt "$ckpt" \
        --llm_path "$MODEL_PATH" \
        --dataset newyork_28x7 \
        --traffic_file anon_28_7_newyork_real_double.json \
        --run_counts 3600 --num_rounds 1 \
        --proj_name "NewYork28x7-ablation-online" \
        --device auto --seed 2024 \
        --result_json "$result_json" \
        --prompt_mode queue_wait_short \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --inference_batch_size 64 --bf16 \
        > "$test_dir/test.log" 2>&1
    log_msg "Online test completed: $result_json"
}

# --- Get Sum metric from result.json ---
get_sum() {
    local result_json="$1"
    "$PYTHON_BIN" -c "
import json
r = json.load(open('$result_json'))['results']
print(f\"{r['test_avg_queue_len_over'] + r['test_avg_waiting_time_over'] + r['test_avg_travel_time_over']:.2f}\")
" 2>/dev/null || echo "99999"
}

# --- Launch training ---
launch_training() {
    local run_root="$1"
    local data_path="$2"
    local stage1="$3"
    local target_mode="$4"
    local ranking_weight="${5:-0}"
    local kl_weight="${6:-0}"
    local soft_alpha="${7:-0.3}"

    if [[ -d "$run_root" ]]; then
        log_msg "Run root already exists: $run_root"
        return 1
    fi

    log_msg "Launching: target_mode=$target_mode ranking=$ranking_weight kl=$kl_weight alpha=$soft_alpha"
    mkdir -p "$run_root/logs" "$run_root/stage2/checkpoints"

    # Build extra args
    local extra_args="--target_mode $target_mode"
    if [[ "$target_mode" == "mixed" ]]; then
        extra_args="$extra_args --soft_target_alpha $soft_alpha --soft_label_temperature 0.35"
    fi
    if [[ "$(echo "$ranking_weight > 0" | bc -l)" == "1" ]]; then
        extra_args="$extra_args --ranking_loss_weight $ranking_weight --ranking_min_cost_gap 0.02"
    fi
    if [[ "$(echo "$kl_weight > 0" | bc -l)" == "1" ]]; then
        extra_args="$extra_args --base_policy_kl_weight $kl_weight --base_policy_prior_mass 0.70 --base_policy_kl_mode match_or_low_weight --base_policy_kl_low_weight_threshold 1.0"
    fi

    # Determine best_metric
    local best_metric="val_acc"
    local best_mode="max"
    if [[ "$(echo "$ranking_weight > 0" | bc -l)" == "1" ]]; then
        best_metric="val_top1_cost_regret"
        best_mode="min"
    fi

    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
    /home/lijn99/miniconda3/envs/lab/bin/torchrun --standalone --nproc_per_node=2 \
        projection_train/train_projection_bc.py \
        --data_path "$data_path" \
        --llm_path "$MODEL_PATH" --align_ckpt "$stage1" \
        --output_dir "$run_root/stage2/checkpoints" \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
        --batch_size 4 --grad_accum_steps 8 --epochs 15 \
        --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
        --weight_decay 1e-4 --max_grad_norm 1.0 \
        --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 \
        --log_interval 20 --save_every 1 --best_metric "$best_metric" --best_mode "$best_mode" \
        $extra_args \
        --ddp_timeout_seconds 1800 \
        > "$run_root/logs/pipeline.log" 2>&1

    printf 'completed_at=%s\nstatus=success\n' "$(date -Is)" > "$run_root/completion.status"
    log_msg "Training completed: $run_root"
}

# --- Main pipeline logic ---
STATE=$(get_state)
log_msg "Pipeline state: $STATE"

case "$STATE" in
    IDLE|WAITING_B)
        # Check Ablation B
        B_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_pure_bc_20260807_run01"
        if training_done "$B_ROOT"; then
            log_msg "Ablation B training done. Running online test."
            set_state "TESTING_B"
            run_online_test "$B_ROOT"
            B_SUM=$(get_sum "$B_ROOT/online_double/result.json")
            log_msg "Ablation B result: Sum=$B_SUM"
            set_state "DONE_B"
        else
            log_msg "Ablation B still training. Check later."
        fi
        ;;

    TESTING_B)
        B_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_pure_bc_20260807_run01"
        B_RESULT="$B_ROOT/online_double/result.json"
        if [[ -f "$B_RESULT" ]]; then
            B_SUM=$(get_sum "$B_RESULT")
            log_msg "Ablation B result: Sum=$B_SUM"
            set_state "DONE_B"
        else
            log_msg "Online test B still running."
        fi
        ;;

    DONE_B)
        B_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_pure_bc_20260807_run01"
        B_SUM=$(get_sum "$B_ROOT/online_double/result.json")
        log_msg "Ablation B Sum=$B_SUM (baseline=$BASELINE_SUM)"

        # Launch Ablation C: BC + cost ranking + KL on qwctx data
        C_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_costrank_20260808_run01"
        if [[ -d "$C_ROOT" ]]; then
            if training_done "$C_ROOT"; then
                log_msg "Ablation C training done."
                set_state "TESTING_C"
            else
                log_msg "Ablation C still training."
            fi
        else
            log_msg "Launching Ablation C: BC+cost ranking+KL on qwctx data"
            set_state "TRAINING_C"
            launch_training "$C_ROOT" "$QWCTX_DATA" "$QWCTX_STAGE1" "mixed" "0.15" "0.02" "0.3"
            set_state "DONE_C_TRAIN"
        fi
        ;;

    TRAINING_C|DONE_C_TRAIN)
        C_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_costrank_20260808_run01"
        if training_done "$C_ROOT"; then
            log_msg "Ablation C training done. Running online test."
            set_state "TESTING_C"
            run_online_test "$C_ROOT"
            C_SUM=$(get_sum "$C_ROOT/online_double/result.json")
            log_msg "Ablation C result: Sum=$C_SUM"
            set_state "DONE_C"
        else
            log_msg "Ablation C still training."
        fi
        ;;

    TESTING_C)
        C_ROOT="$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_costrank_20260808_run01"
        C_RESULT="$C_ROOT/online_double/result.json"
        if [[ -f "$C_RESULT" ]]; then
            C_SUM=$(get_sum "$C_RESULT")
            log_msg "Ablation C result: Sum=$C_SUM"

            # Compare with baseline
            if [[ "$(echo "$C_SUM < $BASELINE_SUM" | bc -l)" == "1" ]]; then
                log_msg "IMPROVEMENT! Ablation C Sum=$C_SUM < baseline=$BASELINE_SUM"
            else
                log_msg "No improvement: C Sum=$C_SUM >= baseline=$BASELINE_SUM"
            fi
            set_state "DONE_C"
        else
            log_msg "Online test C still running."
        fi
        ;;

    DONE_C)
        log_msg "Pipeline complete. Results:"
        B_SUM=$(get_sum "$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_pure_bc_20260807_run01/online_double/result.json")
        C_SUM=$(get_sum "$REPO_ROOT/records/pipeline_runs/newyork28x7_ablation_qwctx_costrank_20260808_run01/online_double/result.json")
        log_msg "  Ablation B (pure BC on qwctx): Sum=$B_SUM"
        log_msg "  Ablation C (BC+costrank+KL on qwctx): Sum=$C_SUM"
        log_msg "  Baseline: Sum=$BASELINE_SUM"
        ;;

    *)
        log_msg "Unknown state: $STATE. Resetting to IDLE."
        set_state "IDLE"
        ;;
esac
