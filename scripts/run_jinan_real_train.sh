#!/usr/bin/env bash
# Train+eval responsive H120 K=32 & K=118 on jinan (9 training seeds total),
# split across 3 parallel single-GPU jobs (GPUs 0/1/2), 3 seeds each.
# Each job runs 6 configs (2 budgets x 3 seeds) serially on one GPU.
set -euo pipefail

REPO=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
OUT="${OUTPUT_DIR:-$REPO/records/pipeline_runs/jinan_real_responsive_h120_20260814}"
LABELS="$OUT/relabeled_policy_responsive_h120.jsonl"
DATASET=jinan
TEST_FLOW=anon_3_4_jinan_real.json

mkdir -p "$OUT"
cd "$REPO"

# job <gpu> <seed1> <seed2> <seed3>
run_job() {
    local gpu="$1"; shift
    local seeds=("$@")
    local jobdir="$OUT/train_h120_gpu${gpu}"
    mkdir -p "$jobdir"
    "$PY" -u scripts/run_phase7_policy_responsive.py \
        --labels "$LABELS" \
        --output_dir "$jobdir" \
        --label_budgets 32 118 \
        --selection_seed 2024 \
        --training_seeds "${seeds[@]}" \
        --eval_seeds 2024 42 123 \
        --train_gpus "$gpu" \
        --eval_gpus "$gpu" \
        --loss mixed \
        --dataset "$DATASET" \
        --test_flow "$TEST_FLOW" \
        > "$jobdir/job.log" 2>&1
}

run_job 0 2024 42 123 &
p0=$!
run_job 1 7 11 99 &
p1=$!
run_job 2 314 2718 9001 &
p2=$!

echo "launched 3 training jobs: pids $p0 $p1 $p2"
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
wait "$p2" || rc=1
exit $rc
