#!/usr/bin/env bash
# Jinan-1 / single-seed / 4-GPU Stage-I convergence study.
# Three arms: aligned -> Stage II, random -> Stage II, shuffled captions -> Stage II.
# One command: bash /home/code/LLMSTSC/recordlocal/run_stage1_efficiency_oneflow.sh
# Preflight:   bash /home/code/LLMSTSC/recordlocal/run_stage1_efficiency_oneflow.sh --preflight-only
set -Eeuo pipefail

REPO=/home/code/LLMSTSC
OUTPUT_BASE="$REPO/recordlocal"
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
TORCHRUN=${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
DATA="$REPO/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
TRAFFIC="anon_3_4_jinan_real.json"
GPU_LIST=${GPUS:-0,1,2,3}
SEED=2024
HIDDEN_DIM=256
MAX_LENGTH=512
STAGE1_EPOCHS=15
STAGE2_EPOCHS=15
EVAL_EPOCHS=(1 3 5 10 15)
ARMS=(full without_stage1 shuffled_alignment)
RUN_ID=${RUN_ID:-stage1_convergence_jinan1_s2024_$(date +%Y%m%d_%H%M%S)}
ROOT="$OUTPUT_BASE/runs/$RUN_ID"
HELPER="$OUTPUT_BASE/stage1_efficiency_helper.py"

cd "$REPO"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
if [[ ${#GPU_ARRAY[@]} -ne 4 ]]; then
  echo "This controlled comparison requires exactly four GPUs: GPUS=0,1,2,3" >&2
  exit 2
fi
for path in "$PYTHON" "$TORCHRUN" "$MODEL" "$DATA" "$HELPER" "$REPO/data/Jinan/3_4/$TRAFFIC"; do
  if [[ ! -e "$path" ]]; then echo "Missing dependency: $path" >&2; exit 2; fi
done
if [[ ! -x "$TORCHRUN" ]]; then echo "torchrun is not executable" >&2; exit 2; fi
if [[ "$GPU_LIST" != "0,1,2,3" ]] && [[ "$GPU_LIST" != "1,2,3,0" ]]; then
  echo "Using explicitly selected GPU ids: $GPU_LIST"
fi
if ! timeout --foreground 12s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >/dev/null; then
  echo "GPU preflight failed" >&2; exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then echo "Invalid GPU id: $gpu" >&2; exit 2; fi
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
  if [[ -z "$used" || "$used" -gt 4096 ]]; then
    echo "GPU $gpu is occupied (used=${used:-unknown} MiB). Refusing to disturb existing work." >&2
    exit 3
  fi
done
if [[ "${1:-}" == "--preflight-only" ]]; then
  printf 'PREFLIGHT_OK dataset=%s model=%s seed=%s GPUS=%s effective_stage1_batch=16 effective_stage2_batch=32\n' "$DATA" "$MODEL" "$SEED" "$GPU_LIST"
  exit 0
fi
if [[ -e "$ROOT" ]]; then
  echo "Run directory already exists: $ROOT; use a new RUN_ID (no implicit overwrite)" >&2
  exit 4
fi
mkdir -p "$ROOT/logs" "$ROOT/derived" "$ROOT"/{full,without_stage1,shuffled_alignment}
printf 'time\tstage\tarm\tstatus\tdetail\n' > "$ROOT/status.tsv"
printf '%s\n' "$$" > "$ROOT/launcher.pid"
cp "$OUTPUT_BASE/run_stage1_efficiency_oneflow.sh" "$ROOT/launcher.snapshot.sh"
cp "$HELPER" "$ROOT/helper.snapshot.py"

note() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" >> "$ROOT/status.tsv"
  printf '[%s] stage=%s arm=%s state=%s detail=%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4"
}
finalize() {
  local code=$?
  trap - EXIT
  if [[ "$code" -eq 0 ]]; then
    note all all completed "$ROOT/REPORT.md"
  else
    note all all failed "exit_code=$code; see logs"
  fi
  exit "$code"
}
trap finalize EXIT

export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

note preflight all complete "GPUS=$GPU_LIST"
"$PYTHON" - "$ROOT/manifest.json" "$DATA" "$MODEL" "$GPU_LIST" "$SEED" "$HIDDEN_DIM" "$MAX_LENGTH" "$TRAFFIC" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, data, model, gpus, seed, hidden, max_length, traffic = sys.argv[1:]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()
manifest = {
    "experiment": "Stage I convergence and training efficiency; one flow and seed",
    "dataset": "Jinan-1", "training_file": data, "data_sha256": sha(data),
    "test_traffic_file": traffic, "model_path": model,
    "seed": int(seed), "gpu_ids": gpus.split(","),
    "arms": ["full", "without_stage1", "shuffled_alignment"],
    "stage1_epochs": 15, "stage2_epochs": 15,
    "stage1_effective_batch": 16, "stage2_effective_batch": 32,
    "stage1_per_gpu_batch": 4, "stage2_per_gpu_batch": 1,
    "stage1_gradient_accumulation": 1, "stage2_gradient_accumulation": 8,
    "architecture": {"sequence_length":16, "image_size":224, "input_dim":8,
                     "hidden_dim":int(hidden), "prefix_tokens":8, "max_length":int(max_length)},
    "stage2_supervision": {"target_mode": "mixed", "soft_alpha":0.3,
                           "soft_label_temperature":0.7, "ranking_weight":0,
                           "base_policy_kl_weight":0, "resample_mode": "none"},
    "checkpoint_selection": "min validation top1 cost regret, NOT online evaluation",
    "online_checkpoints": [1,3,5,10,15],
    "online": {"seed":int(seed),"run_counts":3600, "min_action_time":30,
               "phase_overrides":False, "test_flow_matches_training_flow":True},
    "no_multiseed_claim": True,
    "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "code_sha256": {p:sha(p) for p in ("train_align.py",
                        "projection_train/train_projection_bc.py",
                        "projection_train/run_projection_online.py")},
}
pathlib.Path(out).write_text(json.dumps(manifest,indent=2)+"\n")
PY

note prepare shuffled_alignment running 'Shuffling only labeled decision captions'
"$PYTHON" "$HELPER" shuffle --source "$DATA" \
  --target "$ROOT/derived/shuffled_caption_data.jsonl" \
  --info "$ROOT/derived/shuffle_manifest.json" --seed 99173 \
  > "$ROOT/logs/shuffle.log" 2>&1
note prepare shuffled_alignment complete "$ROOT/derived/shuffle_manifest.json"

run_align() {
  local arm=$1 dataset=$2
  local dest="$ROOT/$arm/stage1"
  mkdir -p "$dest"
  note stage1 "$arm" running '15 epochs / 4 GPUs; decision_only; no action auxiliary'
  local start=$SECONDS
  timeout --foreground 7200s env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    "$TORCHRUN" --standalone --nproc_per_node=4 train_align.py \
      --data_path "$dataset" --model_path "$MODEL" --output_dir "$dest" \
      --decision_only --seq_len 16 --image_size 224 --periodicity 16 \
      --input_dim 8 --hidden_dim "$HIDDEN_DIM" --prefix_tokens 8 \
      --prefix_mode prefix --max_length "$MAX_LENGTH" \
      --temperature 0.05 --action_aux_weight 0 \
      --batch_size 4 --grad_accum_steps 1 --epochs "$STAGE1_EPOCHS" \
      --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
      --num_workers 2 --val_ratio 0.1 --eval_max_samples 1000 \
      --save_every_epochs 15 --log_every_steps 25 --seed "$SEED" --bf16 \
    > "$ROOT/logs/stage1_$arm.log" 2>&1
  [[ -s "$dest/best.pt" && -s "$dest/run_summary.json" ]] ||
    { echo "Stage I did not finish for $arm" >&2; return 1; }
  note stage1 "$arm" complete "elapsed_seconds=$((SECONDS-start))"
}

run_stage2() {
  local arm=$1
  local dest="$ROOT/$arm/stage2"
  local init=(--init_mode random)
  if [[ "$arm" == full || "$arm" == shuffled_alignment ]]; then
    init=(--init_mode align --align_ckpt "$ROOT/$arm/stage1/best.pt")
  fi
  mkdir -p "$dest"
  note stage2 "$arm" running '15 epochs / 4 GPUs / val_top1_cost_regret selector'
  local start=$SECONDS
  timeout --foreground 14400s env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    "$TORCHRUN" --standalone --nproc_per_node=4 \
      projection_train/train_projection_bc.py \
        --data_path "$DATA" --llm_path "$MODEL" "${init[@]}" \
        --output_dir "$dest" --seq_len 16 --image_size 224 \
        --periodicity 16 --input_dim 8 --hidden_dim "$HIDDEN_DIM" \
        --prefix_tokens 8 --prefix_mode prefix --max_length "$MAX_LENGTH" \
        --prompt_mode queue_wait_short --action_eval_chunk_size 0 \
        --batch_size 1 --grad_accum_steps 8 --epochs "$STAGE2_EPOCHS" \
        --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
        --weight_decay 1e-4 --max_grad_norm 1 --val_ratio 0.1 \
        --split_strategy stratified --num_workers 2 \
        --target_mode mixed --soft_target_alpha 0.3 \
        --soft_label_temperature 0.7 --ranking_loss_weight 0 \
        --base_policy_kl_weight 0 --resample_mode none \
        --best_metric val_top1_cost_regret --best_mode min \
        --log_interval 25 --save_every 1 --seed "$SEED" \
        --checkpoint_slim --bf16 --ddp_timeout_seconds 1800 \
    > "$ROOT/logs/stage2_$arm.log" 2>&1
  [[ -s "$dest/best.pt" && -s "$dest/epoch_15.pt" && -s "$dest/metrics.jsonl" ]] ||
    { echo "Stage II did not finish for $arm" >&2; return 1; }
  note stage2 "$arm" complete "elapsed_seconds=$((SECONDS-start))"
  "$PYTHON" "$HELPER" summarize --root "$ROOT" \
    >> "$ROOT/logs/summarize.log" 2>&1
}
run_align full "$DATA"
run_align shuffled_alignment "$ROOT/derived/shuffled_caption_data.jsonl"
for arm in "${ARMS[@]}"; do
  run_stage2 "$arm"
done

note online all running 'Three arms x five selected checkpoints; 4 parallel single-GPU evaluations'
online_one() {
  local arm=$1 epoch=$2 gpu=$3
  local out="$ROOT/$arm/online/epoch_$epoch"
  mkdir -p "$out"
  # Both model and simulator output paths resolve inside recordlocal/runs/$RUN_ID.
  local memo="../recordlocal/runs/$RUN_ID/online_env/$arm/epoch_$epoch"
  local log="$ROOT/logs/online_${arm}_ep${epoch}.log"
  local start=$SECONDS
  note online "$arm" running "epoch=$epoch,gpu=$gpu"
  if ! timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "$memo" --projection_ckpt "$ROOT/$arm/stage2/epoch_$epoch.pt" \
      --llm_path "$MODEL" --dataset jinan --traffic_file "$TRAFFIC" \
      --run_counts 3600 --num_rounds 1 --min_action_time 30 --seed "$SEED" \
      --proj_name "StageI-Convergence-${arm}-ep${epoch}" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
      --periodicity 16 --input_dim 8 --hidden_dim "$HIDDEN_DIM" \
      --prefix_tokens 8 --prefix_mode prefix --max_length "$MAX_LENGTH" \
      --traffic_series_interval 2 --max_phase_override_gap 0 \
      --bf16 --deterministic --disable_snapshots \
      --result_json "$out/result.json" > "$log" 2>&1; then
    note online "$arm" failed "epoch=$epoch,gpu=$gpu; see $log"
    return 1
  fi
  [[ -s "$out/result.json" ]] || { note online "$arm" failed "missing result ep=$epoch"; return 1; }
  note online "$arm" complete "epoch=$epoch,gpu=$gpu,seconds=$((SECONDS-start))"
}
# Fixed batches of four single-GPU evaluations; no GPU memory contention.
job_pids=()
job_labels=()
idx=0
failures=0
wait_group() {
  local i
  for i in "${!job_pids[@]}"; do
    if ! wait "${job_pids[$i]}"; then
      note online "${job_labels[$i]}" failed 'child evaluation failed'
      failures=1
    fi
  done
  job_pids=()
  job_labels=()
  "$PYTHON" "$HELPER" summarize --root "$ROOT" \
    >> "$ROOT/logs/summarize.log" 2>&1 || true
}
for arm in "${ARMS[@]}"; do
  for epoch in "${EVAL_EPOCHS[@]}"; do
    gpu="${GPU_ARRAY[$((idx % 4))]}"
    online_one "$arm" "$epoch" "$gpu" &
    job_pids+=("$!")
    job_labels+=("$arm:ep$epoch")
    idx=$((idx+1))
    if (( ${#job_pids[@]} == 4 )); then
      wait_group
      if ((failures != 0)); then exit 1; fi
    fi
  done
done
if (( ${#job_pids[@]} > 0 )); then wait_group; fi
if ((failures != 0)); then exit 1; fi
"$PYTHON" "$HELPER" summarize --root "$ROOT" >> "$ROOT/logs/summarize.log" 2>&1
note report all complete "$ROOT/REPORT.md"
printf 'SUCCESS: %s\n' "$ROOT"
