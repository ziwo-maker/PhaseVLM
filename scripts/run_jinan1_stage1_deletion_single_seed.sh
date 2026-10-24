#!/usr/bin/env bash
# Jinan-1 Stage I deletion: one training seed, four-GPU DDP, paired online seeds.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
TORCHRUN=${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}
MODEL_PATH=${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}
STAGE1_CKPT=${STAGE1_CKPT:-$REPO_ROOT/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/stage1/best.pt}
RUN_ROOT=${RUN_ROOT:-/mnt/data/LLMSTSCrecords/records/pipeline_runs/jinan1_stage1_deletion_seed2024_$(date +%Y%m%d_%H%M%S)}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_LIST <<< "$TRAIN_GPUS"
TRAIN_SEED=2024
ONLINE_SEEDS=(2024 42 123 7)
DATA_PATHS=(
  "$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
  "$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl"
  "$REPO_ROOT/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl"
)
LOG_DIR=$RUN_ROOT/logs
STATUS=$RUN_ROOT/status.tsv
NOTIFIER_PID=
START_SECONDS=$(date +%s)

timestamp() { date --iso-8601=seconds; }
status() {
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" "$3" >> "$STATUS"
}
latest_progress() {
  local metric
  metric=$(timeout --foreground 5s "$PYTHON" - "$RUN_ROOT" <<'PY' 2>/dev/null || true
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for arm in ('random_30', 'random_15', 'full_15'):
    path = root / 'arms' / arm / 'checkpoints' / 'metrics.jsonl'
    if path.is_file():
        lines = path.read_text().splitlines()
        if lines:
            row = json.loads(lines[-1])
            print(f"{arm} epoch={row.get('epoch')} val_macro_f1={row.get('val_macro_f1', 'NA')}")
            break
PY
)
  printf 'Jinan-1 Stage I deletion: %s; latest=%s; run=%s' "$(tail -n 1 "$STATUS" | cut -f2-)" "${metric:-pending}" "$RUN_ROOT"
}
notify() {
  timeout --foreground 15s "$PYTHON" "$REPO_ROOT/scripts/feishu_push.py" "$1" > /dev/null 2>&1 || true
}
cleanup() {
  local code=$?
  trap - EXIT
  if [[ -n "$NOTIFIER_PID" ]]; then
    kill "$NOTIFIER_PID" 2>/dev/null || true
    wait "$NOTIFIER_PID" 2>/dev/null || true
  fi
  status campaign all "exit=$code"
  if (( $(date +%s) - START_SECONDS >= 3600 )); then
    notify "$(latest_progress); exit=$code"
  fi
  exit "$code"
}

[[ ${#GPU_LIST[@]} -eq 4 ]] || { echo 'Exactly four GPU IDs are required' >&2; exit 2; }
for path in "$PYTHON" "$TORCHRUN" "$MODEL_PATH" "$STAGE1_CKPT" "${DATA_PATHS[@]}" "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_real.json"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$RUN_ROOT/arms"
printf 'time\tstage\tarm\tdetail\n' > "$STATUS"
trap cleanup EXIT
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

timeout --foreground 30s "$PYTHON" - "$STAGE1_CKPT" "${DATA_PATHS[@]}" <<'PY'
import hashlib, json, pathlib, shlex, sys
ckpt = pathlib.Path(sys.argv[1])
data = [pathlib.Path(p) for p in sys.argv[2:]]
root = ckpt.parent.parent
cfg = json.loads((ckpt.parent / 'run_config.json').read_text())
expected = ['da3a7170346a344a87edde9b509c5c48e4b2317f6a0f5b7fce17f1c6c285cffb',
            'cb4c904dcc01923da573e524d10a65eebabfe3ee9276ded0d0d1225a1f990beb',
            '9d6b25ea5e2fac624a75e4460fa170e9c9d89e9a888106a3709066deeeba3ece']
assert cfg['data_path'] == [str(p) for p in data]
assert cfg['epochs'] == 15 and cfg['temperature'] == 0.05
command = shlex.split(cfg['command'])
assert command[command.index('--seed') + 1] == '2024'
assert command[command.index('--model_path') + 1] == '/mnt/data/Qwen3-VL-8B-Instruct'
for path, digest in zip(data, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
print(f'Validated Stage I and data: {root}')
PY

"$PYTHON" - "$RUN_ROOT/manifest.json" "$STAGE1_CKPT" "$TRAIN_GPUS" "${DATA_PATHS[@]}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, ckpt, gpus, *data = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
root = pathlib.Path('/home/code/LLMSTSC')
manifest = {
    'experiment': 'jinan1_stage1_deletion_single_training_seed',
    'training_seed': 2024, 'online_seeds': [2024, 42, 123, 7],
    'model_path': '/mnt/data/Qwen3-VL-8B-Instruct',
    'training_data': [{'path': p, 'sha256': sha(p)} for p in data],
    'training_flows': ['anon_3_4_jinan_real.json', 'anon_3_4_jinan_real_2000.json', 'anon_3_4_jinan_real_2500.json'],
    'test_flow': 'anon_3_4_jinan_real.json',
    'stage1_checkpoint': ckpt, 'stage1_sha256': sha(ckpt), 'stage1_epochs': 15,
    'stage2_arms': {'full_15': {'init_mode': 'align', 'epochs': 15},
                    'random_15': {'init_mode': 'random', 'epochs': 15},
                    'random_30': {'init_mode': 'random', 'epochs': 30, 'scope': 'budget_sensitivity_only'}},
    'checkpoint_selection': 'best.pt by val_macro_f1, no online-test selection',
    'gpus': gpus, 'code_sha256': {p: sha(root / p) for p in
        ['projection_train/train_projection_bc.py', 'projection_train/run_projection_online.py']},
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
}
pathlib.Path(out).write_text(json.dumps(manifest, indent=2) + '\n')
PY

(
  while sleep 2400; do notify "$(latest_progress)"; done
) &
NOTIFIER_PID=$!

train_arm() {
  local arm=$1 epochs=$2 init_mode=$3
  local out=$RUN_ROOT/arms/$arm/checkpoints
  mkdir -p "$out"
  status train "$arm" "running epochs=$epochs init=$init_mode"
  local init_args=(--init_mode "$init_mode")
  [[ "$init_mode" == align ]] && init_args+=(--align_ckpt "$STAGE1_CKPT")
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node=4 \
    projection_train/train_projection_bc.py \
    --data_path "${DATA_PATHS[@]}" --llm_path "$MODEL_PATH" "${init_args[@]}" \
    --output_dir "$out" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --prompt_mode queue_wait_short --action_eval_chunk_size 0 \
    --batch_size 4 --grad_accum_steps 4 \
    --epochs "$epochs" --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers 4 \
    --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 \
    --best_metric val_macro_f1 --best_mode max --best_min_delta 0.0 \
    --resample_mode targeted \
    --resample_late_mode date_step --resample_late_interval_seconds 2 --resample_late_start_step 316 --resample_late_weight 1 \
    --resample_worst_intersections intersection_1_1,intersection_1_2,intersection_1_3,intersection_2_2,intersection_3_2,intersection_4_2 \
    --resample_worst_intersection_weight 2 \
    --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
    --resample_high_queue_threshold 14 --resample_high_queue_weight 1 \
    --resample_high_wait_threshold 84 \
    --resample_target_action_counts ETWT:4248,NTST:3934,ELWL:2055,NLSL:1885 \
    --log_interval 20 --save_every 1 --seed "$TRAIN_SEED" --bf16 \
    --ddp_timeout_seconds 1800 > "$LOG_DIR/train_$arm.log" 2>&1
  [[ -s "$out/best.pt" && -s "$out/metrics.jsonl" ]]
  status train "$arm" "completed checkpoint=$out/best.pt"
}

evaluate_arm() {
  local arm=$1 ckpt=$RUN_ROOT/arms/$1/checkpoints/best.pt
  local -a pids=()
  status online "$arm" running
  for i in "${!ONLINE_SEEDS[@]}"; do
    local seed=${ONLINE_SEEDS[$i]} gpu=${GPU_LIST[$i]}
    local out=$RUN_ROOT/arms/$arm/online/seed_$seed
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" projection_train/run_projection_online.py \
      --memo "online_tests/jinan1_stage1_deletion/$(basename "$RUN_ROOT")/$arm/seed_$seed" \
      --projection_ckpt "$ckpt" --llm_path "$MODEL_PATH" \
      --dataset jinan --traffic_file anon_3_4_jinan_real.json \
      --run_counts 3600 --num_rounds 1 --min_action_time 30 --seed "$seed" \
      --proj_name "Jinan1-stage1-deletion-$arm" --prompt_mode queue_wait_short \
      --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
      --traffic_series_interval 2 --max_phase_override_gap 0 \
      --bf16 --deterministic --disable_snapshots \
      --result_json "$out/result.json" > "$LOG_DIR/online_${arm}_seed_${seed}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || return 1
  for seed in "${ONLINE_SEEDS[@]}"; do
    [[ -s "$RUN_ROOT/arms/$arm/online/seed_$seed/result.json" ]] || return 1
  done
  status online "$arm" completed
}

for spec in 'full_15 15 align' 'random_15 15 random' 'random_30 30 random'; do
  read -r arm epochs mode <<< "$spec"
  train_arm "$arm" "$epochs" "$mode"
  evaluate_arm "$arm"
done

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
keys = {'ATT': 'test_avg_travel_time_over', 'AQL': 'test_avg_queue_len_over', 'AWT': 'test_avg_waiting_time_over'}
arms = {}
for arm in ('full_15', 'random_15', 'random_30'):
    rows = {}
    for seed in (2024, 42, 123, 7):
        payload = json.loads((root / 'arms' / arm / 'online' / f'seed_{seed}' / 'result.json').read_text())
        result = payload['results']
        if isinstance(result, list): result = result[-1]
        assert payload['args']['seed'] == seed
        assert payload['args']['traffic_file'] == 'anon_3_4_jinan_real.json'
        rows[str(seed)] = {name: float(result[key]) for name, key in keys.items()}
    arms[arm] = {'by_online_seed': rows, 'mean': {name: statistics.mean(row[name] for row in rows.values()) for name in keys}}
summary = {'arms': arms, 'delta_random15_minus_full15': {}, 'delta_random30_minus_full15': {}}
for name in keys:
    for arm, field in [('random_15', 'delta_random15_minus_full15'), ('random_30', 'delta_random30_minus_full15')]:
        summary[field][name] = arms[arm]['mean'][name] - arms['full_15']['mean'][name]
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
status summary all "completed path=$RUN_ROOT/summary.json"
