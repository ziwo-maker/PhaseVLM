#!/usr/bin/env bash
# Paired Stage-I deletion on five same-flow tasks, one training/simulator seed.
set -Eeuo pipefail

REPO=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
TORCHRUN=${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
GPUS=${GPUS:-1,2,3}
RUN_ROOT=${RUN_ROOT:-/mnt/data/LLMSTSCrecords/records/pipeline_runs/stage1_deletion_fiveflow_3gpu_$(date +%Y%m%d_%H%M%S)}
SEED=2024
FLOWS=(jinan1 jinan2 jinan3 hangzhou1 hangzhou2)
DATA=(
  data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
  data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
  data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl
  data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl
  data/processed/hangzhou_real_5816_stage3_queue_wait_rule_gap2_p60_20260531.jsonl
)
TRAFFIC=(
  anon_3_4_jinan_real.json
  anon_3_4_jinan_real_2000.json
  anon_3_4_jinan_real_2500.json
  anon_4_4_hangzhou_real.json
  anon_4_4_hangzhou_real_5816.json
)

cd "$REPO"
[[ "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]] || { echo 'GPUS must contain three IDs' >&2; exit 2; }
for path in "$PYTHON" "$TORCHRUN" "$MODEL" "${DATA[@]}"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 2; }
done
[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tasks"
STATUS="$RUN_ROOT/status.tsv"
printf 'time\tflow\tarm\tstage\tstate\tdetail\n' > "$STATUS"
stamp() { date --iso-8601=seconds; }
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(stamp)" "$1" "$2" "$3" "$4" "$5" >> "$STATUS"; }
notify() {
  local current metric
  current=$(tail -n 1 "$STATUS" | tr '\t' ' ')
  metric=$("$PYTHON" - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
paths = sorted(pathlib.Path(sys.argv[1]).glob('tasks/*/*/stage2/metrics.jsonl'))
for path in reversed(paths):
    rows = path.read_text().splitlines()
    if rows:
        row = json.loads(rows[-1])
        print(f'{path.parts[-3]}/{path.parts[-2]} epoch={row.get("epoch")} val_macro_f1={row.get("val_macro_f1", "NA")}')
        break
PY
  )
  timeout --foreground 15s "$PYTHON" scripts/feishu_push.py \
    "Stage I five-flow ablation: $current; latest=${metric:-pending}; run=$RUN_ROOT" \
    >/dev/null 2>&1 || true
}
notifier_pid=
finish() {
  local rc=$?
  trap - EXIT
  if [[ -n "$notifier_pid" ]]; then
    kill "$notifier_pid" 2>/dev/null || true
    wait "$notifier_pid" 2>/dev/null || true
  fi
  record campaign all all "$([[ $rc -eq 0 ]] && echo complete || echo failed)" "exit=$rc"
  notify
  exit "$rc"
}
trap finish EXIT

export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON" - "$RUN_ROOT/manifest.json" "$GPUS" "$MODEL" "${DATA[@]}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, gpus, model, *data = sys.argv[1:]
flows = ['jinan1', 'jinan2', 'jinan3', 'hangzhou1', 'hangzhou2']
traffic = ['anon_3_4_jinan_real.json', 'anon_3_4_jinan_real_2000.json',
           'anon_3_4_jinan_real_2500.json', 'anon_4_4_hangzhou_real.json',
           'anon_4_4_hangzhou_real_5816.json']
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
manifest = {
    'experiment': 'stage1_traffic_semantic_alignment_deletion',
    'training_seed': 2024, 'online_seed': 2024, 'gpus': gpus.split(','),
    'model_path': model, 'architecture': {'seq_len': 16, 'image_size': 224,
        'input_dim': 8, 'hidden_dim': 256, 'prefix_tokens': 8,
        'prefix_mode': 'prefix', 'max_length': 512},
    'stage1_epochs': 12, 'stage2_epochs': 12, 'per_gpu_batch_size': 1,
    'stage1_grad_accum_steps': 5, 'stage2_grad_accum_steps': 11,
    'arms': {'full': 'stage1_best_then_stage2_12',
             'without_stage1': 'random_init_then_stage2_12',
             'jinan1_budget_control': 'random_init_then_stage2_24; sensitivity_only'},
    'stage2_supervision': {'target_mode': 'mixed', 'soft_target_alpha': 0.3,
                           'soft_label_temperature': 0.7},
    'checkpoint_selection': 'best.pt by val_macro_f1; offline-only',
    'evaluation': {'scope': 'same_flow', 'run_counts': 3600, 'rounds': 1,
                   'min_action_time': 30, 'traffic_series_interval': 2,
                   'max_phase_override_gap': 0, 'prompt_mode': 'queue_wait_short'},
    'flows': [{'name': n, 'training_flow': t, 'test_flow': t,
               'data_path': p, 'data_sha256': sha(p)}
              for n, t, p in zip(flows, traffic, data)],
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'code_sha256': {p: sha(p) for p in
                    ['train_align.py', 'projection_train/train_projection_bc.py',
                     'projection_train/run_projection_online.py']},
    'interpretation': 'One paired training seed per flow; five-flow direction is not multi-seed stability.'
}
pathlib.Path(out).write_text(json.dumps(manifest, indent=2) + '\n')
PY

( while sleep 2400; do notify; done ) &
notifier_pid=$!

run_stage1() {
  local flow=$1 data=$2 out="$RUN_ROOT/tasks/$1/stage1"
  mkdir -p "$out"
  record "$flow" full stage1 running 'epochs=12'
  CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node=3 train_align.py \
    --data_path "$data" --model_path "$MODEL" --output_dir "$out" --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 --temperature 0.05 \
    --batch_size 1 --grad_accum_steps 5 --epochs 12 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --num_workers 4 --val_ratio 0.1 --eval_max_samples 2000 \
    --log_every_steps 50 --save_every_epochs 12 --seed "$SEED" --bf16 \
    > "$RUN_ROOT/logs/${flow}_stage1.log" 2>&1
  [[ -s "$out/best.pt" ]]
  record "$flow" full stage1 complete "$out/best.pt"
}

run_stage2() {
  local flow=$1 data=$2 arm=$3 epochs=$4 out="$RUN_ROOT/tasks/$1/$3/stage2"
  local init=(--init_mode random)
  [[ "$arm" != full ]] || init=(--init_mode align --align_ckpt "$RUN_ROOT/tasks/$flow/stage1/best.pt")
  mkdir -p "$out"
  record "$flow" "$arm" stage2 running "epochs=$epochs"
  CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node=3 \
    projection_train/train_projection_bc.py \
    --data_path "$data" --llm_path "$MODEL" "${init[@]}" --output_dir "$out" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 --prompt_mode queue_wait_short \
    --batch_size 1 --grad_accum_steps 11 --epochs "$epochs" \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 --val_ratio 0.1 --split_strategy stratified \
    --num_workers 4 --target_mode mixed --soft_target_alpha 0.3 \
    --soft_label_temperature 0.7 --best_metric val_macro_f1 --best_mode max \
    --log_interval 20 --save_every "$epochs" --seed "$SEED" --bf16 \
    --checkpoint_slim --ddp_timeout_seconds 1800 \
    > "$RUN_ROOT/logs/${flow}_${arm}_stage2.log" 2>&1
  [[ -s "$out/best.pt" && -s "$out/metrics.jsonl" ]]
  record "$flow" "$arm" stage2 complete "$out/best.pt"
}

run_online() {
  local flow=$1 traffic=$2 arm=$3 dataset=jinan gpu=${GPUS%%,*}
  local out="$RUN_ROOT/tasks/$flow/$arm/online"
  [[ "$flow" != hangzhou* ]] || dataset=hangzhou
  mkdir -p "$out"
  record "$flow" "$arm" online running 'seed=2024'
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" projection_train/run_projection_online.py \
    --memo "stage1_deletion_fiveflow/$(basename "$RUN_ROOT")/$flow/$arm" \
    --projection_ckpt "$RUN_ROOT/tasks/$flow/$arm/stage2/best.pt" --llm_path "$MODEL" \
    --dataset "$dataset" --traffic_file "$traffic" --run_counts 3600 \
    --num_rounds 1 --min_action_time 30 --seed "$SEED" \
    --proj_name "stage1-deletion-$flow-$arm" --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --traffic_series_interval 2 --max_phase_override_gap 0 \
    --bf16 --deterministic --disable_snapshots --result_json "$out/result.json" \
    > "$RUN_ROOT/logs/${flow}_${arm}_online.log" 2>&1
  [[ -s "$out/result.json" ]]
  record "$flow" "$arm" online complete "$out/result.json"
}

write_summary() {
  "$PYTHON" - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
keys = {'ATT': 'test_avg_travel_time_over', 'AQL': 'test_avg_queue_len_over',
        'AWT': 'test_avg_waiting_time_over'}
summary = {'seed': 2024, 'flows': {}}
for flow in ('jinan1', 'jinan2', 'jinan3', 'hangzhou1', 'hangzhou2'):
    arms = ('full', 'without_stage1', 'budget_control') if flow == 'jinan1' else ('full', 'without_stage1')
    results = {}
    for arm in arms:
        path = root / 'tasks' / flow / arm / 'online' / 'result.json'
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        row = payload['results']
        if isinstance(row, list):
            row = row[-1]
        results[arm] = {name: float(row[key]) for name, key in keys.items()}
    if 'full' in results and 'without_stage1' in results:
        results['without_minus_full'] = {
            name: results['without_stage1'][name] - results['full'][name] for name in keys}
    summary['flows'][flow] = results
root.joinpath('summary.json').write_text(json.dumps(summary, indent=2) + '\n')
PY
}

for i in "${!FLOWS[@]}"; do
  flow=${FLOWS[$i]}
  data=${DATA[$i]}
  traffic=${TRAFFIC[$i]}
  run_stage1 "$flow" "$data"
  run_stage2 "$flow" "$data" full 12
  run_online "$flow" "$traffic" full
  run_stage2 "$flow" "$data" without_stage1 12
  run_online "$flow" "$traffic" without_stage1
  write_summary
done
run_stage2 jinan1 "${DATA[0]}" budget_control 24
run_online jinan1 "${TRAFFIC[0]}" budget_control
write_summary
record campaign all summary complete "$RUN_ROOT/summary.json"
