#!/usr/bin/env bash
# Paired multi-GPU generalization evaluation for the Jinan-1 Stage I deletion.
set -Eeuo pipefail

REPO_ROOT=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/data/LLMSTSCrecords/records/pipeline_runs/jinan1_stage1_deletion_seed2024_20260930_01}
RUN_ROOT=${RUN_ROOT:-/mnt/data/LLMSTSCrecords/records/online_tests/jinan1_stage1_deletion_generalization_$(date +%Y%m%d_%H%M%S)}
MODEL=/mnt/data/Qwen3-VL-8B-Instruct
GPUS=(1 2 3)
SEEDS=(2024 42 123)
ARMS=(full_15 random_15)
START_SECONDS=$(date +%s)
NOTIFIER_PID=
STATUS=$RUN_ROOT/status.tsv

timestamp() { date --iso-8601=seconds; }
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" "$3" "$4" >> "$STATUS"; }
notify() {
  timeout --foreground 15s "$PYTHON" "$REPO_ROOT/scripts/feishu_push.py" "$1" > /dev/null 2>&1 || true
}
progress() {
  local complete total
  complete=$(rg -c $'\tcompleted\t' "$STATUS" 2>/dev/null || true)
  complete=${complete:-0}
  total=18
  printf 'Jinan-1 Stage I generalization: %s/%s online tests completed; run=%s' "$complete" "$total" "$RUN_ROOT"
}
cleanup() {
  local code=$?
  trap - EXIT
  if [[ -n "$NOTIFIER_PID" ]]; then
    kill "$NOTIFIER_PID" 2>/dev/null || true
    wait "$NOTIFIER_PID" 2>/dev/null || true
  fi
  record campaign all all "exit=$code"
  if (( $(date +%s) - START_SECONDS >= 3600 )); then notify "$(progress); exit=$code"; fi
  exit "$code"
}

for path in "$PYTHON" "$MODEL" \
  "$SOURCE_ROOT/arms/full_15/checkpoints/best.pt" \
  "$SOURCE_ROOT/arms/random_15/checkpoints/best.pt" \
  "$REPO_ROOT/data/Jinan/3_4/anon_3_4_jinan_synthetic_24000_60min.json" \
  "$REPO_ROOT/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json" \
  "$REPO_ROOT/data/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.json"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
printf 'time\tstate\ttarget\tarm\tdetail\n' > "$STATUS"
trap cleanup EXIT
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

"$PYTHON" - "$RUN_ROOT/manifest.json" "$SOURCE_ROOT" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, source = map(pathlib.Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
targets = [
    ('jinan_unseen_synthetic', 'jinan', 'anon_3_4_jinan_synthetic_24000_60min.json', 1, 'same_roadnet_unseen_synthetic_demand'),
    ('hangzhou_1', 'hangzhou', 'anon_4_4_hangzhou_real.json', 2, 'cross_city_real'),
    ('hangzhou_2', 'hangzhou', 'anon_4_4_hangzhou_real_5816.json', 3, 'cross_city_real'),
]
root = pathlib.Path('/home/code/LLMSTSC')
payload = {
    'purpose': 'Stage I deletion generalization, single training seed 2024',
    'source_run': str(source), 'source_training_seed': 2024,
    'source_training_flows': ['anon_3_4_jinan_real.json', 'anon_3_4_jinan_real_2000.json', 'anon_3_4_jinan_real_2500.json'],
    'model': '/mnt/data/Qwen3-VL-8B-Instruct',
    'arms': {arm: {'checkpoint': str(source / 'arms' / arm / 'checkpoints' / 'best.pt'),
                   'sha256': sha(source / 'arms' / arm / 'checkpoints' / 'best.pt')}
             for arm in ('full_15', 'random_15')},
    'targets': [{'id': id, 'dataset': dataset, 'flow': flow, 'gpu': gpu, 'scope': scope,
                 'sha256': sha(root / 'data' / ('Jinan/3_4' if dataset == 'jinan' else 'Hangzhou/4_4') / flow)}
                for id, dataset, flow, gpu, scope in targets],
    'evaluation_seeds': [2024, 42, 123], 'run_counts': 3600, 'num_rounds': 1,
    'min_action_time': 30, 'traffic_series_interval': 2,
    'seq_len': 16, 'image_size': 224, 'periodicity': 16, 'input_dim': 8,
    'hidden_dim': 64, 'prefix_tokens': 8, 'max_length': 64,
    'prompt_mode': 'queue_wait_short', 'bf16': True, 'deterministic': True,
    'phase_overrides': 'disabled', 'snapshots': 'disabled',
    'evaluator_sha256': sha(root / 'projection_train/run_projection_online.py'),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
}
out.write_text(json.dumps(payload, indent=2) + '\n')
PY

(
  while sleep 2400; do notify "$(progress)"; done
) &
NOTIFIER_PID=$!

run_target() {
  local target=$1 dataset=$2 flow=$3 gpu=$4
  local arm seed ckpt out result log rc
  for arm in "${ARMS[@]}"; do
    ckpt=$SOURCE_ROOT/arms/$arm/checkpoints/best.pt
    for seed in "${SEEDS[@]}"; do
      out=$RUN_ROOT/results/$target/$arm/seed_$seed
      result=$out/result.json
      log=$RUN_ROOT/logs/${target}_${arm}_seed_${seed}_gpu${gpu}.log
      mkdir -p "$out"
      record running "$target" "$arm" "seed=$seed gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" projection_train/run_projection_online.py \
        --memo "online_tests/jinan1_stage1_deletion_generalization/$(basename "$RUN_ROOT")/$target/$arm/seed_$seed" \
        --projection_ckpt "$ckpt" --llm_path "$MODEL" \
        --dataset "$dataset" --traffic_file "$flow" \
        --run_counts 3600 --num_rounds 1 --seed "$seed" \
        --proj_name "Jinan1-stage1-generalization-$target-$arm" \
        --prompt_mode queue_wait_short \
        --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
        --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
        --min_action_time 30 --traffic_series_interval 2 \
        --max_phase_override_gap 0 --bf16 --deterministic --disable_snapshots \
        --result_json "$result" > "$log" 2>&1 || {
          rc=$?
          record failed "$target" "$arm" "seed=$seed exit=$rc log=$log"
          return "$rc"
        }
      [[ -s "$result" ]] || { record failed "$target" "$arm" "seed=$seed missing_result"; return 1; }
      record completed "$target" "$arm" "seed=$seed result=$result"
    done
  done
}

run_target jinan_unseen_synthetic jinan anon_3_4_jinan_synthetic_24000_60min.json 1 & pid_jinan=$!
run_target hangzhou_1 hangzhou anon_4_4_hangzhou_real.json 2 & pid_h1=$!
run_target hangzhou_2 hangzhou anon_4_4_hangzhou_real_5816.json 3 & pid_h2=$!
failed=0
for pid in "$pid_jinan" "$pid_h1" "$pid_h2"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
keys = {'ATT': 'test_avg_travel_time_over', 'AQL': 'test_avg_queue_len_over', 'AWT': 'test_avg_waiting_time_over'}
summary = {}
for target in ('jinan_unseen_synthetic', 'hangzhou_1', 'hangzhou_2'):
    arms = {}
    for arm in ('full_15', 'random_15'):
        rows = {}
        for seed in (2024, 42, 123):
            payload = json.loads((root / 'results' / target / arm / f'seed_{seed}' / 'result.json').read_text())
            assert payload['args']['seed'] == seed
            result = payload['results']
            if isinstance(result, list): result = result[-1]
            rows[str(seed)] = {name: float(result[key]) for name, key in keys.items()}
        arms[arm] = {'by_seed': rows, 'mean': {name: statistics.mean(row[name] for row in rows.values()) for name in keys}}
    summary[target] = {'arms': arms, 'delta_wo_minus_full': {
        name: arms['random_15']['mean'][name] - arms['full_15']['mean'][name] for name in keys}}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: {'full': v['arms']['full_15']['mean'],
                      'without_stage1': v['arms']['random_15']['mean'],
                      'delta': v['delta_wo_minus_full']} for k, v in summary.items()}, indent=2))
PY
record completed summary all "path=$RUN_ROOT/summary.json"
