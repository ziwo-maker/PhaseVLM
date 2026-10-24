#!/usr/bin/env bash
# Full directed five-flow transfer matrix for paired Stage-I deletion arms.
set -Eeuo pipefail

REPO=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/LLMSTSCrecords/records/pipeline_runs/stage1_deletion_fiveflow_3gpu_20261001_065302}
RUN_ROOT=${RUN_ROOT:-/mnt/data/LLMSTSCrecords/records/online_tests/stage1_deletion_generalization_$(date +%Y%m%d_%H%M%S)}
GPUS=${GPUS:-1,2,3}
SEED=2024
FLOWS=(jinan1 jinan2 jinan3 hangzhou1 hangzhou2)
TRAFFIC=(anon_3_4_jinan_real.json anon_3_4_jinan_real_2000.json anon_3_4_jinan_real_2500.json anon_4_4_hangzhou_real.json anon_4_4_hangzhou_real_5816.json)

cd "$REPO"
[[ "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]] || { echo 'GPUS must contain three IDs' >&2; exit 2; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2; }
[[ -s "$SOURCE_RUN/summary.json" && -d "$MODEL" ]] || { echo 'Missing source results or model' >&2; exit 2; }
for source in "${FLOWS[@]}"; do
  for arm in full without_stage1; do
    [[ -s "$SOURCE_RUN/tasks/$source/$arm/stage2/best.pt" ]] || { echo "Missing checkpoint $source/$arm" >&2; exit 2; }
  done
done
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
STATUS="$RUN_ROOT/status.tsv"
printf 'time\tpair\tarm\tgpu\tstate\tdetail\n' > "$STATUS"
record() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" "$5" >> "$STATUS"; }

export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
"$PYTHON" - "$RUN_ROOT/manifest.json" "$SOURCE_RUN" "$MODEL" "$GPUS" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, source_run, model, gpus = sys.argv[1:]
root = pathlib.Path(source_run)
flows = [('jinan1', 'anon_3_4_jinan_real.json', 'jinan'),
         ('jinan2', 'anon_3_4_jinan_real_2000.json', 'jinan'),
         ('jinan3', 'anon_3_4_jinan_real_2500.json', 'jinan'),
         ('hangzhou1', 'anon_4_4_hangzhou_real.json', 'hangzhou'),
         ('hangzhou2', 'anon_4_4_hangzhou_real_5816.json', 'hangzhou')]
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
pairs = []
for source, source_traffic, source_city in flows:
    for target, target_traffic, target_city in flows:
        if source == target:
            continue
        pairs.append({'id': f'{source}_to_{target}', 'source': source,
                      'source_training_flow': source_traffic, 'target': target,
                      'traffic_file': target_traffic, 'dataset': target_city,
                      'scope': 'within_city' if source_city == target_city else 'cross_city',
                      'checkpoints': {
                          arm: {'path': str(root/'tasks'/source/arm/'stage2/best.pt'),
                                'sha256': sha(root/'tasks'/source/arm/'stage2/best.pt')}
                          for arm in ('full', 'without_stage1')}})
manifest = {
    'experiment': 'stage1_deletion_directed_fiveflow_generalization',
    'source_run': source_run, 'model_path': model, 'seed': 2024,
    'gpus': gpus.split(','), 'pair_count': len(pairs), 'evaluation_count': 2*len(pairs),
    'common_protocol': {'run_counts': 3600, 'num_rounds': 1,
        'min_action_time': 30, 'traffic_series_interval': 2,
        'prompt_mode': 'queue_wait_short', 'seq_len': 16, 'image_size': 224,
        'periodicity': 16, 'input_dim': 8, 'hidden_dim': 256,
        'prefix_tokens': 8, 'prefix_mode': 'prefix', 'max_length': 512,
        'bf16': True, 'deterministic': True, 'phase_overrides': 'disabled',
        'snapshots': False},
    'pairs': pairs,
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'evaluator_sha256': sha('projection_train/run_projection_online.py'),
}
pathlib.Path(out).write_text(json.dumps(manifest, indent=2) + '\n')
PY

notify() {
  local current done_count metric
  current=$(tail -n 1 "$STATUS" | tr '\t' ' ')
  done_count=$(rg -c $'\tcomplete\t' "$STATUS" || true)
  metric=$("$PYTHON" - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob('results/*/*/result.json'), reverse=True):
    result = json.loads(path.read_text())['results']
    if isinstance(result, list): result = result[-1]
    print(f"{path.parts[-3]}/{path.parts[-2]} ATT={result.get('test_avg_travel_time_over')} AWT={result.get('test_avg_waiting_time_over')}")
    break
PY
  )
  timeout --foreground 15s "$PYTHON" scripts/feishu_push.py \
    "Stage I transfer ablation: ${done_count:-0}/40 completed; $current; latest=${metric:-pending}; run=$RUN_ROOT" \
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
  record campaign all - "$([[ $rc -eq 0 ]] && echo complete || echo failed)" "exit=$rc"
  notify
  exit "$rc"
}
trap finish EXIT
( while sleep 2400; do notify; done ) &
notifier_pid=$!

run_one() {
  local source=$1 target=$2 target_idx=$3 scope=$4 arm=$5 gpu=$6
  local pair="${source}_to_${target}" out="$RUN_ROOT/results/${source}_to_${target}/$arm"
  local dataset=jinan
  [[ "$target" != hangzhou* ]] || dataset=hangzhou
  mkdir -p "$out"
  record "$pair" "$arm" "$gpu" running "$scope"
  if timeout --foreground 1800s env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    projection_train/run_projection_online.py \
    --memo "stage1_deletion_generalization/$(basename "$RUN_ROOT")/$pair/$arm" \
    --projection_ckpt "$SOURCE_RUN/tasks/$source/$arm/stage2/best.pt" --llm_path "$MODEL" \
    --dataset "$dataset" --traffic_file "${TRAFFIC[$target_idx]}" \
    --run_counts 3600 --num_rounds 1 --min_action_time 30 --seed "$SEED" \
    --proj_name "stage1-transfer-$pair-$arm" --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --traffic_series_interval 2 --max_phase_override_gap 0 --bf16 --deterministic \
    --disable_snapshots --result_json "$out/result.json" \
    > "$RUN_ROOT/logs/${pair}_${arm}.log" 2>&1; then
    [[ -s "$out/result.json" ]] || { record "$pair" "$arm" "$gpu" failed 'result_missing'; return 1; }
    record "$pair" "$arm" "$gpu" complete "$out/result.json"
  else
    local rc=$?
    record "$pair" "$arm" "$gpu" failed "exit=$rc"
    return "$rc"
  fi
}

worker() {
  local worker_idx=$1 gpu=${GPU_LIST[$1]} pair_idx=0 source target source_idx target_idx scope failures=0
  for source_idx in "${!FLOWS[@]}"; do
    source=${FLOWS[$source_idx]}
    for target_idx in "${!FLOWS[@]}"; do
      [[ "$source_idx" != "$target_idx" ]] || continue
      target=${FLOWS[$target_idx]}
      if (( pair_idx % 3 == worker_idx )); then
        scope=within_city
        if [[ "$source" == jinan* && "$target" == hangzhou* ]] ||
           [[ "$source" == hangzhou* && "$target" == jinan* ]]; then
          scope=cross_city
        fi
        run_one "$source" "$target" "$target_idx" "$scope" full "$gpu" || failures=1
        run_one "$source" "$target" "$target_idx" "$scope" without_stage1 "$gpu" || failures=1
      fi
      ((pair_idx+=1))
    done
  done
  return "$failures"
}

worker 0 & p0=$!
worker 1 & p1=$!
worker 2 & p2=$!
printf 'pid=%s\nworkers=%s,%s,%s\nrun_root=%s\n' "$$" "$p0" "$p1" "$p2" "$RUN_ROOT" > "$RUN_ROOT/launcher.status"
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
wait "$p2" || failed=1
"$PYTHON" scripts/summarize_stage1_deletion_generalization.py "$RUN_ROOT"
[[ "$failed" -eq 0 ]] && [[ $(rg -c $'\tcomplete\t' "$STATUS") -eq 40 ]]
