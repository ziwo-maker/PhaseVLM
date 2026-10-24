#!/usr/bin/env bash
# Online evaluation: Jinan synthetic 24h traffic, existing recordlocal Stage-II best checkpoints.
# Run detached: nohup bash recordlocal/run_jinan24h_best_online.sh > <launcher_log> 2>&1 < /dev/null &
set -Eeuo pipefail
REPO=/home/code/LLMSTSC
cd "$REPO"
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
SOURCE_RUN=stage1_convergence_jinan1_s2024_20261009_202803
SOURCE_ROOT="$REPO/recordlocal/runs/$SOURCE_RUN"
TRAFFIC=anon_3_4_jinan_synthetic_24h_6000.json
FLOW_PATH="$REPO/data/Jinan/3_4/$TRAFFIC"
SEED=2024
ARMS=(full without_stage1 shuffled_alignment)
GPU_IDS=(0 1 2)
RUN_ID=${RUN_ID:-jinan24h_stage1_best_s2024_$(date +%Y%m%d_%H%M%S)}
ROOT="$REPO/recordlocal/runs/$RUN_ID"
SECONDS_LIMIT=14400
for path in "$PYTHON" "$MODEL" "$FLOW_PATH"; do
  [[ -e "$path" ]] || { echo "Missing dependency: $path" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 2; }
for arm in "${ARMS[@]}"; do
  [[ -s "$SOURCE_ROOT/$arm/stage2/best.pt" ]] || { echo "Missing: $SOURCE_ROOT/$arm/stage2/best.pt" >&2; exit 2; }
done
timeout --foreground 10s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >/dev/null
for gpu in "${GPU_IDS[@]}"; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
  [[ -n "$used" && "$used" -le 4096 ]] || { echo "GPU $gpu busy: $used MiB" >&2; exit 3; }
done
if [[ "${1:-}" == "--preflight-only" ]]; then
  echo "PREFLIGHT_OK traffic=$TRAFFIC duration=86400s arms=${ARMS[*]} GPUs=${GPU_IDS[*]} seed=$SEED"
  exit 0
fi
[[ ! -e "$ROOT" ]] || { echo "Refusing to overwrite $ROOT" >&2; exit 4; }
mkdir -p "$ROOT/logs"
printf 'time\tstage\tarm\tstatus\tdetail\n' > "$ROOT/status.tsv"
printf '%s\n' "$$" > "$ROOT/launcher.pid"
cp "$REPO/recordlocal/run_jinan24h_best_online.sh" "$ROOT/launcher.snapshot.sh"
note() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$4" >> "$ROOT/status.tsv"; }
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" - "$ROOT" "$SOURCE_ROOT" "$FLOW_PATH" "$MODEL" <<'PY'
import json,sys,hashlib,pathlib,datetime
root,source,traffic,model=map(pathlib.Path,sys.argv[1:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
manifest={
 'experiment':'Jinan synthetic 24h online tests of Stage I ablation best-validation checkpoints',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_experiment':str(source),'source_manifest':str(source/'manifest.json'),
 'dataset':'jinan','traffic_file':traffic.name,'traffic_sha256':sha(traffic),
 'traffic_kind':'synthetic 24h, not Jinan-1 real training flow','run_counts_seconds':86400,
 'train_flow':'anon_3_4_jinan_real.json','seed':2024,'model_path':str(model),
 'num_rounds':1,'min_action_time':30,'prompt_mode':'queue_wait_short',
 'representation_mode':'ts2i','sequence_length':16,'image_size':224,'periodicity':16,
 'input_dim':8,'hidden_dim':256,'prefix_tokens':8,'max_length':512,
 'bf16':True,'deterministic':True,'max_phase_override_gap':0,
 'checkpoint_selection':'source run minimum offline validation top1 cost regret',
 'arms':{a:{'gpu':i,'checkpoint':str(source/a/'stage2'/'best.pt'),
             'sha256':sha(source/a/'stage2'/'best.pt')}
         for i,a in enumerate(('full','without_stage1','shuffled_alignment'))}
}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
note prepare all complete "manifest=$ROOT/manifest.json"
run_one() {
  local arm=$1 gpu=$2 start=$SECONDS
  local result="$ROOT/$arm/best/result.json"
  local log="$ROOT/logs/online_${arm}_best.log"
  mkdir -p "$(dirname "$result")"
  note online "$arm" running "gpu=$gpu,flow=$TRAFFIC,run_counts=86400,ckpt=best.pt"
  if timeout --foreground "${SECONDS_LIMIT}s" env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" projection_train/run_projection_online.py \
      --memo "../recordlocal/runs/$RUN_ID/online_env/$arm/best" \
      --projection_ckpt "$SOURCE_ROOT/$arm/stage2/best.pt" \
      --llm_path "$MODEL" --dataset jinan --traffic_file "$TRAFFIC" \
      --run_counts 86400 --num_rounds 1 --min_action_time 30 --seed "$SEED" \
      --proj_name "StageI-Jinan24h-${arm}-best" \
      --prompt_mode queue_wait_short --seq_len 16 --image_size 224 \
      --periodicity 16 --input_dim 8 --hidden_dim 256 \
      --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
      --traffic_series_interval 2 --max_phase_override_gap 0 \
      --bf16 --deterministic --disable_snapshots \
      --result_json "$result" > "$log" 2>&1; then
    if [[ -s "$result" ]]; then
      note online "$arm" complete "gpu=$gpu,wall_seconds=$((SECONDS-start)),result=$result"
      return 0
    fi
    note online "$arm" failed "missing result file; log=$log"
    return 1
  else
    local code=$?
    note online "$arm" failed "gpu=$gpu,exit_code=$code,log=$log"
    return "$code"
  fi
}
pids=()
for i in "${!ARMS[@]}"; do
  run_one "${ARMS[$i]}" "${GPU_IDS[$i]}" &
  pids+=("$!")
done
failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failures=$((failures+1)); fi
done
"$PYTHON" - "$ROOT" <<'PY'
import sys,pathlib,json,csv
root=pathlib.Path(sys.argv[1])
names={'ATT':'test_avg_travel_time_over','AQL':'test_avg_queue_len_over',
       'AWT':'test_avg_waiting_time_over','wait_p95':'test_vehicle_wait_p95_over',
       'wait_max':'test_vehicle_wait_max_over','reward':'test_reward_over',
       'queued_vehicle_count':'test_queuing_vehicle_num_over'}
rows=[]
for a in ('full','without_stage1','shuffled_alignment'):
    p=root/a/'best'/'result.json'
    if not p.exists():continue
    d=json.loads(p.read_text())
    r=d['results']
    if isinstance(r,list):r=r[-1]
    rows.append({'arm':a,'seed':d.get('args',{}).get('seed'),
                **{k:r.get(v) for k,v in names.items()}})
with (root/'online_comparison.csv').open('w',newline='') as out:
    w=csv.DictWriter(out,fieldnames=['arm','seed',*names]);w.writeheader();w.writerows(rows)
(root/'summary.json').write_text(json.dumps({'results':rows,'finished_runs':len(rows),'expected_runs':3},indent=2)+'\n')
PY
if [[ "$failures" -ne 0 ]]; then
  note all all failed "failed_jobs=$failures; inspect logs"
  exit 1
fi
note all all completed "results=$ROOT/online_comparison.csv"
echo "COMPLETED: $ROOT"
