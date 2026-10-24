#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/code/LLMSTSC
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
MODEL=${MODEL:-/mnt/data/Qwen3-VL-8B-Instruct}
SOURCE_ROOT=$REPO/recordlocal/runs/stage1_data_efficiency_s2024_20261010_200623
TRAFFIC=anon_3_4_jinan_synthetic_24h_6000.json
FLOW=$REPO/data/Jinan/3_4/$TRAFFIC
OUT=$REPO/recordlocal/runs/stage1_final_performance_4gpu_$(date +%Y%m%d_%H%M%S)
SEED=2024
mkdir -p $OUT/logs

# Stage I effectiveness on final traffic performance:
# compare identical Stage-II online evaluation with and without Stage I.
# two independent evaluations run on 4 GPUs.

run_eval(){
  ARM=$1
  GPU=$2
  if [ "$ARM" = "full" ]; then
    CKPT=$SOURCE_ROOT/100pct/full/best.pt
  else
    CKPT=$SOURCE_ROOT/100pct/without_stage1/best.pt
  fi
  RESULT=$OUT/$ARM/result.json
  MEMO=stage1_final_performance_4gpu/$ARM
  WORKDIR=$REPO/records/$MEMO
  MODEL_DIR=$REPO/model/$MEMO
  # online runner requires fresh work/model directories; remove stale metadata from previous failed launches
  rm -rf $WORKDIR $MODEL_DIR
  mkdir -p $OUT/$ARM
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON projection_train/run_projection_online.py \
    --memo $MEMO \
    --projection_ckpt $CKPT \
    --llm_path $MODEL \
    --dataset jinan \
    --traffic_file $TRAFFIC \
    --run_counts 86400 \
    --num_rounds 1 \
    --min_action_time 30 \
    --seed $SEED \
    --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 --periodicity 16 \
    --input_dim 8 --hidden_dim 256 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 512 \
    --traffic_series_interval 2 \
    --max_phase_override_gap 0 \
    --bf16 --deterministic --disable_snapshots \
    --result_json $RESULT > $OUT/logs/${ARM}.log 2>&1
}

run_eval full 0,1 &
run_eval without_stage1 2,3 &
wait

$PYTHON - "$OUT" <<'PY'
import json, pathlib, csv, sys
root=pathlib.Path(sys.argv[1])
rows=[]
for arm in ['full','without_stage1']:
    p=root/arm/'result.json'
    if p.exists():
        d=json.loads(p.read_text())
        rows.append({'arm':arm,'raw':d})
(root/'summary.json').write_text(json.dumps(rows,indent=2))
PY

echo completed:$OUT
