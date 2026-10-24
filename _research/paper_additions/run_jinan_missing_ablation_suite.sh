#!/usr/bin/env bash
# Run the missing main-paper ablations on Jinan with Qwen3-VL-8B.
#
# Default launch:
#   nohup bash paper_additions/run_jinan_missing_ablation_suite.sh \
#     > paper_additions/run_jinan_missing_ablation_suite.nohup.log 2>&1 &
#
# The script is intentionally self-contained: it creates derived control data,
# trains each missing Stage-2 arm for multiple training seeds, evaluates every
# finished checkpoint on multiple online simulator seeds, and writes aggregate
# JSON/CSV/Markdown under RUN_ROOT.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
TORCHRUN="${TORCHRUN:-/home/sysu/anaconda3/envs/hnr5090/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3-VL-8B-Instruct}"
RUN_ID="${RUN_ID:-jinan_missing_ablation_suite_qwen3vl8b_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/records/pipeline_runs/$RUN_ID}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS// /,}"
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
TRAIN_NPROC="${TRAIN_NPROC:-${#TRAIN_GPU_ARRAY[@]}}"

EVAL_GPUS="${EVAL_GPUS:-$TRAIN_GPUS}"
EVAL_GPUS="${EVAL_GPUS// /,}"
IFS=',' read -r -a EVAL_GPU_ARRAY <<< "$EVAL_GPUS"

TRAINING_SEEDS_RAW="${TRAINING_SEEDS:-2024 42 123}"
ONLINE_SEEDS_RAW="${ONLINE_SEEDS:-2024 42 123 7}"
read -r -a TRAINING_SEEDS <<< "$TRAINING_SEEDS_RAW"
read -r -a ONLINE_SEEDS <<< "$ONLINE_SEEDS_RAW"

STAGE1_CKPT="${STAGE1_CKPT:-}"
STAGE1_TRAIN_IF_MISSING="${STAGE1_TRAIN_IF_MISSING:-1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-15}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-15}"
ONLINE_RUN_COUNTS="${ONLINE_RUN_COUNTS:-3600}"

STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-2}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}"
STAGE2_ACTION_EVAL_CHUNK_SIZE="${STAGE2_ACTION_EVAL_CHUNK_SIZE:-0}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"

DATA_PATHS=(
  "${JINAN_DATA_MAIN:-$REPO_ROOT/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl}"
  "${JINAN_DATA_2000:-$REPO_ROOT/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl}"
  "${JINAN_DATA_2500:-$REPO_ROOT/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl}"
)
TRAFFIC_FILE="${TRAFFIC_FILE:-anon_3_4_jinan_real.json}"

LOG_DIR="$RUN_ROOT/logs"
STATUS_TSV="$RUN_ROOT/status.tsv"
DERIVED_DIR="$RUN_ROOT/derived_data"
STAGE1_DIR="$RUN_ROOT/stage1"
RANDOM_STAGE1_DIR="$RUN_ROOT/stage1_random_projector"
RESULTS_DIR="$RUN_ROOT/online_eval"
AGG_JSON="$RUN_ROOT/aggregate.json"
AGG_CSV="$RUN_ROOT/aggregate.csv"
REPORT_MD="$RUN_ROOT/RESULTS.md"
PROVENANCE="$RUN_ROOT/provenance.env"

# What is already done in the local result tree and should not be relaunched by
# this Jinan-focused suite.
VISUAL_SIGNAL_RESULT="${VISUAL_SIGNAL_RESULT:-$REPO_ROOT/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/aggregate.json}"

timestamp() { date --iso-8601=seconds; }

log_msg() {
  mkdir -p "$LOG_DIR"
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_DIR/launcher.log"
}

record_status() {
  mkdir -p "$RUN_ROOT"
  if [[ ! -s "$STATUS_TSV" ]]; then
    printf 'time\tstage\tarm\ttraining_seed\tonline_seed\tstatus\tpath\tmessage\n' > "$STATUS_TSV"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(timestamp)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$STATUS_TSV"
}

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
}

preflight() {
  require_path "$PYTHON"
  require_path "$TORCHRUN"
  require_path "$MODEL_PATH"
  require_path "$REPO_ROOT/data/Jinan/3_4/$TRAFFIC_FILE"
  local data_path
  for data_path in "${DATA_PATHS[@]}"; do
    require_path "$data_path"
  done
  [[ ${#TRAIN_GPU_ARRAY[@]} -gt 0 ]] || { echo "TRAIN_GPUS is empty" >&2; exit 2; }
  [[ ${#EVAL_GPU_ARRAY[@]} -gt 0 ]] || { echo "EVAL_GPUS is empty" >&2; exit 2; }
  [[ "$TRAIN_NPROC" -eq "${#TRAIN_GPU_ARRAY[@]}" ]] || {
    echo "TRAIN_NPROC=$TRAIN_NPROC does not match TRAIN_GPUS=$TRAIN_GPUS" >&2
    exit 2
  }
  timeout --foreground 30s "$PYTHON" - "$MODEL_PATH" <<'PY'
import sys
from transformers import AutoConfig
p = sys.argv[1]
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
if getattr(cfg, "vision_config", None) is None:
    raise SystemExit(f"{p} is not a VLM config")
print(f"[model] path={p} model_type={getattr(cfg, 'model_type', '')}")
PY
}

write_provenance() {
  mkdir -p "$RUN_ROOT" "$LOG_DIR" "$DERIVED_DIR" "$RESULTS_DIR"
  {
    printf 'schema=jinan_missing_ablation_suite_v1\n'
    printf 'created_at=%s\n' "$(timestamp)"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'python=%s\n' "$PYTHON"
    printf 'torchrun=%s\n' "$TORCHRUN"
    printf 'model_path=%s\n' "$MODEL_PATH"
    printf 'train_gpus=%s\n' "$TRAIN_GPUS"
    printf 'train_nproc=%s\n' "$TRAIN_NPROC"
    printf 'eval_gpus=%s\n' "$EVAL_GPUS"
    printf 'training_seeds=%s\n' "${TRAINING_SEEDS[*]}"
    printf 'online_seeds=%s\n' "${ONLINE_SEEDS[*]}"
    printf 'traffic_file=%s\n' "$TRAFFIC_FILE"
    printf 'data_paths=%s\n' "${DATA_PATHS[*]}"
    printf 'visual_signal_result=%s\n' "$VISUAL_SIGNAL_RESULT"
    printf 'stage1_checkpoint_input=%s\n' "$STAGE1_CKPT"
    printf 'stage1_train_if_missing=%s\n' "$STAGE1_TRAIN_IF_MISSING"
    printf 'stage1_epochs=%s\n' "$STAGE1_EPOCHS"
    printf 'stage2_epochs=%s\n' "$STAGE2_EPOCHS"
    printf 'online_run_counts=%s\n' "$ONLINE_RUN_COUNTS"
    printf 'git_head=%s\n' "$(timeout --foreground 10s git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'git_diff_sha256=%s\n' "$({ timeout --foreground 30s git -C "$REPO_ROOT" diff --no-ext-diff --binary 2>/dev/null || true; } | sha256sum | awk '{print $1}')"
    local path
    for path in "${DATA_PATHS[@]}" "$REPO_ROOT/projection_train/train_projection_bc.py" "$REPO_ROOT/projection_train/run_projection_online.py" "$REPO_ROOT/train_align.py"; do
      if [[ -s "$path" ]]; then
        printf 'sha256:%s=%s\n' "$path" "$(sha256sum "$path" | awk '{print $1}')"
      fi
    done
  } > "$PROVENANCE"
}

train_stage1_if_needed() {
  if [[ -n "$STAGE1_CKPT" ]]; then
    require_path "$STAGE1_CKPT"
    mkdir -p "$STAGE1_DIR"
    cp -p "$STAGE1_CKPT" "$STAGE1_DIR/best.pt"
    printf 'status=reused\nsource_checkpoint=%s\ncopied_at=%s\n' "$STAGE1_CKPT" "$(timestamp)" > "$STAGE1_DIR/reused_stage1.info"
    log_msg "Reused Stage-1 checkpoint: $STAGE1_CKPT"
    return 0
  fi

  if [[ -s "$STAGE1_DIR/best.pt" ]]; then
    log_msg "Stage-1 already exists: $STAGE1_DIR/best.pt"
    return 0
  fi

  if [[ "$STAGE1_TRAIN_IF_MISSING" != "1" ]]; then
    echo "No STAGE1_CKPT provided and STAGE1_TRAIN_IF_MISSING=$STAGE1_TRAIN_IF_MISSING" >&2
    exit 2
  fi

  log_msg "Training shared Stage-1 alignment checkpoint on Jinan data."
  record_status stage1 shared shared none running "$STAGE1_DIR" "training"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" "$REPO_ROOT/train_align.py" \
    --data_path "${DATA_PATHS[@]}" \
    --model_path "$MODEL_PATH" \
    --output_dir "$STAGE1_DIR" \
    --decision_only \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix \
    --temperature 0.05 \
    --batch_size "$STAGE1_BATCH_SIZE" --grad_accum_steps "$STAGE1_GRAD_ACCUM_STEPS" \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --epochs "$STAGE1_EPOCHS" --num_workers "$STAGE1_NUM_WORKERS" --val_ratio 0.1 \
    --eval_max_samples 1000 --log_every_steps 20 --save_every_epochs 5 \
    --seed 2024 --bf16 \
    > "$LOG_DIR/stage1.log" 2>&1
  [[ -s "$STAGE1_DIR/best.pt" ]] || { echo "Stage-1 did not create $STAGE1_DIR/best.pt" >&2; exit 1; }
  record_status stage1 shared shared none completed "$STAGE1_DIR/best.pt" "ok"
}

make_random_stage1() {
  local src="$STAGE1_DIR/best.pt"
  local dst="$RANDOM_STAGE1_DIR/best.pt"
  [[ -s "$dst" ]] && { log_msg "Random-projector Stage-1 already exists: $dst"; return 0; }
  require_path "$src"
  mkdir -p "$RANDOM_STAGE1_DIR"
  log_msg "Creating random-projector Stage-1 checkpoint from the aligned checkpoint structure."
  timeout --foreground 120s "$PYTHON" - "$src" "$dst" <<'PY'
import copy
import json
import sys
from pathlib import Path

import torch

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = 99173
torch.manual_seed(seed)
ckpt = torch.load(src, map_location="cpu", weights_only=False)
out = copy.deepcopy(ckpt)

def randomize_state_dict(sd):
    randomized = type(sd)()
    for key, value in sd.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                scale = float(value.detach().std().item())
                if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
                    scale = 0.02
                randomized[key] = torch.randn_like(value) * scale
            else:
                randomized[key] = torch.zeros_like(value)
        else:
            randomized[key] = copy.deepcopy(value)
    return randomized

for section in ("ts_projector", "vision_projector"):
    if section not in out or not isinstance(out[section], dict):
        raise SystemExit(f"missing checkpoint section: {section}")
    out[section] = randomize_state_dict(out[section])
out["random_projector_ablation"] = {
    "source_checkpoint": str(src),
    "seed": seed,
    "description": "Projector tensors reinitialized; args/logit_scale preserved for architecture compatibility.",
}
torch.save(out, dst)
(dst.parent / "random_projector.info.json").write_text(json.dumps(out["random_projector_ablation"], indent=2) + "\n")
print(dst)
PY
  record_status stage1 random_projector shared none completed "$dst" "randomized"
}

derive_control_data() {
  local out_real="$DERIVED_DIR/real_targets.jsonl"
  local out_shuffled="$DERIVED_DIR/shuffled_targets_seed991.jsonl"
  [[ -s "$out_real" && -s "$out_shuffled" ]] && { log_msg "Derived control data already exists."; return 0; }

  log_msg "Creating same-state real/shuffled-target control JSONL files."
  timeout --foreground 120s "$PYTHON" - "$out_real" "$out_shuffled" "${DATA_PATHS[@]}" <<'PY'
import json
import random
import sys
from collections import Counter
from pathlib import Path

out_real = Path(sys.argv[1])
out_shuf = Path(sys.argv[2])
inputs = [Path(p) for p in sys.argv[3:]]
seed = 991
rng = random.Random(seed)
records = []
for path in inputs:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("expert_action"):
                record.setdefault("meta", {})
                if isinstance(record["meta"], dict):
                    record["meta"]["control_source_path"] = str(path)
                records.append(record)

labels = [str(r.get("expert_action")) for r in records]
shuffled = labels[:]
if len(shuffled) > 1:
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(labels, shuffled)):
            break
    else:
        rng.shuffle(shuffled)

out_real.parent.mkdir(parents=True, exist_ok=True)
with out_real.open("w", encoding="utf-8") as handle:
    for record in records:
        record = dict(record)
        record.setdefault("meta", {})
        if isinstance(record["meta"], dict):
            record["meta"]["label_control"] = "real_targets_same_state_pool"
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

with out_shuf.open("w", encoding="utf-8") as handle:
    for record, label in zip(records, shuffled):
        record = dict(record)
        original = record.get("expert_action")
        record["expert_action"] = label
        record.setdefault("meta", {})
        if isinstance(record["meta"], dict):
            record["meta"].update({
                "label_control": "shuffled_targets_same_state_pool",
                "original_expert_action": original,
                "permutation_seed": seed,
            })
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

summary = {
    "schema": "jinan_same_state_label_control_v1",
    "inputs": [str(p) for p in inputs],
    "rows": len(records),
    "seed": seed,
    "real_histogram": Counter(labels),
    "shuffled_histogram": Counter(shuffled),
    "fixed_points": sum(a == b for a, b in zip(labels, shuffled)),
}
(out_real.parent / "control_data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY
}

arm_data_paths() {
  local arm="$1"
  case "$arm" in
    stage1_random_projector|hard_bc|hard_plus_soft|hard_soft_ranking|full_loss)
      printf '%s\n' "${DATA_PATHS[@]}"
      ;;
    control_shuffled_targets)
      printf '%s\n' "$DERIVED_DIR/shuffled_targets_seed991.jsonl"
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      return 2
      ;;
  esac
}

arm_stage1_ckpt() {
  local arm="$1"
  case "$arm" in
    stage1_random_projector)
      printf '%s\n' "$RANDOM_STAGE1_DIR/best.pt"
      ;;
    *)
      printf '%s\n' "$STAGE1_DIR/best.pt"
      ;;
  esac
}

arm_loss_args() {
  local arm="$1"
  case "$arm" in
    hard_bc|control_shuffled_targets)
      printf '%s\n' "--target_mode hard --soft_target_alpha 0 --ranking_loss_weight 0 --base_policy_kl_weight 0"
      ;;
    hard_plus_soft)
      printf '%s\n' "--target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight 0 --base_policy_kl_weight 0"
      ;;
    hard_soft_ranking)
      printf '%s\n' "--target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0"
      ;;
    full_loss|stage1_random_projector)
      printf '%s\n' "--target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.7 --ranking_loss_weight 0.05 --ranking_min_cost_gap 0.02 --base_policy_kl_weight 0.02 --base_policy_prior_mass 0.70 --base_policy_kl_mode match_or_low_weight"
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      return 2
      ;;
  esac
}

stage2_train_complete() {
  local ckpt_dir="$1"
  local metrics="$ckpt_dir/metrics.jsonl"
  [[ -s "$ckpt_dir/best.pt" && -s "$metrics" ]] || return 1
  timeout --foreground 30s "$PYTHON" - "$metrics" "$STAGE2_EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

metrics = Path(sys.argv[1])
expected_epochs = int(sys.argv[2])
max_epoch = 0
with metrics.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(1)
        try:
            max_epoch = max(max_epoch, int(payload.get("epoch", 0)))
        except (TypeError, ValueError):
            raise SystemExit(1)
raise SystemExit(0 if max_epoch >= expected_epochs else 1)
PY
}

train_arm_seed() {
  local arm="$1"
  local seed="$2"
  local arm_root="$RUN_ROOT/stage2/$arm/train_seed_$seed"
  local ckpt_dir="$arm_root/checkpoints"
  local log_path="$LOG_DIR/train_${arm}_seed_${seed}.log"
  local data_args=()
  local loss_args_string
  local align_ckpt
  local data_path

  if stage2_train_complete "$ckpt_dir"; then
    log_msg "Reuse Stage-2 checkpoint: arm=$arm seed=$seed"
    record_status train "$arm" "$seed" none skipped "$ckpt_dir/best.pt" "checkpoint exists"
    return 0
  fi
  if [[ -e "$ckpt_dir" ]]; then
    local interrupted_dir="$arm_root/checkpoints.interrupted_$(date +%Y%m%d_%H%M%S)"
    log_msg "Archiving incomplete Stage-2 checkpoint directory before rerun: arm=$arm seed=$seed -> $interrupted_dir"
    mv "$ckpt_dir" "$interrupted_dir"
    record_status train "$arm" "$seed" none archived "$interrupted_dir" "incomplete checkpoint directory"
  fi

  mapfile -t data_args < <(arm_data_paths "$arm")
  for data_path in "${data_args[@]}"; do
    require_path "$data_path"
  done
  align_ckpt="$(arm_stage1_ckpt "$arm")"
  require_path "$align_ckpt"
  loss_args_string="$(arm_loss_args "$arm")"

  mkdir -p "$ckpt_dir" "$arm_root"
  {
    printf 'arm=%s\ntraining_seed=%s\nstarted_at=%s\n' "$arm" "$seed" "$(timestamp)"
    printf 'align_ckpt=%s\nalign_ckpt_sha256=%s\n' "$align_ckpt" "$(sha256sum "$align_ckpt" | awk '{print $1}')"
    printf 'data_paths=%s\n' "${data_args[*]}"
    printf 'loss_args=%s\n' "$loss_args_string"
  } > "$arm_root/run.env"

  log_msg "Training arm=$arm seed=$seed"
  record_status train "$arm" "$seed" none running "$ckpt_dir" "training"
  # shellcheck disable=SC2206
  local loss_args=( $loss_args_string )
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" --standalone --nproc_per_node="$TRAIN_NPROC" "$REPO_ROOT/projection_train/train_projection_bc.py" \
    --data_path "${data_args[@]}" \
    --llm_path "$MODEL_PATH" --align_ckpt "$align_ckpt" \
    --output_dir "$ckpt_dir" \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --prompt_mode queue_wait_short \
    --action_eval_chunk_size "$STAGE2_ACTION_EVAL_CHUNK_SIZE" \
    --batch_size "$STAGE2_BATCH_SIZE" --grad_accum_steps "$STAGE2_GRAD_ACCUM_STEPS" \
    --epochs "$STAGE2_EPOCHS" --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers "$STAGE2_NUM_WORKERS" \
    --best_metric val_macro_f1 --best_mode max --best_min_delta 0.0 \
    --resample_mode targeted \
    --resample_late_mode date_step --resample_late_interval_seconds 2 --resample_late_start_step 316 --resample_late_weight 1 \
    --resample_worst_intersections intersection_1_1,intersection_1_2,intersection_1_3,intersection_2_2,intersection_3_2,intersection_4_2 \
    --resample_worst_intersection_weight 2 \
    --resample_action_labels ELWL,NLSL --resample_action_weight 1 \
    --resample_high_queue_threshold 14 --resample_high_queue_weight 1 \
    --resample_high_wait_threshold 84 \
    --resample_target_action_counts ETWT:4248,NTST:3934,ELWL:2055,NLSL:1885 \
    --log_interval 20 --save_every 1 --seed "$seed" --bf16 \
    --ddp_timeout_seconds 1800 \
    "${loss_args[@]}" \
    > "$log_path" 2>&1

  [[ -s "$ckpt_dir/best.pt" ]] || { record_status train "$arm" "$seed" none failed "$log_path" "missing best.pt"; return 1; }
  record_status train "$arm" "$seed" none completed "$ckpt_dir/best.pt" "ok"
}

online_result_ok() {
  local result="$1"
  [[ -s "$result" ]] || return 1
  timeout --foreground 30s "$PYTHON" - "$result" "$ONLINE_RUN_COUNTS" <<'PY'
import json
import math
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
args = payload.get("args", {})
results = payload.get("results", {})
if int(args.get("run_counts", -1)) != int(sys.argv[2]):
    raise SystemExit(1)
if args.get("traffic_file") != "anon_3_4_jinan_real.json":
    raise SystemExit(1)
if args.get("dataset") != "jinan":
    raise SystemExit(1)
for key in ("test_reward_over", "test_avg_queue_len_over", "test_avg_waiting_time_over", "test_avg_travel_time_over"):
    if key not in results or not math.isfinite(float(results[key])):
        raise SystemExit(1)
PY
}

run_online_seed_for_ckpt() {
  local gpu="$1"
  local arm="$2"
  local train_seed="$3"
  local online_seed="$4"
  local ckpt="$5"
  local visual_ablation="${6:-full}"
  local out_dir="$RESULTS_DIR/$arm/train_seed_$train_seed/online_seed_$online_seed"
  local result="$out_dir/result.json"
  local log_path="$LOG_DIR/online_${arm}_train${train_seed}_eval${online_seed}_gpu${gpu}.log"

  require_path "$ckpt"
  mkdir -p "$out_dir"
  if online_result_ok "$result"; then
    log_msg "Reuse online result: arm=$arm train_seed=$train_seed online_seed=$online_seed"
    record_status online "$arm" "$train_seed" "$online_seed" skipped "$result" "result exists"
    return 0
  fi

  log_msg "Online eval arm=$arm train_seed=$train_seed online_seed=$online_seed gpu=$gpu"
  record_status online "$arm" "$train_seed" "$online_seed" running "$result" "evaluating"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON" -u "$REPO_ROOT/projection_train/run_projection_online.py" \
    --memo "online_tests/$RUN_ID/$arm/train_seed_$train_seed/online_seed_$online_seed" \
    --projection_ckpt "$ckpt" \
    --llm_path "$MODEL_PATH" \
    --dataset jinan \
    --traffic_file "$TRAFFIC_FILE" \
    --run_counts "$ONLINE_RUN_COUNTS" \
    --num_rounds 1 \
    --seed "$online_seed" \
    --proj_name "Jinan-missing-ablation-$arm" \
    --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --traffic_series_interval 2 \
    --inference_batch_size 64 \
    --visual_ablation "$visual_ablation" \
    --bf16 --deterministic --disable_snapshots \
    --result_json "$result" \
    > "$log_path" 2>&1
  online_result_ok "$result"
  record_status online "$arm" "$train_seed" "$online_seed" completed "$result" "ok"
}

run_online_seed() {
  local gpu="$1"
  local arm="$2"
  local train_seed="$3"
  local online_seed="$4"
  local ckpt="$RUN_ROOT/stage2/$arm/train_seed_$train_seed/checkpoints/best.pt"
  run_online_seed_for_ckpt "$gpu" "$arm" "$train_seed" "$online_seed" "$ckpt" full
}

run_online_matrix_for_arm_seed() {
  local arm="$1"
  local train_seed="$2"
  local -a pids=()
  local total=${#ONLINE_SEEDS[@]}
  local slots=${#EVAL_GPU_ARRAY[@]}
  local start idx seed_index
  local online_seed gpu pid status

  status=0
  for ((start=0; start<total; start+=slots)); do
    pids=()
    for ((idx=0; idx<slots; idx+=1)); do
      seed_index=$((start + idx))
      (( seed_index < total )) || continue
      online_seed="${ONLINE_SEEDS[$seed_index]}"
      gpu="${EVAL_GPU_ARRAY[$idx]}"
      run_online_seed "$gpu" "$arm" "$train_seed" "$online_seed" &
      pid="$!"
      pids+=("$pid")
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
  done
  return "$status"
}

run_visual_online_matrix_for_full_loss_seed() {
  local train_seed="$1"
  local ckpt="$RUN_ROOT/stage2/full_loss/train_seed_$train_seed/checkpoints/best.pt"
  local -a visual_modes=(shuffled_image constant_image zero_prefix)
  local visual_mode arm total slots start idx seed_index online_seed gpu pid status
  local -a pids=()

  require_path "$ckpt"
  status=0
  for visual_mode in "${visual_modes[@]}"; do
    arm="visual_${visual_mode}"
    total=${#ONLINE_SEEDS[@]}
    slots=${#EVAL_GPU_ARRAY[@]}
    for ((start=0; start<total; start+=slots)); do
      pids=()
      for ((idx=0; idx<slots; idx+=1)); do
        seed_index=$((start + idx))
        (( seed_index < total )) || continue
        online_seed="${ONLINE_SEEDS[$seed_index]}"
        gpu="${EVAL_GPU_ARRAY[$idx]}"
        run_online_seed_for_ckpt "$gpu" "$arm" "$train_seed" "$online_seed" "$ckpt" "$visual_mode" &
        pid="$!"
        pids+=("$pid")
      done
      for pid in "${pids[@]}"; do
        wait "$pid" || status=1
      done
    done
  done
  return "$status"
}

aggregate_results() {
  log_msg "Aggregating ablation results."
  timeout --foreground 120s "$PYTHON" - "$RUN_ROOT" "$VISUAL_SIGNAL_RESULT" <<'PY'
import csv
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
visual = Path(sys.argv[2])
arms = [
    "control_shuffled_targets",
    "stage1_random_projector",
    "hard_bc",
    "hard_plus_soft",
    "hard_soft_ranking",
    "full_loss",
    "visual_shuffled_image",
    "visual_constant_image",
    "visual_zero_prefix",
]
metrics = [
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
]

rows = []
for result in sorted((root / "online_eval").glob("*/*/*/result.json")):
    rel = result.relative_to(root / "online_eval").parts
    if len(rel) < 4:
        continue
    arm, train_seed_name, online_seed_name = rel[0], rel[1], rel[2]
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
        vals = payload.get("results") or {}
        args = payload.get("args") or {}
    except Exception as exc:
        rows.append({"arm": arm, "result_json": str(result), "error": type(exc).__name__})
        continue
    row = {
        "arm": arm,
        "training_seed": train_seed_name.replace("train_seed_", ""),
        "online_seed": online_seed_name.replace("online_seed_", ""),
        "result_json": str(result),
        "checkpoint": args.get("projection_ckpt", ""),
    }
    for metric in metrics:
        value = vals.get(metric)
        if value is not None:
            row[metric] = float(value)
    rows.append(row)

summary = {}
for arm in arms:
    arm_rows = [r for r in rows if r.get("arm") == arm and "error" not in r]
    by_train = {}
    for row in arm_rows:
        by_train.setdefault(row["training_seed"], []).append(row)
    train_seed_stats = {}
    for seed, seed_rows in sorted(by_train.items()):
        train_seed_stats[seed] = {}
        for metric in metrics:
            values = [r[metric] for r in seed_rows if metric in r and math.isfinite(float(r[metric]))]
            if values:
                train_seed_stats[seed][metric] = {
                    "n_online": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                }
    arm_summary = {
        "n_rows": len(arm_rows),
        "n_training_seeds": len(by_train),
        "train_seed_stats": train_seed_stats,
        "across_training_seed_mean": {},
    }
    for metric in metrics:
        seed_means = [
            stat[metric]["mean"]
            for stat in train_seed_stats.values()
            if metric in stat and math.isfinite(float(stat[metric]["mean"]))
        ]
        if seed_means:
            arm_summary["across_training_seed_mean"][metric] = {
                "n_training_seeds": len(seed_means),
                "mean": statistics.fmean(seed_means),
                "std": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
                "seed_means": seed_means,
            }
    summary[arm] = arm_summary

visual_summary = None
if visual.is_file():
    try:
        visual_summary = json.loads(visual.read_text(encoding="utf-8"))
    except Exception as exc:
        visual_summary = {"error": type(exc).__name__, "path": str(visual)}

payload = {
    "schema": "jinan_missing_ablation_suite_aggregate_v1",
    "run_root": str(root),
    "rows": rows,
    "arms": summary,
    "completed_arms": [arm for arm, item in summary.items() if item["n_training_seeds"] > 0],
    "known_completed_external": {
        "visual_signal_ablation": str(visual) if visual.is_file() else "",
        "visual_signal_summary": visual_summary,
    },
    "not_run_by_this_script": {
        "modality_2x2_factorial": (
            "The current code has no parameter-matched numeric-only prefix encoder or "
            "clean switch for removing compact queue/wait cues during training. "
            "The script records this gap instead of producing a confounded pseudo-ablation."
        ),
        "data_source_matrix": (
            "The local Jinan RuleOnly/DAgger/SimExpert datasets are not present as a "
            "shared same-state, equal-budget label pool with identical candidate-cost metadata. "
            "The shell runs same-state real-vs-shuffled label controls instead."
        ),
    },
}
(root / "aggregate.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

with (root / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["arm", "training_seed", "online_seed", *metrics, "checkpoint", "result_json", "error"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

def fmt(item, metric):
    vals = item.get("across_training_seed_mean", {}).get(metric)
    if not vals:
        return "NA"
    return f"{vals['mean']:.4f} +/- {vals['std']:.4f}"

lines = [
    "# Jinan Missing Main-Paper Ablation Suite",
    "",
    f"Run root: `{root}`",
    "",
    "## Newly Run Arms",
    "",
    "| Arm | training seeds | online rows | Reward | AQL | AWT | ATT |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for arm in arms:
    item = summary[arm]
    lines.append(
        f"| `{arm}` | {item['n_training_seeds']} | {item['n_rows']} | "
        f"{fmt(item, 'test_reward_over')} | "
        f"{fmt(item, 'test_avg_queue_len_over')} | "
        f"{fmt(item, 'test_avg_waiting_time_over')} | "
        f"{fmt(item, 'test_avg_travel_time_over')} |"
    )
lines.extend([
    "",
    "## Existing Result Used Only As Prior Reference",
    "",
    "- Same-state real-target hard BC baseline is the newly run `hard_bc` arm.",
    "- Stage-1 aligned baseline is the newly run `full_loss` arm.",
    "- Exact visual intervention results for the requested model are newly run as `visual_shuffled_image`, `visual_constant_image`, and `visual_zero_prefix` using `full_loss` checkpoints.",
])
if visual_summary and isinstance(visual_summary, dict):
    lines.append(f"- Visual state-dependence ablation: `{visual}`")
    arms_dict = visual_summary.get("arms") or {}
    for arm in ("full", "shuffled_image", "constant_image", "zero_prefix"):
        item = arms_dict.get(arm, {})
        vals = item.get("metrics", {})
        awt = vals.get("test_avg_waiting_time_over", {})
        aql = vals.get("test_avg_queue_len_over", {})
        if awt and aql:
            lines.append(f"- `{arm}`: n={item.get('n', 0)}, AQL={aql.get('mean'):.4f}, AWT={awt.get('mean'):.4f}")
else:
    lines.append("- Visual state-dependence ablation: not found in this workspace.")
lines.extend([
    "",
    "## Explicit Gaps Not Claimed",
    "",
    "- Modality 2x2 is not launched because the repository does not expose a clean parameter-matched numeric-only prefix encoder.",
    "- Full data-source matrix is not launched because equal-budget Jinan RuleOnly/DAgger/SimExpert same-state label pools with identical candidate costs are not available locally.",
])
(root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(root / "RESULTS.md")
PY
}

main() {
  cd "$REPO_ROOT"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_ASYNC_ERROR_HANDLING=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  preflight
  write_provenance
  derive_control_data
  train_stage1_if_needed
  make_random_stage1

  # These are the missing cleanly-runnable Jinan additions:
  # - same-state negative control: real labels vs shuffled labels
  # - Stage-1 deletion: aligned projector vs random projector
  # - fixed-data loss deletion matrix: hard -> soft -> ranking -> KL/full
  local -a arms=(
    control_shuffled_targets
    stage1_random_projector
    hard_bc
    hard_plus_soft
    hard_soft_ranking
    full_loss
  )
  local arm seed
  for arm in "${arms[@]}"; do
    for seed in "${TRAINING_SEEDS[@]}"; do
      train_arm_seed "$arm" "$seed"
      run_online_matrix_for_arm_seed "$arm" "$seed"
      if [[ "$arm" == "full_loss" ]]; then
        run_visual_online_matrix_for_full_loss_seed "$seed"
      fi
      aggregate_results
    done
  done

  aggregate_results
  record_status suite all all all completed "$REPORT_MD" "done"
  log_msg "Completed. Report: $REPORT_MD"
}

main "$@"
