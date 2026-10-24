# PhaseVLM

This is the official repository for **PhaseVLM: Topology-Aware Vision-Language
Modeling with Time Series Traffic Images for Traffic Signal Control**.

The paper is submitted to **AAAI 2027**.

## Basic Usage

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run a basic experiment:

```bash
python run.py
```

For projection-stage training, see `train_align.py` and the scripts under
`projection_train/`.

## Main Experiment Startup

Use the following shell script as the canonical startup entry for the main
Jinan3 qwen short-prompt targeted experiment:

```bash
scripts/reproduce_jinan3_qwshort_targeted_20260805_epoch15.sh
```

This is the primary experiment launcher for the August 2026 main rerun. It
keeps the training, checkpoint selection target, online evaluation, logs, summary
JSON, and Feishu progress/completion notifications in one reproducible pipeline.
Run the component commands manually only when debugging a failed stage.

The launcher runs three stages:

1. Stage-1 alignment training with `train_align.py`.
2. Stage-2 projection BC training with
   `projection_train/train_projection_bc.py`.
3. Online evaluation with `projection_train/run_projection_online.py`.

Default paths used by the launcher:

```bash
REPO_ROOT=/home/code/LLMSTSC
PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
TORCHRUN=/home/sysu/anaconda3/envs/hnr5090/bin/torchrun
MODEL_PATH=/home/data/model/qwen-vl-8B
```

To run the same Jinan3 targeted pipeline as a Qwen3.5-9B backbone ablation, keep
the training/evaluation recipe fixed and only override the model path plus a
clear run label:

```bash
cd /home/code/LLMSTSC

MODEL_PATH=/mnt/data/Qwen3.5-9B \
MODEL_LABEL=qwen35_9b \
RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_$(date +%Y%m%d_%H%M%S) \
TRAIN_GPUS=0,1,2,3 \
EVAL_GPU=0 \
nohup scripts/reproduce_jinan3_qwshort_targeted_20260805_epoch15.sh \
  > /tmp/ablation_qwen35_9b_jinan3_qwshort_targeted_launcher.log 2>&1 &

echo $!
```

The launcher performs a lightweight Hugging Face preflight before training. It
checks that the selected model exposes `vision_config`, loads
`AutoProcessor`/`AutoTokenizer`, and chooses the compatible model class such as
`Qwen3_5ForConditionalGeneration` for `/mnt/data/Qwen3.5-9B`.

Do not evaluate an old `/home/data/model/qwen-vl-8B` Stage-2 checkpoint with
`MODEL_PATH=/mnt/data/Qwen3.5-9B`; the action prefix projector is tied to the
backbone text/vision dimensions. For a fair model-backbone ablation, run Stage 1
and Stage 2 from scratch with the Qwen3.5 path, then evaluate the resulting
`epoch_15.pt`.

To start the main experiment on four GPUs, request approval with the complete
command first, then run it detached and give every run a fresh `RUN_ROOT`:

```bash
cd /home/code/LLMSTSC

RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/main_jinan3_qwshort_targeted_4gpu_$(date +%Y%m%d_%H%M%S) \
TRAIN_GPUS=0,1,2,3 \
EVAL_GPU=0 \
nohup scripts/reproduce_jinan3_qwshort_targeted_20260805_epoch15.sh \
  > /tmp/main_jinan3_qwshort_targeted_4gpu_launcher.log 2>&1 &

echo $!
```

Check progress with bounded log snapshots:

```bash
RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/main_jinan3_qwshort_targeted_4gpu_YYYYMMDD_HHMMSS

tail -n 120 "$RUN_ROOT/logs/stage1.log"
tail -n 120 "$RUN_ROOT/logs/stage2.log"
tail -n 120 "$RUN_ROOT/logs/online_seed_2024.log"
sed -n '1,80p' "$RUN_ROOT/logs/completion.status"
```

To run one online test for a selected Stage-2 checkpoint:

```bash
cd /home/code/LLMSTSC

PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python
MODEL_PATH=/home/data/model/qwen-vl-8B
RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642
EPOCH=12
OUT_DIR="$RUN_ROOT/online_eval_epoch_${EPOCH}/seed_2024"
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" projection_train/run_projection_online.py \
  --memo "online_tests/repro_jinan3_qwshort_targeted_epoch_${EPOCH}/epoch_${EPOCH}" \
  --projection_ckpt "$RUN_ROOT/stage2/checkpoints/epoch_${EPOCH}.pt" \
  --llm_path "$MODEL_PATH" \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --run_counts 3600 \
  --seed 2024 \
  --proj_name "Jinan3-qwshort-targeted-epoch${EPOCH}-online" \
  --prompt_mode queue_wait_short \
  --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
  --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
  --traffic_series_interval 2 \
  --bf16 \
  --result_json "$OUT_DIR/result.json" \
  > "$OUT_DIR/online.log" 2>&1
```

To compare several epochs, run independent online tests in parallel, one worker
per GPU. This example evaluates epochs `3, 5, 7, 12, 14`; GPU 0 runs epochs `3`
and `14` sequentially:

```bash
cd /home/code/LLMSTSC

RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642
SWEEP_ROOT="$RUN_ROOT/online_eval_epoch_sweep_20260825_multi"
mkdir -p "$SWEEP_ROOT/logs"

setsid bash -c 'set -uo pipefail; cd /home/code/LLMSTSC; PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python; MODEL_PATH=/home/data/model/qwen-vl-8B; RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642; CKPT_DIR="$RUN_ROOT/stage2/checkpoints"; SWEEP_ROOT="$RUN_ROOT/online_eval_epoch_sweep_20260825_multi"; LOG_DIR="$SWEEP_ROOT/logs"; for epoch in 3 14; do out_dir="$SWEEP_ROOT/epoch_${epoch}"; mkdir -p "$out_dir"; CUDA_VISIBLE_DEVICES=0 "$PYTHON" projection_train/run_projection_online.py --memo "online_tests/repro_jinan3_qwshort_targeted_epoch_sweep_20260825/epoch_${epoch}" --projection_ckpt "$CKPT_DIR/epoch_${epoch}.pt" --llm_path "$MODEL_PATH" --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --seed 2024 --proj_name "Jinan3-qwshort-targeted-epoch${epoch}-online" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --traffic_series_interval 2 --bf16 --result_json "$out_dir/result.json" > "$LOG_DIR/epoch_${epoch}_gpu0.log" 2>&1; status=$?; echo "exit_status=$status completed_at=$(date --iso-8601=seconds) epoch=$epoch gpu=0" > "$out_dir/status.txt"; if [ "$status" -ne 0 ]; then exit "$status"; fi; done' > /tmp/jinan3_epoch_sweep_worker_gpu0.log 2>&1 < /dev/null &

setsid bash -c 'set -uo pipefail; cd /home/code/LLMSTSC; PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python; MODEL_PATH=/home/data/model/qwen-vl-8B; RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642; CKPT_DIR="$RUN_ROOT/stage2/checkpoints"; SWEEP_ROOT="$RUN_ROOT/online_eval_epoch_sweep_20260825_multi"; LOG_DIR="$SWEEP_ROOT/logs"; epoch=5; out_dir="$SWEEP_ROOT/epoch_${epoch}"; mkdir -p "$out_dir"; CUDA_VISIBLE_DEVICES=1 "$PYTHON" projection_train/run_projection_online.py --memo "online_tests/repro_jinan3_qwshort_targeted_epoch_sweep_20260825/epoch_${epoch}" --projection_ckpt "$CKPT_DIR/epoch_${epoch}.pt" --llm_path "$MODEL_PATH" --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --seed 2024 --proj_name "Jinan3-qwshort-targeted-epoch${epoch}-online" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --traffic_series_interval 2 --bf16 --result_json "$out_dir/result.json" > "$LOG_DIR/epoch_${epoch}_gpu1.log" 2>&1; status=$?; echo "exit_status=$status completed_at=$(date --iso-8601=seconds) epoch=$epoch gpu=1" > "$out_dir/status.txt"; exit "$status"' > /tmp/jinan3_epoch_sweep_worker_gpu1.log 2>&1 < /dev/null &

setsid bash -c 'set -uo pipefail; cd /home/code/LLMSTSC; PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python; MODEL_PATH=/home/data/model/qwen-vl-8B; RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642; CKPT_DIR="$RUN_ROOT/stage2/checkpoints"; SWEEP_ROOT="$RUN_ROOT/online_eval_epoch_sweep_20260825_multi"; LOG_DIR="$SWEEP_ROOT/logs"; epoch=7; out_dir="$SWEEP_ROOT/epoch_${epoch}"; mkdir -p "$out_dir"; CUDA_VISIBLE_DEVICES=2 "$PYTHON" projection_train/run_projection_online.py --memo "online_tests/repro_jinan3_qwshort_targeted_epoch_sweep_20260825/epoch_${epoch}" --projection_ckpt "$CKPT_DIR/epoch_${epoch}.pt" --llm_path "$MODEL_PATH" --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --seed 2024 --proj_name "Jinan3-qwshort-targeted-epoch${epoch}-online" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --traffic_series_interval 2 --bf16 --result_json "$out_dir/result.json" > "$LOG_DIR/epoch_${epoch}_gpu2.log" 2>&1; status=$?; echo "exit_status=$status completed_at=$(date --iso-8601=seconds) epoch=$epoch gpu=2" > "$out_dir/status.txt"; exit "$status"' > /tmp/jinan3_epoch_sweep_worker_gpu2.log 2>&1 < /dev/null &

setsid bash -c 'set -uo pipefail; cd /home/code/LLMSTSC; PYTHON=/home/sysu/anaconda3/envs/hnr5090/bin/python; MODEL_PATH=/home/data/model/qwen-vl-8B; RUN_ROOT=/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_4gpu_20260825_182642; CKPT_DIR="$RUN_ROOT/stage2/checkpoints"; SWEEP_ROOT="$RUN_ROOT/online_eval_epoch_sweep_20260825_multi"; LOG_DIR="$SWEEP_ROOT/logs"; epoch=12; out_dir="$SWEEP_ROOT/epoch_${epoch}"; mkdir -p "$out_dir"; CUDA_VISIBLE_DEVICES=3 "$PYTHON" projection_train/run_projection_online.py --memo "online_tests/repro_jinan3_qwshort_targeted_epoch_sweep_20260825/epoch_${epoch}" --projection_ckpt "$CKPT_DIR/epoch_${epoch}.pt" --llm_path "$MODEL_PATH" --dataset jinan --traffic_file anon_3_4_jinan_real.json --run_counts 3600 --seed 2024 --proj_name "Jinan3-qwshort-targeted-epoch${epoch}-online" --prompt_mode queue_wait_short --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --traffic_series_interval 2 --bf16 --result_json "$out_dir/result.json" > "$LOG_DIR/epoch_${epoch}_gpu3.log" 2>&1; status=$?; echo "exit_status=$status completed_at=$(date --iso-8601=seconds) epoch=$epoch gpu=3" > "$out_dir/status.txt"; exit "$status"' > /tmp/jinan3_epoch_sweep_worker_gpu3.log 2>&1 < /dev/null &
```

Check the online sweep:

```bash
ps -eo pid,ppid,pgid,sid,stat,etime,cmd | rg 'projection_train/run_projection_online.py'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
tail -n 80 "$SWEEP_ROOT/logs/epoch_12_gpu3.log"
sed -n '1,20p' "$SWEEP_ROOT"/epoch_*/status.txt
```

## 开发注意事项

- DDP/NCCL 训练中，不能只让 `rank 0` 在 epoch 边界执行 `eval`、`checkpoint`、长耗时 I/O 或其他独占逻辑，而不和其他 rank 做同步。
- 否则其他 rank 可能提前进入下一轮 `backward`，卡死在首个梯度 `all_reduce`，最终表现为 NCCL collective timeout。
- 规则：凡是只在部分 rank 执行的 epoch 级逻辑，前后都要显式做 `dist.barrier()`，或者改成所有 rank 都参与的实现。
- 如果只是 `rank 0` 做验证，优先使用 `model.module` 这类非 DDP wrapper 对象，避免验证阶段误触发额外 collective。
- 这次 `train_align.py` 的修复背景见 `README/train_align.md`。
