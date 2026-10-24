# Projection Stage Report

## Stage 3 PPO | 2026-05-24 16:18:29

- status: `success`
- stage_root: `/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825`
- started_at: `2026-05-24 16:18:29`
- ended_at: `2026-05-24 17:24:12`
- duration: `01:05:43`
- duration_seconds: `3942.982`
- train_started_at: `2026-05-24 16:18:29`
- train_ended_at: `2026-05-24 17:07:50`
- train_duration: `00:49:20`
- train_duration_seconds: `2960.288`
- test_started_at: `2026-05-24 17:08:04`
- test_ended_at: `2026-05-24 17:24:12`
- test_duration: `00:16:08`
- test_duration_seconds: `968.364`
- train_log: `/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/logs/train.log`
- test_log: `/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/logs/test.log`
- checkpoint: `/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints/best.pt`
- checkpoint_selected: `best.pt`
- test_work_dir: `records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/test_runs/anon_4_4_hangzhou_real.json_05_24_17_08_12`

### Train Command

```bash
/root/miniconda3/bin/torchrun --standalone --nproc_per_node 1 /gemini/code/LLMSTSC/projection_train/train_projection_ppo_traj.py --stage2_ckpt /gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_5816/stage3_waitfirst_lr1e6_clip002_kl08_ent0005_roll16_rc1800_u4_eval2_save1_20260523_053204/checkpoints/epoch_0004.pt --dataset hangzhou --traffic_file anon_4_4_hangzhou_real.json --eval_traffic_file anon_4_4_hangzhou_real.json --roadnet_path data/Hangzhou/4_4/roadnet_4_4.json --traffic_root data/Hangzhou/4_4 --llm_path /gemini/pretrain2/qwen-vl-8B/ --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --value_hidden_dim 512 --prompt_mode image_only_short --train_run_counts 3600 --eval_run_counts 3600 --num_updates 4 --rollout_steps 16 --ppo_epochs 1 --mini_batch_size 8 --grad_accum_steps 2 --action_eval_chunk_size 1 --snapshot_reset_prob 0 --eval_every 2 --save_every 1 --lr 1e-6 --clip_range 0.02 --kl_coef 0.8 --entropy_coef 0.0005 --value_coef 0.2 --reward_alpha 0.1 --switch_penalty_beta 0.02 --queue_reward_coef -0.25 --pressure_reward_coef 0.0 --waiting_reward_coef -0.01 --system_vehicle_reward_coef -0.01 --throughput_reward_coef 0.02 --starvation_penalty_coef 0.1 --starvation_threshold_steps 2 --starvation_queue_threshold 8.0 --starvation_wait_threshold 45.0 --wait_reward_scope neighborhood --wait_reward_reducer max --reward_mode stable_delta --queue_delta_scale 4.0 --wait_delta_scale 45.0 --wait_abs_coef 0.3 --wait_abs_scale 60.0 --active_delta_scale 2.0 --throughput_scale 0.5 --reward_clip 3.0 --global_reward_scale 1.0 --model_selection_metric composite --selection_queue_weight 1.0 --selection_wait_weight 2.0 --selection_travel_weight 1.5 --selection_max_queue_ratio 1.02 --selection_max_wait_ratio 1.03 --selection_max_travel_ratio 1.01 --seed 2024 --bf16 --output_dir /gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints --work_root /gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/env_work
```

### Test Command

```bash
/root/miniconda3/bin/python /gemini/code/LLMSTSC/projection_train/run_projection_online.py --memo pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/test_runs --projection_ckpt /gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints/best.pt --dataset hangzhou --traffic_file anon_4_4_hangzhou_real.json --device auto --seed 2024 --num_rounds 1 --proj_name LLM-TSCS-extreme --run_counts 3600 --result_json /gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/artifacts/test_result.json --llm_path /gemini/pretrain2/qwen-vl-8B --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode image_only_short --bf16
```

### Environment

```json
{
  "CUDA_VISIBLE_DEVICES": "0",
  "PYTHONUNBUFFERED": "1",
  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
}
```

### Training Args

```json
{
  "action_eval_chunk_size": 1,
  "active_delta_scale": 2.0,
  "bf16": true,
  "clip_range": 0.02,
  "dataset": "hangzhou",
  "device": "auto",
  "entropy_coef": 0.0005,
  "eval_every": 2,
  "eval_run_counts": 3600,
  "eval_traffic_file": "anon_4_4_hangzhou_real.json",
  "fp16": false,
  "fsdp_sharding_strategy": "full_shard",
  "gae_lambda": 0.95,
  "gamma": 0.99,
  "global_reward_scale": 1.0,
  "grad_accum_steps": 2,
  "gradient_checkpointing": "off",
  "hidden_dim": 64,
  "image_size": 224,
  "input_dim": 8,
  "kl_coef": 0.8,
  "llm_fsdp": "off",
  "llm_path": "/gemini/pretrain2/qwen-vl-8B/",
  "lr": 1e-06,
  "max_grad_norm": 1.0,
  "max_length": 64,
  "min_action_time": 30,
  "mini_batch_size": 8,
  "model_selection_metric": "composite",
  "num_updates": 4,
  "output_dir": "/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints",
  "periodicity": 16,
  "ppo_epochs": 1,
  "prefix_mode": "prefix",
  "prefix_tokens": 8,
  "pressure_reward_coef": 0.0,
  "prompt_mode": "image_only_short",
  "queue_delta_scale": 4.0,
  "queue_reward_coef": -0.25,
  "reward_alpha": 0.1,
  "reward_clip": 3.0,
  "reward_mode": "stable_delta",
  "roadnet_path": "data/Hangzhou/4_4/roadnet_4_4.json",
  "rollout_steps": 16,
  "run_counts": 0,
  "save_every": 1,
  "seed": 2024,
  "selection_max_queue_ratio": 1.02,
  "selection_max_travel_ratio": 1.01,
  "selection_max_wait_ratio": 1.03,
  "selection_queue_weight": 1.0,
  "selection_require_all_improved": false,
  "selection_travel_weight": 1.5,
  "selection_wait_weight": 2.0,
  "seq_len": 16,
  "snapshot_jsonl": null,
  "snapshot_max_samples": null,
  "snapshot_reset_prob": 0.0,
  "snapshot_traffic_file": "",
  "stage2_ckpt": "/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_5816/stage3_waitfirst_lr1e6_clip002_kl08_ent0005_roll16_rc1800_u4_eval2_save1_20260523_053204/checkpoints/epoch_0004.pt",
  "starvation_penalty_coef": 0.1,
  "starvation_queue_threshold": 8.0,
  "starvation_threshold_steps": 2,
  "starvation_wait_threshold": 45.0,
  "switch_penalty_beta": 0.02,
  "system_vehicle_reward_coef": -0.01,
  "throughput_reward_coef": 0.02,
  "throughput_scale": 0.5,
  "traffic_file": "anon_4_4_hangzhou_real.json",
  "traffic_root": "data/Hangzhou/4_4",
  "traffic_series_interval": 2,
  "train_run_counts": 3600,
  "value_coef": 0.2,
  "value_hidden_dim": 512,
  "wait_abs_coef": 0.3,
  "wait_abs_scale": 60.0,
  "wait_delta_scale": 45.0,
  "wait_reward_reducer": "max",
  "wait_reward_scope": "neighborhood",
  "waiting_reward_coef": -0.01,
  "weight_decay": 0.0001,
  "window_s": 30,
  "work_root": "/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/env_work"
}
```

### Training Metrics

```json
{
  "eval_all_metrics_improved": 0.0,
  "eval_avg_queue": 68.86666666666666,
  "eval_avg_travel_time": 322.91652698625546,
  "eval_avg_waiting": 54.95095414692479,
  "eval_composite_score": -1.0,
  "eval_left_turn_wait_max": 748.4,
  "eval_