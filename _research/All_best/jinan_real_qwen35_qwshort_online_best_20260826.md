# Jinan Real Qwen3.5 Queue-Wait-Short Online Best

Generated: 2026-08-26

## Run

- Dataset: `jinan_real`
- Traffic file: `anon_3_4_jinan_real.json`
- Model path: `/mnt/data/Qwen3.5-9B`
- Prompt mode: `queue_wait_short`
- Architecture: `seq_len=16`, `image_size=224`, `periodicity=16`, `input_dim=8`, `hidden_dim=64`, `prefix_tokens=8`, `max_length=64`
- Stage-1 checkpoint: `/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_mntdata_qwen_jinan_real_qwen35_20260825_210721/stage1/best.pt`
- Stage-2 checkpoint directory: `/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_mntdata_qwen_jinan_real_qwen35_20260825_210721/stage2_restart_bs1/checkpoints`
- Online sweep root: `/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_mntdata_qwen_jinan_real_qwen35_20260825_210721/online_eval_epoch_sweep_top5_20260826`
- Online summary JSON: `/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_mntdata_qwen_jinan_real_qwen35_20260825_210721/online_eval_epoch_sweep_top5_20260826/summary.json`

## Selected Best

Best online checkpoint: `epoch_10.pt`

Path:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_mntdata_qwen_jinan_real_qwen35_20260825_210721/stage2_restart_bs1/checkpoints/epoch_10.pt
```

Reason: `epoch_10` has the best online reward among the tested checkpoints, and also has the lowest average queue length, average waiting time, and average travel time in this sweep.

## Online Results

Lower is better for AQL, AWT, ATT, and queued vehicles. Higher is better for reward.

| Rank | Epoch | Reward | AQL | AWT | ATT | Queued |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | -610.4482 | 162.5500 | 43.6375 | 271.9438 | 19506 |
| 2 | 15 existing | -612.4346 | 166.0417 | 47.9552 | 274.0531 | 19925 |
| 3 | 8 | -613.6755 | 164.7000 | 47.1492 | 273.2696 | 19764 |
| 4 | 14 | -615.3201 | 164.0000 | 46.5017 | 272.2294 | 19680 |
| 5 | 12 | -615.5749 | 166.8917 | 47.5301 | 274.9060 | 20027 |
| 6 | 6 | -619.3335 | 169.6417 | 49.5280 | 276.6070 | 20357 |

## Delta Versus Existing Epoch 15

Baseline: existing `epoch_15` online result from `online_eval_restart_bs1`.

| Metric | Epoch 10 | Epoch 15 Existing | Delta |
|---|---:|---:|---:|
| Reward | -610.4482 | -612.4346 | +1.9864 |
| AQL | 162.5500 | 166.0417 | -3.4917 |
| AWT | 43.6375 | 47.9552 | -4.3177 |
| ATT | 271.9438 | 274.0531 | -2.1093 |
| Queued | 19506 | 19925 | -419 |

## Delta Versus Source Archive

Source archive metrics recorded by the reproduction wrapper:

- Reward: `-599.1446`
- AQL: `157.1833`
- AWT: `42.8476`
- ATT: `268.4078`
- Queued: `18862`

| Metric | Epoch 10 | Source Archive | Delta |
|---|---:|---:|---:|
| Reward | -610.4482 | -599.1446 | -11.3036 |
| AQL | 162.5500 | 157.1833 | +5.3667 |
| AWT | 43.6375 | 42.8476 | +0.7899 |
| ATT | 271.9438 | 268.4078 | +3.5360 |
| Queued | 19506 | 18862 | +644 |

## Training Notes

- Stage-2 restart used `batch_size=1`, `grad_accum_steps=16`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid the earlier CUDA OOM.
- Stage-2 validation best checkpoint was `best.pt` from epoch 12 with `val_macro_f1=0.696436`.
- Online selection disagrees with validation selection: epoch 12 is validation-best, but epoch 10 is online-best.
- The online sweep completed normally with `exit_status=0` at `2026-08-26T09:02:28+08:00`.
