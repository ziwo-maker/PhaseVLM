# Jinan Real Qwen3.5 Queue-Wait-Short Early-Stop Multi-Checkpoint Online Test

Generated: 2026-08-26

## Purpose

This record documents the Qwen3.5-9B backbone ablation on Jinan real traffic using the queue-wait-short targeted recipe. The Stage-2 training was manually early-stopped after seven completed epochs, then multiple available checkpoints were evaluated online in parallel.

This is **not** a full epoch-15 run. It should be interpreted as an early-stop ablation snapshot.

## Run Identity

- Run root:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe
```

- Source archive used as the reproduction target:

```text
/mnt/data/LLMSTSCrecords/records/online_tests/jinan3_qwshort_targeted_20260805_epoch9_epoch15/epoch_15/anon_3_4_jinan_real.json_08_05_12_35_29
```

- Online multi-checkpoint summary:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/online_eval_multi/summary.json
```

## Experiment Conditions

### Backbone and Model

- Model path: `/mnt/data/Qwen3.5-9B`
- Model family detected by loader: `qwen3_5`
- Selected HF model class: `Qwen3_5ForConditionalGeneration`
- Processor/tokenizer observed in preflight: `Qwen3VLProcessor` / `Qwen2Tokenizer`
- Precision: `bf16`

### Dataset and Traffic

- Dataset: `jinan_real`
- Traffic file: `anon_3_4_jinan_real.json`
- Traffic data directory: `data/Jinan/3_4`
- Training data files:
  - `/home/code/LLMSTSC/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
  - `/home/code/LLMSTSC/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
  - `/home/code/LLMSTSC/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl`

### Shared Architecture / Prompt Settings

- Prompt mode: `queue_wait_short`
- `seq_len=16`
- `image_size=224`
- `periodicity=16`
- `input_dim=8`
- `hidden_dim=64`
- `prefix_tokens=8`
- `prefix_mode=prefix`
- `max_length=64`

### Stage 1

Stage 1 was reused from the first Qwen3.5-9B ablation attempt, because that attempt completed Stage 1 successfully and only failed in Stage 2 due to CUDA OOM.

- Reused source checkpoint:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_104538/stage1/best.pt
```

- Copied checkpoint used by this run:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/stage1/best.pt
```

- Stage-1 recipe label: `pt8_hd64_temp0.05_epochs15_decision_only`
- Stage-1 settings recorded by launcher:
  - `batch_size=4`
  - `grad_accum_steps=2`
  - `num_workers=4`

### Stage 2

The initial Qwen3.5-9B Stage-2 attempt with `batch_size=4` OOMed. This run used memory-safe settings:

- Stage-2 recipe label: `pt8_hd64_queue_wait_short_epochs15_targeted_resample_fitted_to_20260805_archive`
- GPUs: `0,1,2,3`
- DDP processes: `4`
- `batch_size=1`
- `grad_accum_steps=16`
- `action_eval_chunk_size=1`
- `num_workers=4`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Target mode: `mixed`
- `soft_target_alpha=0.3`
- `soft_label_temperature=0.7`
- Best checkpoint metric: `val_macro_f1`, mode `max`
- Targeted resampling:
  - worst intersections: `intersection_1_1, intersection_1_2, intersection_1_3, intersection_2_2, intersection_3_2, intersection_4_2`
  - action labels: `ELWL,NLSL`
  - high queue threshold: `14`
  - high wait threshold: `84`
  - target action counts: `ETWT:4248, NTST:3934, ELWL:2055, NLSL:1885`

### Early Stop

- Manual early stop time: `2026-08-26T22:09:09+08:00`
- Launcher `completion.status`: `exit_status=1`
- Interpretation: the nonzero exit status is expected because the training process was manually stopped before epoch 15. It is not evidence of a training crash.
- Completed Stage-2 epochs: `1` through `7`
- Epoch 8 had started but was stopped before producing a complete checkpoint.

## Offline Stage-2 Validation

The training script selected `best.pt` by maximum `val_macro_f1`.

| Epoch | Val Macro-F1 ↑ | Val Acc ↑ | Val Loss ↓ | Val Top-1 Cost ↓ | Val Expected Cost ↓ | Val Top-1 Regret ↓ | Val Expected Regret ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.444948 | 0.548837 | 1.420056 | 1.008250 | 1.192785 | 0.615220 | 0.799756 |
| 2 | 0.641561 | 0.704651 | 0.928429 | 0.636829 | 0.841124 | 0.243799 | 0.448095 |
| 3 | 0.654526 | 0.720930 | 0.863591 | 0.591857 | 0.786889 | 0.198827 | 0.393860 |
| 4 | 0.675487 | 0.732558 | 0.851333 | 0.572881 | 0.729700 | 0.179852 | 0.336671 |
| 5 | 0.682871 | **0.741860** | **0.845778** | **0.568169** | 0.732095 | **0.175140** | 0.339066 |
| 6 | 0.674824 | 0.739535 | 0.865314 | 0.585570 | **0.712706** | 0.192541 | **0.319677** |
| 7 | **0.684948** | 0.723256 | 0.879438 | 0.615580 | 0.768015 | 0.222551 | 0.374985 |

Offline observations:

1. `best.pt` corresponds to epoch 7 by `val_macro_f1=0.684948`.
2. Epoch 5 is better on `val_acc`, `val_loss`, `val_top1_cost`, and `val_top1_cost_regret`.
3. Epoch 6 is better on expected-cost metrics.
4. This disagreement motivated running multiple online tests instead of trusting only `best.pt`.

## Online Test Conditions

- Online seed: `2024`
- Run counts: `3600`
- Traffic series interval: `2`
- Dataset: `jinan`
- Traffic file: `anon_3_4_jinan_real.json`
- Same-flow evaluation: yes, same Jinan real flow used by this recipe
- Online tests launched in parallel:
  - `best.pt` on GPU 0
  - `epoch_5.pt` on GPU 1
  - `epoch_6.pt` on GPU 2
  - `epoch_7.pt` on GPU 3

Result JSON files:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/online_eval_multi/seed_2024/best/result.json
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/online_eval_multi/seed_2024/epoch5/result.json
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/online_eval_multi/seed_2024/epoch6/result.json
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/online_eval_multi/seed_2024/epoch7/result.json
```

All four online tests completed normally. No OOM, traceback, or exception was found in the online logs.

## Online Results

Lower is better for AQL, AWT, ATT, queued vehicles, p95 wait, and max wait. Higher is better for reward.

| Checkpoint | Reward ↑ | AQL ↓ | Queued ↓ | AWT ↓ | ATT ↓ | Wait P95 ↓ | Wait Max ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best.pt` | **-611.7125** | **165.3667** | **19844** | 43.4024 | **274.3434** | 176.0 | 758 |
| `epoch_5.pt` | -615.9895 | 169.6333 | 20356 | 46.0708 | 276.2909 | 187.0 | **603** |
| `epoch_6.pt` | -614.2527 | 167.2917 | 20075 | 51.8085 | 274.7461 | 211.0 | 767 |
| `epoch_7.pt` | -615.8709 | 168.5917 | 20231 | **42.3080** | 276.0337 | **162.2** | 733 |

## Selected Best Checkpoint

Best overall online checkpoint: `best.pt`.

Path:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen35_9b_jinan3_qwshort_targeted_20260826_111532_oomsafe/stage2/checkpoints/best.pt
```

Reason:

- Best reward among tested checkpoints: `-611.7125`
- Best AQL among tested checkpoints: `165.3667`
- Best queued-vehicle count among tested checkpoints: `19844`
- Best ATT among tested checkpoints: `274.3434`

Note: `epoch_7.pt` is better for waiting-time tail metrics (`AWT=42.3080`, `Wait P95=162.2`), but its reward, AQL, queued count, and ATT are worse than `best.pt`.

## Delta Versus Source Archive

Source archive metrics recorded by the reproduction wrapper:

- Reward: `-599.1446`
- AQL: `157.1833`
- AWT: `42.8476`
- ATT: `268.4078`
- Queued: `18862`

Comparison using the selected online-best `best.pt`:

| Metric | Qwen3.5 Early-Stop `best.pt` | Source Archive | Delta |
|---|---:|---:|---:|
| Reward ↑ | -611.7125 | -599.1446 | -12.5679 |
| AQL ↓ | 165.3667 | 157.1833 | +8.1833 |
| AWT ↓ | 43.4024 | 42.8476 | +0.5548 |
| ATT ↓ | 274.3434 | 268.4078 | +5.9357 |
| Queued ↓ | 19844 | 18862 | +982 |

The early-stop Qwen3.5-9B ablation is worse than the source archive on the main online metrics.

## Interpretation

1. The memory-safe Stage-2 settings make Qwen3.5-9B trainable on 4x RTX 5090, but the early-stopped checkpoint does not match the source archive online performance.
2. Offline validation is not perfectly aligned with online traffic-control quality. Epoch 7 is best by `val_macro_f1`, epoch 5 is better by several offline cost metrics, but online `best.pt` is strongest on reward/AQL/ATT.
3. Early stopping after epoch 7 is reasonable for saving GPU time because online performance is already below the source archive and offline metrics show mixed improvement/overfitting signals.
4. If Qwen3.5-9B remains of interest, the next useful comparison is against the earlier Qwen3.5 epoch sweep result where `epoch_10.pt` was online-best, rather than continuing only this early-stop branch.

## Status

- Stage 2: manually early-stopped after 7 full epochs.
- Online multi-checkpoint sweep: completed.
- Recommended checkpoint for this early-stop run: `best.pt`.
- Recommended use in tables: mark clearly as `Qwen3.5-9B, early-stop, online-best among {best, epoch5, epoch6, epoch7}`.

