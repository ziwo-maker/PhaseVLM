# Jinan Real Qwen3-VL-4B Queue-Wait-Short Epoch-15 Online Test

Generated: 2026-08-27

## Purpose

This record documents the Qwen3-VL-4B-Instruct backbone ablation on Jinan real traffic using the queue-wait-short targeted recipe. The run completed the full Stage-1 alignment, full Stage-2 behavior-cloning training for 15 epochs, and one online evaluation using `epoch_15.pt`.

This is a full epoch-15 run, but not a multi-checkpoint online sweep. Offline Stage-2 metrics suggest that several earlier checkpoints may deserve online evaluation.

## Run Identity

- Run root:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_Qwen3-VL-4B-Instruct_20260826_222215
```

- Launcher script:

```text
/home/code/LLMSTSC/scripts/reproduce_jinan3_qwshort_targeted_20260805_epoch15.sh
```

- Source archive used as the reproduction target:

```text
/mnt/data/LLMSTSCrecords/records/online_tests/jinan3_qwshort_targeted_20260805_epoch9_epoch15/epoch_15/anon_3_4_jinan_real.json_08_05_12_35_29
```

- Completion status:

```text
exit_status=0
completed_at=2026-08-27T00:01:00+08:00
```

- Online summary JSON:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_Qwen3-VL-4B-Instruct_20260826_222215/online_eval/summary.json
```

- Online result JSON:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_Qwen3-VL-4B-Instruct_20260826_222215/online_eval/seed_2024/result.json
```

## Experiment Conditions

### Backbone and Model

- Model path: `/mnt/data/Qwen3-VL-4B-Instruct/`
- Model type observed in preflight: `qwen3_vl`
- Architecture observed in preflight: `Qwen3VLForConditionalGeneration`
- Processor/tokenizer observed in preflight: `Qwen3VLProcessor` / `Qwen2Tokenizer`
- Selected HF model class: `Qwen3VLForConditionalGeneration`
- Precision: `bf16`
- Stage-1 trainable parameters: `21,012,011`
- Stage-1 total parameters: `4,458,827,820`

### Dataset and Traffic

- Dataset: `jinan_real`
- Online dataset argument: `jinan`
- Traffic file: `anon_3_4_jinan_real.json`
- Traffic data directory: `data/Jinan/3_4`
- Online seed: `2024`
- Run counts: `3600`
- Traffic series interval: `2`
- Same-flow evaluation: yes, same Jinan real flow used by this recipe.

Training data files:

```text
/home/code/LLMSTSC/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
/home/code/LLMSTSC/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl
/home/code/LLMSTSC/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl
```

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

### GPU and Runtime Settings

- Stage-1/Stage-2 training GPUs: `0,1`
- DDP processes: `2`
- Online evaluation GPU: `0`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Python: `/home/sysu/anaconda3/envs/hnr5090/bin/python`
- Torchrun: `/home/sysu/anaconda3/envs/hnr5090/bin/torchrun`

## Stage 1 Alignment

- Recipe label: `pt8_hd64_temp0.05_epochs15_decision_only`
- Dataset size: `4320`
- Train size: `3888`
- Eval size: `432`
- Epochs: `15`
- Batch size per device: `4`
- Grad accumulation steps: `2`
- Effective global batch size: `16`
- Best checkpoint:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_Qwen3-VL-4B-Instruct_20260826_222215/stage1/best.pt
```

Stage-1 best checkpoint is epoch 15 by `mean_recall`.

| Metric | Value |
|---|---:|
| Best epoch | 15 |
| Best mean recall | 0.153935 |
| Train loss | 1.677284 |
| R@1 image-to-text | 0.030093 |
| R@1 text-to-image | 0.034722 |
| R@5 image-to-text | 0.127315 |
| R@5 text-to-image | 0.185185 |
| R@10 image-to-text | 0.240741 |
| R@10 text-to-image | 0.305556 |
| Total elapsed | 00:11:31 |
| Peak reserved memory | 9.312 GB |

## Stage 2 Behavior Cloning

- Recipe label: `pt8_hd64_queue_wait_short_epochs15_targeted_resample_fitted_to_20260805_archive`
- Epochs: `15`
- Batch size per device: `4`
- Grad accumulation steps: `4`
- Action eval chunk size: `0`
- Target mode: `mixed`
- `soft_target_alpha=0.3`
- `soft_label_temperature=0.7`
- Best checkpoint metric: `val_macro_f1`, mode `max`
- Online evaluated checkpoint:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_Qwen3-VL-4B-Instruct_20260826_222215/stage2/checkpoints/epoch_15.pt
```

### Stage-2 Offline Validation Summary

| Selection | Epoch | Val Macro-F1 ↑ | Val Acc ↑ | Val Loss ↓ | Val Top-1 Regret ↓ | Val Expected Regret ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Best macro-F1 | 5 | 0.776152 | 0.811628 | 0.682124 | 0.107599 | 0.272671 |
| Best top-1 cost regret | 8 | 0.761000 | 0.802326 | 0.696317 | 0.104122 | 0.256118 |
| Best expected cost regret | 9 | 0.743363 | 0.781395 | 0.706385 | 0.117164 | 0.234784 |
| Final / online-tested | 15 | 0.753248 | 0.793023 | 0.697695 | 0.112125 | 0.236290 |

### Stage-2 Epoch Table

| Epoch | Val Acc ↑ | Val Macro-F1 ↑ | Val Loss ↓ | Val Top-1 Regret ↓ | Val Expected Regret ↓ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.686047 | 0.620103 | 0.896240 | 0.239064 | 0.466676 |
| 2 | 0.760465 | 0.724994 | 0.856082 | 0.174213 | 0.350530 |
| 3 | 0.786047 | 0.741113 | 0.736279 | 0.133455 | 0.294370 |
| 4 | 0.779070 | 0.734162 | 0.690373 | 0.127841 | 0.260709 |
| 5 | **0.811628** | **0.776152** | **0.682124** | 0.107599 | 0.272671 |
| 6 | 0.788372 | 0.747849 | 0.690251 | 0.120735 | 0.261367 |
| 7 | 0.772093 | 0.747239 | 0.718545 | 0.160532 | 0.284604 |
| 8 | 0.802326 | 0.761000 | 0.696317 | **0.104122** | 0.256118 |
| 9 | 0.781395 | 0.743363 | 0.706385 | 0.117164 | **0.234784** |
| 10 | 0.793023 | 0.753614 | 0.708869 | 0.112822 | 0.250894 |
| 11 | 0.800000 | 0.757754 | 0.702951 | 0.111453 | 0.239434 |
| 12 | 0.793023 | 0.759931 | 0.704565 | 0.122552 | 0.247770 |
| 13 | 0.802326 | 0.767821 | 0.699603 | 0.110975 | 0.239087 |
| 14 | 0.800000 | 0.767844 | 0.699746 | 0.112387 | 0.241862 |
| 15 | 0.793023 | 0.753248 | 0.697695 | 0.112125 | 0.236290 |

Offline observations:

1. Stage-2 validation best by macro-F1 is epoch 5, not epoch 15.
2. Cost-oriented validation metrics choose different checkpoints: epoch 8 for top-1 regret and epoch 9 for expected regret.
3. Training macro-F1 reached 0.962668 by epoch 15, while validation macro-F1 peaked at epoch 5 and ended at 0.753248. This suggests overfitting or offline/online metric mismatch after the early-middle epochs.

## Online Test Conditions

- Online checkpoint: `epoch_15.pt`
- Online seed: `2024`
- Run counts: `3600`
- Prompt mode: `queue_wait_short`
- Deterministic mode: `false`
- `bf16=true`
- Online work directory:

```text
records/online_tests/repro_jinan3_qwshort_targeted_20260805_epoch15/Qwen3-VL-4B-Instruct/epoch_15/anon_3_4_jinan_real.json_08_26_23_59_48
```

- Online evaluator log:

```text
/home/code/LLMSTSC/records/pipeline_runs/repro_jinan3_qwshort_targeted_20260805_epoch15_Qwen3-VL-4B-Instruct_20260826_222215/logs/online_seed_2024.log
```

- Artifact hashes:
  - `projection_ckpt_sha256=e010107fb38a5fe2f5f168432fd2abfa07768d26f00b8cd4018e1bb425064bc2`
  - `online_evaluator_sha256=1ffe30b0759d1e1017117dc59ebb35aeedda1af266112fbabc767ecc6db08f75`

## Online Results

Lower is better for AQL, AWT, ATT, queued vehicles, p95 wait, and max wait. Higher is better for reward.

| Checkpoint | Reward ↑ | AQL ↓ | Queued ↓ | AWT ↓ | ATT ↓ | Wait P95 ↓ | Wait Max ↓ | Observed Wait Vehicles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch_15.pt` | -601.2091 | 158.6250 | 19035 | 46.2319 | 269.5698 | 185.65 | 617 | 4688 |

## Delta Versus Source Archive

Source archive metrics were verified from:

```text
/mnt/data/LLMSTSCrecords/records/online_tests/jinan3_qwshort_targeted_20260805_epoch9_epoch15/epoch_15/anon_3_4_jinan_real.json_08_05_12_35_29/projection_test.log
```

Source archive metrics:

- Reward: `-599.1446`
- AQL: `157.1833`
- AWT: `42.8476`
- ATT: `268.4078`
- Queued: `18862`

| Metric | Qwen3-VL-4B `epoch_15.pt` | Source Archive | Delta | Relative Delta |
|---|---:|---:|---:|---:|
| Reward ↑ | -601.2091 | -599.1446 | -2.0645 | -0.35% |
| AQL ↓ | 158.6250 | 157.1833 | +1.4417 | +0.92% |
| AWT ↓ | 46.2319 | 42.8476 | +3.3843 | +7.90% |
| ATT ↓ | 269.5698 | 268.4078 | +1.1620 | +0.43% |
| Queued ↓ | 19035 | 18862 | +173 | +0.92% |

Interpretation: the Qwen3-VL-4B epoch-15 run is worse than the source archive on all primary online metrics. The largest degradation is average waiting time.

## Comparison With Existing All_best Records

The existing `All_best` records contain two Qwen3.5-9B references:

1. Full/restarted Qwen3.5-9B online sweep:
   - Record: `/home/code/LLMSTSC/All_best/jinan_real_qwen35_qwshort_online_best_20260826.md`
   - Online-best checkpoint: `epoch_10.pt`
2. Qwen3.5-9B early-stop multi-checkpoint online test:
   - Record: `/home/code/LLMSTSC/All_best/jinan_real_qwen35_qwshort_earlystop_online_multi_20260826.md`
   - Online-best checkpoint in that early-stop branch: `best.pt`

### Main Online Comparison

| Run | Checkpoint | Reward ↑ | AQL ↓ | Queued ↓ | AWT ↓ | ATT ↓ |
|---|---|---:|---:|---:|---:|---:|
| Source archive | archived epoch 15 | **-599.1446** | **157.1833** | **18862** | **42.8476** | **268.4078** |
| Qwen3-VL-4B full run | epoch_15.pt | -601.2091 | 158.6250 | 19035 | 46.2319 | 269.5698 |
| Qwen3.5-9B full/restart sweep | epoch_10.pt | -610.4482 | 162.5500 | 19506 | 43.6375 | 271.9438 |
| Qwen3.5-9B early-stop branch | best.pt | -611.7125 | 165.3667 | 19844 | 43.4024 | 274.3434 |

### Delta Versus Qwen3.5-9B Full/Restart Online-Best

Baseline here is Qwen3.5-9B full/restart sweep `epoch_10.pt`.

| Metric | Qwen3-VL-4B `epoch_15.pt` | Qwen3.5-9B `epoch_10.pt` | Delta |
|---|---:|---:|---:|
| Reward ↑ | -601.2091 | -610.4482 | +9.2391 |
| AQL ↓ | 158.6250 | 162.5500 | -3.9250 |
| AWT ↓ | 46.2319 | 43.6375 | +2.5944 |
| ATT ↓ | 269.5698 | 271.9438 | -2.3740 |
| Queued ↓ | 19035 | 19506 | -471 |

Interpretation: Qwen3-VL-4B is stronger than the recorded Qwen3.5-9B online-best on reward, AQL, queued vehicles, and ATT, but worse on average waiting time.

### Delta Versus Qwen3.5-9B Early-Stop Online-Best

Baseline here is Qwen3.5-9B early-stop `best.pt`.

| Metric | Qwen3-VL-4B `epoch_15.pt` | Qwen3.5 Early-Stop `best.pt` | Delta |
|---|---:|---:|---:|
| Reward ↑ | -601.2091 | -611.7125 | +10.5033 |
| AQL ↓ | 158.6250 | 165.3667 | -6.7417 |
| AWT ↓ | 46.2319 | 43.4024 | +2.8295 |
| ATT ↓ | 269.5698 | 274.3434 | -4.7736 |
| Queued ↓ | 19035 | 19844 | -809 |

Interpretation: Qwen3-VL-4B is clearly better than the early-stop Qwen3.5 branch on reward, AQL, queued vehicles, and ATT, while still worse on average waiting time.

## Key Findings

1. Qwen3-VL-4B is the closest tested backbone to the source archive among the recorded Qwen3.5/Qwen3-VL alternatives, but it still does not match the source archive.
2. Compared with the source archive, Qwen3-VL-4B only has small degradation in reward, AQL, ATT, and queued vehicles, but a larger degradation in AWT.
3. Compared with Qwen3.5-9B records in `All_best`, Qwen3-VL-4B is substantially better on reward, queue length, queued count, and travel time. Its weakness is waiting time.
4. Offline Stage-2 validation does not agree with the online checkpoint choice. The online run used epoch 15, but validation macro-F1 peaks at epoch 5, top-1 cost regret at epoch 8, and expected cost regret at epoch 9.
5. The current result should be reported as `Qwen3-VL-4B, full epoch-15, online-tested epoch_15 only`, not as the best possible Qwen3-VL-4B checkpoint.

## Suggested Next Experiments

1. Run online evaluation for Qwen3-VL-4B checkpoints selected by offline metrics:
   - `stage2/checkpoints/best.pt`
   - `stage2/checkpoints/epoch_5.pt`
   - `stage2/checkpoints/epoch_8.pt`
   - `stage2/checkpoints/epoch_9.pt`
2. If compute budget is limited, prioritize `best.pt`, `epoch_8.pt`, and `epoch_9.pt`, because they correspond to macro-F1, top-1 cost regret, and expected cost regret selections.
3. If waiting time remains worse across those checkpoints, inspect per-intersection waiting tails and action distribution, especially for the targeted worst intersections and `ELWL/NLSL` resampling actions.
4. For paper/table reporting, keep source archive as the main baseline and report this run as a strong but still inferior backbone ablation.

## Status

- Stage 1: completed.
- Stage 2: completed through epoch 15.
- Online evaluation: completed for `epoch_15.pt`.
- Recommended checkpoint for current record: `epoch_15.pt`, because it is the only Qwen3-VL-4B checkpoint evaluated online so far.
- Recommended table label: `Qwen3-VL-4B-Instruct, full epoch-15, online epoch_15`.
