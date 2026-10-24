# Jinan Real Qwen3-VL-2B Queue-Wait-Short Fullmem Epoch-15 Online Test

Generated: 2026-08-27

## Purpose

This record documents the Qwen3-VL-2B-Instruct backbone ablation on Jinan real traffic using the queue-wait-short targeted recipe. The run was restarted with a larger Stage-2 batch size to occupy GPU memory more fully. It reused the Stage-1 alignment checkpoint from the earlier Qwen3-VL-2B run, then completed Stage-2 behavior-cloning training for 15 epochs and one online evaluation using `epoch_15.pt`.

This is a full epoch-15 online test, not a multi-checkpoint online sweep. Offline validation suggests that epochs 8, 10, and 11 may deserve online evaluation.

## Run Identity

- Run root:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710
```

- Launcher script:

```text
/home/code/LLMSTSC/scripts/reproduce_jinan3_qwshort_targeted_20260805_epoch15.sh
```

- Source archive used as reproduction target:

```text
/mnt/data/LLMSTSCrecords/records/online_tests/jinan3_qwshort_targeted_20260805_epoch9_epoch15/epoch_15/anon_3_4_jinan_real.json_08_05_12_35_29
```

- Completion status:

```text
exit_status=0
completed_at=2026-08-27T00:05:01+08:00
```

- Online summary JSON:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/online_eval/summary.json
```

- Online result JSON:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/online_eval/seed_2024/result.json
```

## Experiment Conditions

### Backbone and Model

- Model path: `/mnt/data/Qwen3-VL-2B-Instruct/`
- Model type observed in preflight: `qwen3_vl`
- Architecture observed in preflight: `Qwen3VLForConditionalGeneration`
- Processor/tokenizer observed in preflight: `Qwen3VLProcessor` / `Qwen2Tokenizer`
- Selected HF model class: `Qwen3VLForConditionalGeneration`
- Precision: `bf16`
- Model label: `qwen3vl_2b_fullmem`

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

- Stage-2 training GPUs: `2,3`
- DDP processes: `2`
- Online evaluation GPU: `2`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Python: `/home/sysu/anaconda3/envs/hnr5090/bin/python`
- Torchrun: `/home/sysu/anaconda3/envs/hnr5090/bin/torchrun`
- Observed Stage-2 training memory during startup monitoring: about `28852 MiB / 32607 MiB` on each of GPU 2 and GPU 3.

## Stage 1 Alignment

Stage 1 was reused from an earlier Qwen3-VL-2B run instead of retrained in this fullmem run.

- Reused checkpoint:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_20260826_224359/stage1/best.pt
```

- Copied checkpoint in this run:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage1/best.pt
```

- Recipe label recorded by launcher: `pt8_hd64_temp0.05_epochs15_decision_only`
- Stage-1 batch size per device in source recipe: `4`
- Stage-1 grad accumulation steps in source recipe: `2`
- Reuse info:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage1/reused_stage1.info
```

Important note: because Stage 1 was reused, this run should be interpreted as a Qwen3-VL-2B Stage-2/fullmem recipe evaluation, not a fully independent from-scratch Stage-1 plus Stage-2 rerun.

## Stage 2 Behavior Cloning

- Recipe label: `pt8_hd64_queue_wait_short_epochs15_targeted_resample_fitted_to_20260805_archive`
- Epochs: `15`
- Batch size per device: `24`
- Grad accumulation steps: `1`
- Action eval chunk size: `0`
- Target mode: `mixed`
- `soft_target_alpha=0.3`
- `soft_label_temperature=0.7`
- Best checkpoint metric: `val_macro_f1`, mode `max`
- Online evaluated checkpoint:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage2/checkpoints/epoch_15.pt
```

### Stage-2 Offline Validation Summary

| Selection | Epoch | Val Macro-F1 | Val Acc | Val Loss | Val Top-1 Regret | Val Expected Regret |
|---|---:|---:|---:|---:|---:|---:|
| Best macro-F1 | 10 | 0.702521 | 0.751163 | 0.810883 | 0.165348 | 0.292839 |
| Best validation accuracy | 8 | 0.702477 | 0.753488 | 0.807570 | 0.152993 | 0.308775 |
| Best top-1 cost regret | 8 | 0.702477 | 0.753488 | 0.807570 | 0.152993 | 0.308775 |
| Best expected cost regret | 11 | 0.697883 | 0.741860 | 0.810686 | 0.161089 | 0.283796 |
| Final / online-tested | 15 | 0.697034 | 0.739535 | 0.814027 | 0.173014 | 0.284640 |

### Stage-2 Epoch Table

| Epoch | Val F1 | Val Acc | Val Loss | Top1 Regret | Expected Regret | Train Acc | Train F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3817 | 0.4349 | 1.3925 | 0.8213 | 1.0483 | 0.4302 | 0.3515 |
| 2 | 0.6453 | 0.6907 | 0.9532 | 0.2803 | 0.5375 | 0.5538 | 0.4657 |
| 3 | 0.6388 | 0.7047 | 0.8819 | 0.2114 | 0.4020 | 0.6773 | 0.6165 |
| 4 | 0.6847 | 0.7395 | 0.8095 | 0.1775 | 0.3224 | 0.7903 | 0.7514 |
| 5 | 0.6954 | 0.7442 | 0.8099 | 0.1730 | 0.3571 | 0.8287 | 0.7982 |
| 6 | 0.6870 | 0.7326 | 0.8251 | 0.1740 | 0.3274 | 0.8612 | 0.8400 |
| 7 | 0.6806 | 0.7279 | 0.8188 | 0.1999 | 0.3247 | 0.8848 | 0.8674 |
| 8 | 0.7025 | 0.7535 | 0.8076 | 0.1530 | 0.3088 | 0.9057 | 0.8928 |
| 9 | 0.6864 | 0.7395 | 0.8148 | 0.1544 | 0.3007 | 0.9235 | 0.9138 |
| 10 | 0.7025 | 0.7512 | 0.8109 | 0.1653 | 0.2928 | 0.9366 | 0.9286 |
| 11 | 0.6979 | 0.7419 | 0.8107 | 0.1611 | 0.2838 | 0.9458 | 0.9392 |
| 12 | 0.7004 | 0.7465 | 0.8129 | 0.1665 | 0.2857 | 0.9509 | 0.9451 |
| 13 | 0.6878 | 0.7349 | 0.8171 | 0.1737 | 0.2869 | 0.9546 | 0.9492 |
| 14 | 0.6906 | 0.7372 | 0.8159 | 0.1710 | 0.2857 | 0.9583 | 0.9532 |
| 15 | 0.6970 | 0.7395 | 0.8140 | 0.1730 | 0.2846 | 0.9597 | 0.9551 |

Offline observations:

1. Validation macro-F1 peaks around epochs 8-10, with the numerical best at epoch 10.
2. Validation accuracy and top-1 cost regret both prefer epoch 8.
3. Expected cost regret prefers epoch 11.
4. Epoch 15 is close to the late plateau but is not the best checkpoint by any offline validation criterion listed above.
5. Training metrics continue improving through epoch 15, while validation metrics plateau after epoch 8-10. This suggests mild overfitting or checkpoint-selection mismatch.

## Online Test Conditions

- Online checkpoint: `epoch_15.pt`
- Online seed: `2024`
- Run counts: `3600`
- Prompt mode: `queue_wait_short`
- Deterministic mode: `false`
- `bf16=true`
- Online work directory:

```text
records/online_tests/repro_jinan3_qwshort_targeted_20260805_epoch15/qwen3vl_2b_fullmem/epoch_15/anon_3_4_jinan_real.json_08_27_00_03_56
```

- Online evaluator log:

```text
/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/logs/online_seed_2024.log
```

- Artifact hashes:
  - `projection_ckpt_sha256=68382b18e79e45c9a46284499775c251233fede429469c88606c3dabeb986bb7`
  - `online_evaluator_sha256=1ffe30b0759d1e1017117dc59ebb35aeedda1af266112fbabc767ecc6db08f75`

## Online Results

Lower is better for AQL, AWT, ATT, queued vehicles, p95 wait, and max wait. Higher is better for reward.

| Checkpoint | Reward | AQL | Queued | AWT | ATT | Wait P95 | Wait Max | Observed Wait Vehicles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch_15.pt` | -615.6274 | 165.7167 | 19886 | 49.1005 | 273.2653 | 197.60 | 779 | 4729 |

## Delta Versus Source Archive

Source archive metrics:

| Metric | Source Archive | 2B Fullmem Epoch 15 | Delta | Relative Change |
|---|---:|---:|---:|---:|
| Reward | -599.1446 | -615.6274 | -16.4828 | 2.75% more negative |
| Avg queue length | 157.1833 | 165.7167 | +8.5333 | +5.43% |
| Queued vehicle num | 18862 | 19886 | +1024 | +5.43% |
| Avg waiting time | 42.8476 | 49.1005 | +6.2529 | +14.59% |
| Avg travel time | 268.4078 | 273.2653 | +4.8575 | +1.81% |

Interpretation: the 2B fullmem epoch-15 online controller underperforms the 2026-08-05 archive target on every reported online metric. The largest relative degradation is average waiting time.

## Comparison Against Other Runs

| Run | Reward | AQL | Queued | AWT | ATT |
|---|---:|---:|---:|---:|---:|
| 2026-08-05 archive | -599.1446 | 157.1833 | 18862 | 42.8476 | 268.4078 |
| Qwen3-VL-2B fullmem epoch15 | -615.6274 | 165.7167 | 19886 | 49.1005 | 273.2653 |
| Qwen3-VL-4B epoch15 | -601.2091 | 158.6250 | 19035 | 46.2319 | 269.5698 |
| Main 4GPU epoch15 | -612.0425 | 164.6250 | 19755 | 45.3606 | 273.3319 |

### Qwen3-VL-2B Fullmem Versus Qwen3-VL-4B

| Metric | 2B Fullmem | 4B | 2B - 4B | Relative Difference |
|---|---:|---:|---:|---:|
| Reward | -615.6274 | -601.2091 | -14.4183 | 2.40% more negative |
| Avg queue length | 165.7167 | 158.6250 | +7.0917 | +4.47% |
| Queued vehicle num | 19886 | 19035 | +851 | +4.47% |
| Avg waiting time | 49.1005 | 46.2319 | +2.8685 | +6.20% |
| Avg travel time | 273.2653 | 269.5698 | +3.6955 | +1.37% |

Interpretation: Qwen3-VL-4B is clearly stronger than Qwen3-VL-2B under this recipe, especially on average waiting time and queue metrics.

## Key Findings

1. The Qwen3-VL-2B fullmem run completed successfully, but online performance is worse than both the source archive and the Qwen3-VL-4B epoch-15 run.
2. The most important failure mode is waiting time: AWT degrades by 14.59% versus the archive and by 6.20% versus Qwen3-VL-4B.
3. Offline classification quality is not catastrophic, with best validation macro-F1 around 0.7025, but online control quality still lags. This indicates that offline macro-F1 alone is insufficient for selecting a smaller backbone.
4. Epoch 15 is not the offline-best checkpoint. Epoch 8 is preferred by validation accuracy and top-1 cost regret; epoch 10 is preferred by macro-F1; epoch 11 is preferred by expected cost regret.
5. The fullmem run changed Stage-2 optimization dynamics relative to the default recipe: `batch_size=24, grad_accum=1` instead of the default `batch_size=4, grad_accum=4`. This makes the result both a 2B backbone ablation and a high-batch training variant.

## Recommended Follow-Up

1. Run online checkpoint sweep for Qwen3-VL-2B fullmem at epochs 8, 10, 11, and 15. This tests whether the poor online result is due to checkpoint selection or the 2B backbone itself.
2. Run a fair default-batch Qwen3-VL-2B Stage-2 recipe using `STAGE2_BATCH_SIZE=4` and `STAGE2_GRAD_ACCUM_STEPS=4`, reusing the same Stage-1 checkpoint. This separates backbone size from fullmem batch effects.
3. If compute budget allows, evaluate p95/max wait for the 4B and 2B checkpoints side by side under identical online seeds, because the 2B run has worse max wait (`779`) and p95 wait (`197.6`) than the 4B epoch-15 result (`617` max, `185.65` p95).
4. Treat the current evidence as supporting the claim that Qwen3-VL-4B is a materially better backbone than Qwen3-VL-2B for this Jinan real queue-wait-short targeted recipe.

## Artifact Index

```text
run_root=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710
run_info=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/logs/run.info
completion_status=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/logs/completion.status
stage1_best=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage1/best.pt
stage2_metrics=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage2/checkpoints/metrics.jsonl
stage2_epoch15=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage2/checkpoints/epoch_15.pt
stage2_best=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/stage2/checkpoints/best.pt
online_result=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/online_eval/seed_2024/result.json
online_summary=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/online_eval/summary.json
online_log=/home/code/LLMSTSC/records/pipeline_runs/ablation_qwen3vl_2b_jinan3_qwshort_targeted_fullmem_20260826_232710/logs/online_seed_2024.log
```
