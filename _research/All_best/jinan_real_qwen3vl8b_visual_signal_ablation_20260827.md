# Jinan Real Qwen3-VL-8B Visual-Signal Ablation

Generated: 2026-08-27

## Purpose

This experiment tests whether the trained Qwen3-VL-8B traffic controller uses
state-dependent visual information or relies only on the compact textual
queue/wait prompt. The same Stage-2 checkpoint, model, compact prompt, traffic
flow, and online evaluation settings are held fixed. Only the visual signal is
changed at inference time.

The experiment is an inference-time causal intervention on one trained
checkpoint. It is not a comparison of independently retrained visual and
text-only controllers.

## Completion Status

- Experiment status: complete
- Online runs: 16/16 completed
- Failed runs: 0
- Missing result files: 0
- Traceback, CUDA OOM, exception, or NaN matches in experiment logs: 0
- Completion time: `2026-08-27T00:47:24+08:00`

Completion record:

```text
/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/completion.status
```

## Controlled Experiment Configuration

The following settings are identical for all 16 runs:

| Field | Value |
|---|---|
| Backbone | Qwen3-VL-8B |
| Model path | `/home/data/model/qwen-vl-8B` |
| Stage-2 checkpoint | `repro_jinan3_qwshort_targeted_4gpu_20260825_182642/stage2/checkpoints/epoch_15.pt` |
| Checkpoint SHA256 | `233fb2e382d021722ce98b0602738bbbeedfc755f86226ab29035eeeb3fb5be5` |
| Dataset | `jinan` |
| Traffic flow | `anon_3_4_jinan_real.json` |
| Same-flow evaluation | yes |
| Prompt mode | `queue_wait_short` |
| Episode | 3600 s |
| Decision interval | 30 s |
| Sequence length | 16 |
| Image size | 224 |
| Input dimension | 8 |
| Hidden dimension | 64 |
| Prefix tokens | 8 |
| Prefix mode | `prefix` |
| Maximum text length | 64 |
| Precision | bf16 |
| Deterministic PyTorch mode | enabled |
| Online seeds | 2024, 42, 123, 7 |

The four online seeds evaluate the same trained checkpoint. They are simulator
seeds, not independent training repetitions.

## Visual Interventions

| Arm | Intervention |
|---|---|
| `full` | Use the correct TS2I image for each intersection and prompt. |
| `shuffled_image` | Cyclically shift images within each 12-intersection inference batch, so every prompt receives another intersection's real TS2I image. |
| `constant_image` | Replace every TS2I image with the same all-zero image before the frozen vision tower. |
| `zero_prefix` | Preserve the eight visual-prefix positions but replace all projected visual-prefix values with zeros. |

`shuffled_image` is the cleanest state-dependence test because it preserves the
real TS2I image distribution while breaking the image-to-intersection pairing.
`constant_image` and `zero_prefix` are stronger out-of-distribution dependency
tests.

## Aggregate Online Results

Lower is better for AQL, AWT, ATT, queued vehicles, wait p95, and wait max.
Higher is better for reward.

| Visual arm | n | Reward | AQL | AWT | ATT | Queued | Wait p95 | Wait max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 4 | -617.7951 +/- 0.9760 | 169.7021 +/- 0.9708 | 46.1883 +/- 0.4276 | 276.5120 +/- 0.7079 | 20364.25 +/- 116.50 | 173.55 +/- 3.10 | 701.00 +/- 0.00 |
| `shuffled_image` | 4 | -647.2461 +/- 0.0385 | 189.9625 +/- 0.0048 | 47.7952 +/- 0.0001 | 289.2199 +/- 0.0018 | 22795.50 +/- 0.58 | 196.30 +/- 0.00 | 632.00 +/- 0.00 |
| `constant_image` | 4 | -893.7012 +/- 0.0000 | 417.0250 +/- 0.0000 | 236.4576 +/- 0.0000 | 412.1984 +/- 0.0000 | 50043.00 +/- 0.00 | 954.00 +/- 0.00 | 2035.00 +/- 0.00 |
| `zero_prefix` | 4 | -1333.8764 +/- 0.0000 | 1729.2000 +/- 0.0000 | 1019.2841 +/- 0.0000 | 1579.1160 +/- 0.0000 | 207504.00 +/- 0.00 | 3431.00 +/- 0.00 | 3561.00 +/- 0.00 |

Values are mean +/- sample standard deviation across the four online seeds.

## Relative Change Versus Full Visual Input

Positive values denote degradation. For reward, the percentage measures the
increase in absolute negative reward magnitude.

| Arm | Reward worse | AQL worse | AWT worse | ATT worse | Queued worse | Wait p95 worse | Wait max change |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shuffled_image` | +4.77% | +11.94% | +3.48% | +4.60% | +11.94% | +13.11% | -9.84% |
| `constant_image` | +44.66% | +145.74% | +411.94% | +49.07% | +145.74% | +449.70% | +190.30% |
| `zero_prefix` | +115.91% | +918.96% | +2106.80% | +471.08% | +918.96% | +1876.95% | +407.99% |

The shuffled-image maximum wait is lower than Full despite worse reward, AQL,
AWT, ATT, queued count, and wait p95. Therefore maximum wait alone is not a
monotonic summary of overall controller quality in this experiment.

## Per-Seed Primary Metrics

| Arm | Seed | Reward | AQL | AWT | ATT |
|---|---:|---:|---:|---:|---:|
| `full` | 7 | -619.2591 | 171.1583 | 46.8297 | 277.5739 |
| `full` | 42 | -617.3071 | 169.2167 | 45.9745 | 276.1581 |
| `full` | 123 | -617.3071 | 169.2167 | 45.9745 | 276.1581 |
| `full` | 2024 | -617.3071 | 169.2167 | 45.9745 | 276.1581 |
| `shuffled_image` | 7 | -647.2794 | 189.9667 | 47.7951 | 289.2214 |
| `shuffled_image` | 42 | -647.2128 | 189.9583 | 47.7952 | 289.2183 |
| `shuffled_image` | 123 | -647.2794 | 189.9667 | 47.7952 | 289.2214 |
| `shuffled_image` | 2024 | -647.2128 | 189.9583 | 47.7952 | 289.2184 |
| `constant_image` | 7 | -893.7012 | 417.0250 | 236.4576 | 412.1984 |
| `constant_image` | 42 | -893.7012 | 417.0250 | 236.4576 | 412.1984 |
| `constant_image` | 123 | -893.7012 | 417.0250 | 236.4576 | 412.1984 |
| `constant_image` | 2024 | -893.7012 | 417.0250 | 236.4576 | 412.1984 |
| `zero_prefix` | 7 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 |
| `zero_prefix` | 42 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 |
| `zero_prefix` | 123 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 |
| `zero_prefix` | 2024 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 |

## Action-Distribution Diagnosis

The following counts use seed 2024 and cover 1440 intersection decisions per
arm.

| Arm | ETWT | NTST | ELWL | NLSL | Agreement with Full trajectory |
|---|---:|---:|---:|---:|---:|
| `full` | 525 | 490 | 221 | 204 | 100.00% |
| `shuffled_image` | 529 | 506 | 198 | 207 | 35.07% |
| `constant_image` | 870 | 377 | 54 | 139 | 32.92% |
| `zero_prefix` | 0 | 1 | 0 | 1439 | 14.24% |

`zero_prefix` collapses almost completely to NLSL, while `constant_image`
strongly biases the controller toward ETWT. These are out-of-distribution
failure modes, not calibrated estimates of the percentage contribution of
vision. Since the experiment is closed loop, action agreement after the first
divergent action also reflects different subsequent traffic states.

## Result-to-Claim Assessment

- `claim_supported`: yes, within the tested checkpoint and traffic flow
- `confidence`: medium-high for state-dependent visual use
- `integrity_status`: provisional; no separate `EXPERIMENT_AUDIT.json` was available

### What the Results Support

1. The trained Qwen3-VL-8B controller uses state-dependent visual information.
2. Correct correspondence between an intersection's TS2I image and compact
   prompt materially affects closed-loop control.
3. Visual information contributes most clearly to queue-related metrics:
   shuffling images worsens AQL and queued vehicles by 11.94% and wait p95 by
   13.11%.
4. Compact textual queue/wait cues provide partial compensation because the
   shuffled-image arm degrades much less than the constant-image and
   zero-prefix arms.

### What the Results Do Not Support

1. They do not prove that a pretrained VLM vision tower is better than a
   parameter-matched direct numeric encoder.
2. They do not isolate generic visual pretraining from the benefit of adding a
   high-capacity structured state-encoding path.
3. `zero_prefix` is not a retrained text-only controller and must not be
   reported as one.
4. The four online seeds are not independent training seeds, so they do not
   establish robustness across trained checkpoints.
5. The result is limited to one Jinan flow and should not be generalized to
   other flows, road networks, lane configurations, or phase topologies.

## Important Baseline Note

The current deterministic Full seed-2024 result is:

```text
Reward=-617.3071, AQL=169.2167, AWT=45.9745, ATT=276.1581
```

The older online record for the same checkpoint reports:

```text
Reward=-612.0425, AQL=164.6250, AWT=45.3606, ATT=273.3319
```

The new experiment explicitly enables deterministic PyTorch settings and
records current evaluator hashes; the older result does not record equivalent
determinism or evaluator provenance. All visual-ablation claims must therefore
use the new `full` arm as the matched baseline and must not mix the older Full
metrics into the ablation table.

## Paper-Ready Finding

Holding the Qwen3-VL-8B checkpoint, compact prompt, traffic flow, and inference
configuration fixed, cyclically mismatching TS2I images across intersections
increased AQL by 11.94%, ATT by 4.60%, AWT by 3.48%, and wait p95 by 13.11%.
Constant-image and zero-prefix interventions caused severe policy collapse.
These results show that the trained controller uses state-dependent visual
information rather than relying only on compact textual cues, while the
smaller shuffled-image degradation indicates that the textual cues provide
complementary control evidence.

## Recommended Follow-Up

1. Retrain a compact-text-only 8B controller under the same data, optimization,
   and checkpoint-selection protocol. This avoids the out-of-distribution
   interpretation problem of `zero_prefix`.
2. Compare TS2I plus the frozen vision tower against a parameter-matched direct
   numeric-to-prefix encoder to test whether visual pretraining is necessary.
3. Repeat Full and shuffled-image interventions across independent training
   seeds and the other paper scenarios.
4. Use a newly sampled image derangement at every decision or shuffle images
   across time as an additional control for correlated neighboring traffic.

## Artifact Index

```text
run_root=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728
manifest=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/manifest.json
completion=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/completion.status
aggregate_json=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/aggregate.json
aggregate_csv=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/aggregate.csv
summary_md=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/RESULTS.md
per_run_results=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/results/<arm>/seed_<seed>/result.json
per_run_status=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/status/<arm>_seed_<seed>.status
logs=/home/code/LLMSTSC/records/pipeline_runs/jinan3_8b_visual_signal_ablation_20260827_003728/logs
```
