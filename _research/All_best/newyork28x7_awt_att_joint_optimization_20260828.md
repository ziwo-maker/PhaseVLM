# NewYork 28x7 Joint AWT/ATT Optimization

Generated: 2026-08-28

## Purpose

This experiment improves average waiting time (AWT) and average travel time
(ATT) jointly on the NewYork 28x7 traffic flow. It uses the epoch-11 policy as
the matched baseline, generates wait-oriented counterfactual trigger data with
the phase override policy, and distills that data into three cost-weighted
behavior-cloning variants.

## Completion and Scope

- Experiment status: `success`, `complete`
- Flow: `anon_28_7_newyork_real_double.json`
- Evaluation protocol: same-flow, 3600-second episode, four online simulator
  seeds (`3031`, `3032`, `3033`, `3034`)
- Completed online results: phase 1 `20/20`, phase 2 `12/12`, collection
  `4/4`, phase 3 `12/12`
- Training variants: three, each trained for three epochs
- Error scan: no traceback, NCCL, CUDA OOM, or abnormal-exit match found

The four online seeds evaluate one trained checkpoint per variant. They are not
independent training repetitions.

## Main Result

Lower is better for Queue, AWT, ATT, aggregate sum, p95 wait, and maximum
wait. Higher is better for reward. Aggregate sum is `Queue + AWT + ATT`.
Values are mean +/- sample standard deviation across four online seeds.

| Policy | n | Reward | Queue | AWT | ATT | Aggregate sum | Wait p95 | Wait max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch_11` baseline | 4 | -7070.77 | 1731.05 +/- 77.50 | 246.15 +/- 8.54 | 998.80 +/- 10.10 | 2975.99 | 1437.76 +/- 55.18 | 2791.00 |
| `awt2_aql075_tail1_r005` | 4 | -6661.31 | 1559.81 +/- 12.76 | 214.84 +/- 8.99 | 941.82 +/- 4.07 | 2716.47 | 1400.05 +/- 25.60 | 2632.25 |
| `awt2_aql1_tail05_r003` | 4 | -6777.46 | 1651.80 +/- 84.99 | 212.87 +/- 34.26 | 972.09 +/- 26.79 | 2836.76 | 1321.55 +/- 163.17 | 2666.00 |
| `awt3_aql1_tail1_r003` | 4 | -6637.80 | 1521.81 +/- 51.19 | 202.26 +/- 13.86 | 927.48 +/- 14.67 | 2651.55 | 1207.21 +/- 37.90 | 2894.75 |

## Relative Change Versus Epoch-11

Positive values below are improvements, except the final Max wait column where
a positive value is degradation.

| Policy | Reward | Queue | AWT | ATT | Aggregate sum | Wait p95 | Max wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| `awt2_aql075_tail1_r005` | +5.79% | +9.89% | +12.72% | +5.70% | +8.72% | +2.62% | -5.69% |
| `awt2_aql1_tail05_r003` | +4.15% | +4.58% | +13.52% | +2.67% | +4.68% | +8.08% | -4.48% |
| `awt3_aql1_tail1_r003` | +6.12% | +12.09% | +17.83% | +7.14% | +10.90% | +16.04% | +3.72% |

## Recommendation

`awt3_aql1_tail1_r003` is the best **joint AWT/ATT** candidate in this run:
AWT decreases from `246.15` to `202.26` (-17.83%), ATT decreases from `998.80`
to `927.48` (-7.14%), and the aggregate sum decreases by 10.90%.

It is not yet the default deployment candidate because maximum wait increases
from `2791.00` to `2894.75` (+3.72%). Use
`awt2_aql075_tail1_r005` as the more conservative candidate when reducing
maximum wait and cross-seed variability matters: it improves both AWT and ATT
while reducing maximum wait by 5.69%.

## Override Finding

The strongest direct override, `w30_p10_g20`, sharply reduced AWT to `136.08`
(-44.72%) and p95 wait to `1035.58` (-27.97%), but increased Queue by 16.60%,
ATT by 5.16%, and the aggregate sum by 7.69%. The selector therefore marked it
as unaccepted. It was useful as a counterfactual trigger-data generator, not
as a final online policy.

## Training Variants

| Label | AWT cost | AQL cost | Tail-wait cost | Ranking loss |
|---|---:|---:|---:|---:|
| `awt2_aql1_tail05_r003` | 2.0 | 1.0 | 0.5 | 0.03 |
| `awt2_aql075_tail1_r005` | 2.0 | 0.75 | 1.0 | 0.05 |
| `awt3_aql1_tail1_r003` | 3.0 | 1.0 | 1.0 | 0.03 |

All variants begin from the epoch-11 policy and use the same augmented dataset:
the original behavior-cloning data plus 4,300 override-trigger rows, each
given a minimum sample weight of 3.0. The triggers originated from four
collection seeds; counts were 1,039, 1,076, 1,033, and 1,152.

## Interpretation Limits

1. The best online variant has worse maximum wait, so the result supports a
   mean-wait/travel-time improvement, not an unqualified tail-risk improvement.
2. Offline `val_expected_cost_regret` did not rank variants consistently with
   online aggregate sum: `awt3_aql1_tail1_r003` had the best online result but
   the largest offline regret. Online validation must remain in the selector.
3. Results apply only to the recorded NewYork 28x7 traffic flow and the stated
   simulator protocol; no demand-shift or cross-flow claim is supported.

## Artifact Index

```text
run_root=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry
status=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/status.env
baseline_summary=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/phase1_checkpoint_eval/summary.csv
override_summary=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/phase2_override_grid/summary.csv
override_selection=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/phase2_override_grid/selection.env
online_summary=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/phase3_online/summary.csv
best_joint_checkpoint=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/phase3_train/awt3_aql1_tail1_r003/checkpoints/best.pt
stable_checkpoint=/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry/phase3_train/awt2_aql075_tail1_r005/checkpoints/best.pt
```
