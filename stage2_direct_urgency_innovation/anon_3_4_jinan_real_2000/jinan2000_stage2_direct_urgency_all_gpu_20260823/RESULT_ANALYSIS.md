# Jinan-2000 Direct-Urgency All-GPU Result Analysis

## Executive Conclusion

The Jinan-2000 direct-urgency campaign completed successfully. It used a
three-flow Stage-1 initializer, then trained and evaluated Stage 2 on the same
Jinan-2000 flow, `anon_3_4_jinan_real_2000.json`.

The strongest average-objective result is
`constrained + wait225_mixed_rank`. On the primary online seed `2024`, averaged
over three training seeds, it achieves the best Reward, AQL, and ATT among the
evaluated cells, while also giving a large P95 waiting-time gain relative to
hard BC.

The result should be claimed as an average-efficiency and P95 improvement, not
as a maximum-wait solution. Maximum wait remains better for the `wait2` branch,
especially `offline_val_acc + wait2_mixed_rank`.

## Scope And Protocol

| Item | Value |
|---|---|
| Run root | `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312` |
| Status | `success` |
| Stage-2 training cells | `21 / 21` |
| Final online rows | `135 / 135` |
| Incomplete rows | `0` |
| Training flow | `anon_3_4_jinan_real_2000.json` |
| Evaluation flow | `anon_3_4_jinan_real_2000.json` |
| Flow protocol | same-flow Jinan-2000 |
| Stage-1 data mode | three-flow initializer |
| Stage-1 data | `jinan_real`, `jinan_real_2000`, `jinan_real_2500` |
| Stage-2 data | direct-urgency version of `jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl` |
| Primary online seed | `2024` |
| Descriptive online seeds | `42`, `123` |
| Training seeds | `2024`, `42`, `123` |
| Final online horizon | 3,600 seconds |
| Short selector horizon | 1,200 seconds |
| GPUs | `0,1,2` |

Important boundary: all reported conclusions are same-flow Jinan-2000
closed-loop results. The online seeds `42` and `123` are descriptive repeats,
not independent policy samples. The primary independent unit remains the
training seed.

## Shell And Launcher

The campaign launcher shell script used for this run is:

`/home/code/LLMSTSC/scripts/run_jinan2000_stage2_direct_urgency_all_gpu_campaign.sh`

The script shebang is:

`#!/usr/bin/env bash`

On this machine, `bash` resolves to:

`/usr/bin/bash`

The run was launched detached with `setsid` so the long experiment process
survived the launching shell and wrote its state under the run root above.

## Raw Primary Online Summary

Primary online seed: `2024`. Values are mean +/- sample standard deviation over
three training seeds. Reward is better when higher; AQL, AWT, ATT, P95 wait,
and max wait are better when lower.

| Selector | Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `constrained` | `wait225_mixed_rank` | 3 | **-599.592 +/- 2.530** | **110.922 +/- 0.494** | 44.387 +/- 0.597 | **273.066 +/- 0.405** | 161.417 +/- 5.725 | 745.333 +/- 68.923 |
| `short_online` | `wait225_mixed_rank` | 3 | -600.898 +/- 3.342 | 111.328 +/- 1.060 | 45.282 +/- 0.911 | 273.785 +/- 0.889 | 168.400 +/- 5.973 | 872.000 +/- 245.514 |
| `short_online_seed42` | `wait225_mixed_rank` | 3 | -600.898 +/- 3.342 | 111.328 +/- 1.060 | 45.282 +/- 0.911 | 273.785 +/- 0.889 | 168.400 +/- 5.973 | 872.000 +/- 245.514 |
| `tail_risk` | `wait225_mixed_rank` | 3 | -600.898 +/- 3.342 | 111.328 +/- 1.060 | 45.282 +/- 0.911 | 273.785 +/- 0.889 | 168.400 +/- 5.973 | 872.000 +/- 245.514 |
| `tail_risk` | `wait2_mixed_rank` | 3 | -601.822 +/- 6.514 | 111.494 +/- 1.313 | 44.464 +/- 2.220 | 273.651 +/- 1.051 | 164.333 +/- 1.155 | 738.667 +/- 186.004 |
| `constrained` | `wait2_mixed_rank` | 3 | -603.831 +/- 6.806 | 111.617 +/- 1.334 | **44.003 +/- 2.157** | 273.660 +/- 1.047 | 163.600 +/- 1.217 | 632.667 +/- 189.951 |
| `short_online` | `wait2_mixed_rank` | 3 | -603.831 +/- 6.806 | 111.617 +/- 1.334 | **44.003 +/- 2.157** | 273.660 +/- 1.047 | 163.600 +/- 1.217 | 632.667 +/- 189.951 |
| `short_online_seed42` | `wait2_mixed_rank` | 3 | -603.831 +/- 6.806 | 111.617 +/- 1.334 | **44.003 +/- 2.157** | 273.660 +/- 1.047 | 163.600 +/- 1.217 | 632.667 +/- 189.951 |
| `offline_val_acc` | `wait25_conservative_rank` | 3 | -607.603 +/- 5.812 | 112.783 +/- 2.718 | 47.840 +/- 2.995 | 274.306 +/- 2.365 | 184.950 +/- 16.070 | 720.333 +/- 72.418 |
| `offline_val_acc` | `wait2_mixed_rank` | 3 | -608.485 +/- 5.343 | 113.117 +/- 1.471 | 46.974 +/- 4.252 | 275.088 +/- 1.567 | 179.133 +/- 14.670 | **610.333 +/- 126.065** |
| `offline_val_acc` | `wait25_strong_rank` | 3 | -608.781 +/- 7.829 | 114.669 +/- 3.063 | 44.834 +/- 4.656 | 276.675 +/- 2.989 | **159.867 +/- 6.568** | 801.667 +/- 304.484 |
| `offline_val_acc` | `wait225_mixed_rank` | 3 | -608.934 +/- 3.399 | 113.183 +/- 0.983 | 45.951 +/- 2.053 | 275.131 +/- 0.760 | 174.917 +/- 7.434 | 777.000 +/- 47.286 |
| `offline_val_acc` | `equal_mixed_rank` | 3 | -610.662 +/- 5.033 | 113.883 +/- 1.507 | 50.902 +/- 2.471 | 275.654 +/- 1.236 | 197.083 +/- 16.016 | 976.000 +/- 161.391 |
| `offline_val_acc` | `wait25_mixed_rank` | 3 | -612.933 +/- 1.450 | 115.136 +/- 0.420 | 48.712 +/- 0.340 | 276.756 +/- 1.111 | 173.750 +/- 10.123 | 1052.000 +/- 70.150 |
| `offline_val_acc` | `hard_bc` | 3 | -617.567 +/- 2.457 | 116.322 +/- 1.285 | 56.443 +/- 0.962 | 277.606 +/- 0.856 | 218.233 +/- 6.242 | 777.667 +/- 108.288 |

## Main Comparison Against Hard BC

Relative to `offline_val_acc + hard_bc`, every direct-urgency arm improves
primary Reward in all three training seeds.

The largest average-objective gain is from
`constrained + wait225_mixed_rank`:

| Metric | Benefit vs hard BC | Interpretation |
|---|---:|---|
| Reward | +17.975 +/- 3.881 | Higher reward, wins 3/3 seeds |
| AQL | +5.400 | Lower queue length |
| AWT | +12.055 | Lower average waiting time |
| ATT | +4.540 | Lower average travel time |
| P95 wait | +56.817 | Much lower P95 observed wait |
| Max wait | +32.333 | Slightly lower max wait on average |

This is the cleanest Jinan-2000 average-objective result in the current matrix.

## Selector Findings

For `wait225_mixed_rank`, constrained selection improves over short-online on
all six primary metrics:

| Metric | `constrained + wait225` | `short_online + wait225` |
|---|---:|---:|
| Reward | -599.592 | -600.898 |
| AQL | 110.922 | 111.328 |
| AWT | 44.387 | 45.282 |
| ATT | 273.066 | 273.785 |
| P95 wait | 161.417 | 168.400 |
| Max wait | 745.333 | 872.000 |

For `wait2_mixed_rank`, constrained and short-online select the same
checkpoints. Tail-risk changes only training seed `123`, improving Reward but
worsening AWT, P95, and maximum wait relative to constrained/short-online.

Changing the short-selector seed from `2024` to `42` changes none of the six
selected arm/seed checkpoints. The final metrics for `short_online_seed42` are
therefore identical to `short_online`.

## Branch Decisions

1. **Average-objective main line:** keep
   `constrained + wait225_mixed_rank`. It has the best Reward, AQL, and ATT,
   and a strong P95 reduction.
2. **AWT branch:** `constrained + wait2_mixed_rank` has the best AWT at
   `44.003 +/- 2.157`.
3. **Maximum-wait branch:** `offline_val_acc + wait2_mixed_rank` has the lowest
   maximum wait at `610.333 +/- 126.065`.
4. **P95-only branch:** `offline_val_acc + wait25_strong_rank` has the lowest
   P95 at `159.867 +/- 6.568`, but it has worse Reward/AQL/ATT and high
   maximum-wait variance, so it should not be promoted as a main-line policy.

## Interpretation

This Jinan-2000 result is stronger than the earlier Jinan-real result for the
average-objective claim. On the same-flow Jinan-2000 protocol, direct urgency
substantially improves reward, queue length, average waiting time, travel time,
and P95 observed vehicle wait relative to hard BC.

The maximum-wait story remains separate. The best average-objective policy is
not the best maximum-wait policy. Therefore the defensible claim is:

> On `anon_3_4_jinan_real_2000.json`, direct phase-urgency ranking with a
> three-flow Stage-1 initializer and constrained online checkpoint selection
> makes `wait225_mixed_rank` the strongest average-objective candidate,
> improving reward, queue, average wait, travel time, and P95 wait over hard BC.
> It does not establish a hard maximum-wait guarantee.

## Suggested Next Experiments

1. Freeze `constrained + wait225_mixed_rank` as the Jinan-2000
   average-objective candidate.
2. If maximum wait is a target, compare only `constrained + wait225_mixed_rank`
   and `offline_val_acc + wait2_mixed_rank` after adding an explicit
   vehicle-level max-wait protection mechanism.
3. Do not expand `wait25_strong_rank` as a main line. It has an attractive P95
   mean but worse average metrics and unstable maximum wait.
4. Run matched confirmation on `jinan_real` or `jinan_2500` only if the paper
   needs a cross-flow consistency statement. Otherwise, report this as
   same-flow Jinan-2000 evidence.

## Source Artifacts

- Campaign status:
  `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312/campaign.status`
- Aggregate:
  `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312/aggregate.json`
- Aggregate Markdown:
  `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312/AGGREGATE.md`
- Raw online rows:
  `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312/online_results.csv`
- Incomplete audit:
  `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312/incomplete_runs.csv`
- Provenance:
  `records/pipeline_runs/jinan2000_stage2_direct_urgency_all_gpu_20260823_154312/provenance.env`
- Campaign script:
  `scripts/run_jinan2000_stage2_direct_urgency_all_gpu_campaign.sh`
