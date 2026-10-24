# Jinan_real Wait25 Refinement Result Analysis

## Executive Conclusion

The `jinan_real` wait25-refinement campaign completed successfully. None of
the three follow-up arms satisfies the pre-registered promotion criterion:
relative to `wait2_mixed_rank`, it must improve the same-flow primary result
without a material AQL/ATT regression and must improve either P95 waiting time
or maximum waiting time consistently across the three training seeds.

`wait225_mixed_rank` is the best overall trade-off among the new arms: it has
the best mean reward and improves mean AQL and ATT relative to both existing
wait-weighted baselines. However, it does not improve P95 waiting time relative
to `wait2_mixed_rank`, and its maximum-wait improvement is not seed-stable.
`wait25_strong_rank` is materially unstable and should not be expanded.

The correct current research claim is therefore:

> On `anon_3_4_jinan_real.json`, moderating the static AWT coefficient from
> 2.5 to 2.25 improves the average reward/queue/travel trade-off, but does not
> establish a tail-wait improvement. Increasing soft-target and ranking
> strength reduces robustness.

## Scope And Protocol

| Item | Value |
|---|---|
| Training flow | `anon_3_4_jinan_real.json` |
| Evaluation flow | `anon_3_4_jinan_real.json` |
| Online horizon | 3,600 seconds |
| Training seeds | `2024`, `42`, `123` |
| Primary online realization seed | `2024` |
| Descriptive online realization seeds | `42`, `123` |
| Online evaluation | Deterministic, no phase override |
| Source data | Direct-Urgency frozen r3 dataset |
| Stage-1 checkpoint | Same checkpoint as r3 |
| Completed rows | 27 / 27 |

All primary comparisons use exactly three independent training seeds at online
seed `2024`. The online realization seeds `42` and `123` are repeat
realizations of the same trained policies, not independent policy samples.

## Artifact Integrity

- Campaign status: `success`, completed at `2026-08-22T12:34:21+08:00`.
- Every arm/seed has a successful training status, a hash-verified `best.pt`,
  and a successful deterministic online evaluation status.
- `incomplete_runs.csv` contains no incomplete rows.
- No traceback, error, NaN, out-of-memory, killed-process, or segmentation
  fault keyword occurred in the campaign logs.
- The new campaign and r3 baseline have identical SHA-256 hashes for
  `train_projection_bc.py`, `run_projection_online.py`, `cost_ranking.py`,
  `grouped_dataset.py`, and `data_split.py`.
- All 27 online result rows use only `anon_3_4_jinan_real.json`; no other
  Jinan flow is part of this campaign.

## Arms

| Arm | AQL weight | AWT weight | Soft-target alpha | Ranking weight | Controlled purpose |
|---|---:|---:|---:|---:|---|
| `wait225_mixed_rank` | 1.00 | 2.25 | 0.30 | 0.05 | Local interpolation between wait 2 and 2.5 |
| `wait25_conservative_rank` | 1.00 | 2.50 | 0.15 | 0.02 | Weaker soft/ranking guidance |
| `wait25_strong_rank` | 1.00 | 2.50 | 0.45 | 0.08 | Stronger soft/ranking guidance |

The fixed r3 comparators are `wait2_mixed_rank` and `wait25_mixed_rank`.

## Primary Same-Flow Results

The following statistics are mean +/- sample standard deviation over the three
training seeds, using the primary deterministic online seed `2024`. Reward is
higher-is-better; all other metrics are lower-is-better.

| Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---:|---:|---:|---:|---:|---:|
| `wait2_mixed_rank` | -608.331 +/- 3.234 | 162.489 +/- 0.604 | 45.533 +/- 2.849 | 272.296 +/- 0.312 | **177.467 +/- 7.583** | 755.333 +/- 254.406 |
| `wait25_mixed_rank` | -609.542 +/- 1.686 | 162.231 +/- 0.340 | 45.703 +/- 1.348 | 271.702 +/- 0.492 | 182.667 +/- 7.095 | 698.333 +/- 106.702 |
| `wait225_mixed_rank` | **-606.327 +/- 6.054** | 161.097 +/- 2.333 | 45.655 +/- 1.524 | 271.127 +/- 1.205 | 182.000 +/- 6.083 | 715.000 +/- 89.062 |
| `wait25_conservative_rank` | -607.903 +/- 6.099 | **160.692 +/- 4.312** | 48.395 +/- 0.653 | **270.417 +/- 2.811** | 193.333 +/- 6.807 | **699.333 +/- 59.138** |
| `wait25_strong_rank` | -624.002 +/- 25.273 | 175.133 +/- 19.344 | 53.168 +/- 18.908 | 279.984 +/- 11.545 | 193.667 +/- 48.211 | 813.333 +/- 298.172 |

## Per-Seed Primary Results

Each cell contains `Reward / AQL / AWT / ATT / P95 / Max` for the primary
online seed `2024`.

| Arm | Training seed 2024 | Training seed 42 | Training seed 123 |
|---|---|---|---|
| `wait225_mixed_rank` | -599.667 / 158.775 / 47.323 / 270.379 / 185 / 791 | -607.816 / 161.075 / 44.333 / 270.486 / 175 / 617 | -611.497 / 163.442 / 45.308 / 272.517 / 186 / 737 |
| `wait25_conservative_rank` | -609.009 / 161.000 / 47.899 / 270.468 / 191 / 704 | -601.326 / 156.233 / 48.151 / 267.580 / 188 / 638 | -613.373 / 164.842 / 49.135 / 273.201 / 201 / 756 |
| `wait25_strong_rank` | -610.644 / 163.658 / 44.312 / 272.886 / 177 / 613 | -653.151 / 197.467 / 74.879 / 293.305 / 248 / 1156 | -608.212 / 164.275 / 40.314 / 273.760 / 156 / 671 |

## Paired Comparison Against Wait2

Positive values favor the candidate. For reward, the benefit is candidate
minus baseline; for lower-is-better metrics, it is baseline minus candidate.
The bracket is a two-sided 95% t interval with `n=3`, so it is necessarily
wide and should not be used to claim significance.

| Candidate | Metric | Mean paired benefit | 95% interval | Seeds favoring candidate |
|---|---|---:|---:|---:|
| `wait225_mixed_rank` | Reward | +2.004 | [-10.336, +14.344] | 2 / 3 |
|  | AQL | +1.392 | [-3.257, +6.041] | 2 / 3 |
|  | AWT | -0.121 | [-8.824, +8.582] | 1 / 3 |
|  | ATT | +1.168 | [-1.928, +4.265] | 2 / 3 |
|  | P95 wait | -4.533 | [-22.474, +13.407] | 1 / 3 |
|  | Max wait | +40.333 | [-588.623, +669.290] | 1 / 3 |
| `wait25_conservative_rank` | Reward | +0.428 | [-8.282, +9.138] | 1 / 3 |
|  | AQL | +1.797 | [-7.617, +11.212] | 2 / 3 |
|  | AWT | -2.861 | [-8.351, +2.628] | 0 / 3 |
|  | ATT | +1.879 | [-4.713, +8.471] | 2 / 3 |
|  | P95 wait | -15.867 | [-22.393, -9.340] | 0 / 3 |
|  | Max wait | +56.000 | [-463.262, +575.262] | 1 / 3 |
| `wait25_strong_rank` | Reward | -15.672 | [-84.419, +53.076] | 1 / 3 |
|  | AQL | -12.644 | [-61.569, +36.280] | 0 / 3 |
|  | AWT | -7.635 | [-59.019, +43.750] | 1 / 3 |
|  | ATT | -7.688 | [-37.071, +21.694] | 0 / 3 |
|  | P95 wait | -16.200 | [-147.734, +115.334] | 1 / 3 |
|  | Max wait | -58.000 | [-1204.023, +1088.023] | 1 / 3 |

### Wait225 Versus Existing Wait25

`wait225_mixed_rank` versus `wait25_mixed_rank` has mean paired benefits of
`+3.215` reward, `+1.133` AQL, `+0.048` AWT, `+0.575` ATT, `+0.667` P95 wait,
and `-16.667` max wait. It wins on reward, AQL, AWT, and ATT in two of the
three seed-pairs. This supports a local interpolation benefit, not a reliable
tail-wait benefit.

## Offline Training Diagnostics

The selected checkpoints have the following mean best-validation metrics.

| Arm | Best val. loss | Best val. accuracy | Top-1 cost regret | Expected cost regret |
|---|---:|---:|---:|---:|
| `wait225_mixed_rank` | 0.7903 +/- 0.0199 | 0.7894 +/- 0.0106 | 0.7040 +/- 0.0158 | **0.7963 +/- 0.0242** |
| `wait25_conservative_rank` | **0.6907 +/- 0.0203** | **0.7963 +/- 0.0145** | 0.8550 +/- 0.0279 | 0.9209 +/- 0.0499 |
| `wait25_strong_rank` | 0.9178 +/- 0.1434 | 0.6806 +/- 0.0069 | 0.6992 +/- 0.2150 | 0.9217 +/- 0.2571 |

The strong-ranking arm has substantially lower validation accuracy. Its low
mean top-1 regret is not sufficient to produce robust online behavior; this is
consistent with over-emphasizing the ranking objective at the expense of
action-policy fidelity.

## Robustness And Checkpoint-Selection Diagnostic

`wait25_strong_rank`, training seed `42`, is the decisive failure case:

| Metric | Strong rank seed 42 | Wait2 seed 42 |
|---|---:|---:|
| Reward | -653.151 | -605.796 |
| AQL | 197.467 | 162.083 |
| AWT | 74.879 | 43.930 |
| ATT | 293.305 | 271.977 |
| P95 wait | 248 | 174.1 |
| Max wait | 1156 | 615 |

The selected strong-rank seed-42 checkpoint is epoch 2 because it first reached
`val_acc=0.6875`. Epoch 8 had the same validation accuracy but lower validation
loss and lower top-1/expected cost regret. The current checkpoint selection
accepts only strictly better `val_acc`, so an accuracy tie cannot update
`best.pt`. This may amplify this specific online failure, but it does not make
the strong-ranking direction viable: every strong-ranking seed has notably
lower offline validation accuracy than the other two new arms, and its online
mean/variance are both poor.

## Online Realization-Seed Check

For a fixed arm and training seed, the maximum observed range across online
seeds `2024`, `42`, and `123` is:

| Metric | Maximum range | Mean range |
|---|---:|---:|
| Reward | 1.754618 | 0.301734 |
| AQL | 0.508333 | 0.067778 |
| AWT | 0.445036 | 0.037426 |
| ATT | 0.284670 | 0.050410 |
| P95 wait | 0.200000 | 0.023333 |
| Max wait | 9.000000 | 0.600000 |

These ranges are much smaller than the cross-training-seed variation. This
validates treating the three training seeds, rather than the 27 rows, as the
independent unit for primary conclusions.

## Decision

1. Keep `wait2_mixed_rank` as the primary wait-sensitive reference. No new arm
   has earned promotion as a tail-wait improvement.
2. Treat `wait225_mixed_rank` as the best average-objective trade-off candidate
   for future confirmation, not as proof of better tail fairness.
3. Reject `wait25_conservative_rank` for a waiting-time objective: P95 wait is
   worse in all three paired seeds and AWT is worse in all three paired seeds.
4. Stop expanding `wait25_strong_rank`: its seed-42 collapse, high variance,
   and reduced offline accuracy constitute a robustness failure.
5. Do not continue a broad static-AWT sweep. The current evidence is a local
   trade-off, not a monotonic AWT-weight trend.

## Proposed Next Experiments (Not Executed)

1. Evaluate the existing `wait25_strong_rank / seed_42 / epoch_8.pt` once under
   the same `jinan_real` protocol. This isolates whether the strict-tie
   checkpoint rule materially amplified the observed failure.
2. If tail improvement remains the objective, augment the teacher/candidate
   construction with action-specific tail-wait information before designing a
   tail-aware ranking loss. Existing metadata contains `att`, `aql`, `awt`, and
   `spillback`, but not per-action P95 or maximum waiting time.
3. If the average-objective trade-off is the target, add training seeds only to
   `wait2_mixed_rank` and `wait225_mixed_rank` under the same flow and protocol.
   This is the smallest experiment that can reduce uncertainty around their
   paired differences.

## Source Artifacts

- New campaign aggregate: `records/pipeline_runs/jinan_real_wait25_improvement_20260822/AGGREGATE.md`
- New campaign raw online rows: `records/pipeline_runs/jinan_real_wait25_improvement_20260822/online_results.csv`
- New campaign provenance: `records/pipeline_runs/jinan_real_wait25_improvement_20260822/provenance.env`
- Frozen r3 baseline: `records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/WAIT25_IMPROVEMENT_BASELINE.md`
- r3 raw online rows: `records/pipeline_runs/jinan_real_direct_urgency_20260821_r3/online_results.csv`
