# Jinan Online Checkpoint Selection Result Analysis

## Executive Conclusion

The Jinan online-checkpoint-selection campaign completed successfully. It
evaluated a short-online selector that ranks saved epoch checkpoints by
1,200-second same-flow online reward, then promotes the selected checkpoint for
the 3,600-second same-flow online test.

Within this campaign, `wait225_mixed_rank` is the better main-line arm than
`wait2_mixed_rank`: it improves mean reward, AQL, AWT, ATT, and P95 waiting
time over three training seeds. The improvement is clearest for AWT, where
`wait225_mixed_rank` wins all three paired training seeds. However, it worsens
maximum observed vehicle wait on average, so this result should not be claimed
as a max-wait or hard tail-wait improvement.

The key checkpoint-selection finding is that the short-online selector chose a
different epoch than offline `val_acc` in all six arm/seed cells. This supports
the hypothesis that pure offline `val_acc` is not aligned enough with the
closed-loop traffic-control objective.

## Scope And Protocol

| Item | Value |
|---|---|
| Run root | `records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2` |
| Status | `success` |
| Completed online rows | `18 / 18` |
| Incomplete rows | `0` |
| Training flow | `anon_3_4_jinan_real.json` |
| Evaluation flow | `anon_3_4_jinan_real.json` |
| Short selector | 1,200-second online reward |
| Short selector seed | `2024` |
| Final online horizon | 3,600 seconds |
| Primary final online seed | `2024` |
| Descriptive final online seeds | `42`, `123` |
| Training seeds | `2024`, `42`, `123` |
| Arms | `wait2_mixed_rank`, `wait225_mixed_rank` |
| Stage-1 checkpoint | `records/pipeline_runs/jinan_direct_urgency_five_tracks_20260822_recovered_cohort_r3/baseline_threeflow/stage1/best.pt` |

Important boundary: the short selector and the primary final online test both
use online seed `2024` on the same flow. Therefore the primary result is an
online-validation-tuned result, not a fully held-out online-test result. The
result is still useful for testing checkpoint-selection behavior, but a
stronger promotion claim needs a held-out final online seed or a short-selector
seed that is distinct from the primary test seed.

## Raw Primary Online Summary

Primary online seed: `2024`. Values are mean +/- sample standard deviation over
three training seeds.

Reward is better when higher. AQL, AWT, ATT, P95 wait, and max wait are better
when lower.

| Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wait225_mixed_rank` | 3 | **-624.280 +/- 1.363** | **175.228 +/- 2.782** | **43.447 +/- 0.781** | **280.462 +/- 1.806** | **180.667 +/- 3.512** | 583.333 +/- 42.724 |
| `wait2_mixed_rank` | 3 | -628.029 +/- 7.148 | 177.247 +/- 5.323 | 45.517 +/- 0.686 | 281.203 +/- 3.369 | 184.333 +/- 4.726 | **548.667 +/- 30.567** |

## Paired Comparison Against Wait2

The table reports paired benefits of `wait225_mixed_rank` relative to
`wait2_mixed_rank` over matched training seeds. Positive reward benefit means
higher reward. Positive benefits for AQL, AWT, ATT, P95, and max wait mean the
metric is lower for `wait225_mixed_rank`.

| Metric | Mean paired benefit | Std | Seeds favoring `wait225` |
|---|---:|---:|---:|
| Reward | +3.748 | 7.825 | 2 / 3 |
| AQL | +2.019 | 8.078 | 2 / 3 |
| AWT | +2.070 | 1.460 | 3 / 3 |
| ATT | +0.741 | 5.163 | 2 / 3 |
| P95 wait | +3.667 | 6.658 | 2 / 3 |
| Max wait | -34.667 | 70.216 | 1 / 3 |

## Per-Seed Primary Results

Each cell contains `Reward / AQL / AWT / ATT / P95 / Max` for primary online
seed `2024`.

| Arm | Training seed 2024 | Training seed 42 | Training seed 123 |
|---|---|---|---|
| `wait2_mixed_rank` | -635.324 / 182.108 / 45.929 / 284.112 / 188 / 517 | -627.725 / 178.075 / 45.896 / 281.984 / 186 / 551 | -621.037 / 171.558 / 44.725 / 277.512 / 179 / 578 |
| `wait225_mixed_rank` | -624.357 / 173.067 / 43.135 / 279.108 / 177 / 608 | -622.881 / 174.250 / 42.870 / 279.766 / 184 / 608 | -625.604 / 178.367 / 44.336 / 282.512 / 181 / 534 |

## Checkpoint Selection Diagnostic

The short-online selector chose a different checkpoint than offline `val_acc`
in all six arm/seed cells.

| Arm | Training seed | Short-online selected | Short reward | Offline best `val_acc` epoch | Offline best top-1 regret epoch | Offline best expected regret epoch |
|---|---:|---|---:|---:|---:|---:|
| `wait2_mixed_rank` | 2024 | `epoch_12` | -190.168 | 10 | 13 | 13 |
| `wait2_mixed_rank` | 42 | `epoch_11` | -189.670 | 14 | 13 | 13 |
| `wait2_mixed_rank` | 123 | `epoch_9` | -188.094 | 10 | 13 | 13 |
| `wait225_mixed_rank` | 2024 | `epoch_13` | -187.680 | 12 | 15 | 13 |
| `wait225_mixed_rank` | 42 | `epoch_15` | -186.813 | 13 | 13 | 10 |
| `wait225_mixed_rank` | 123 | `epoch_15` | -186.023 | 13 | 11 | 11 |

This confirms that checkpoint selection is an active experimental factor.
It does not yet prove that the short-online selector is superior to offline
`val_acc`, because this campaign did not run a matched final-online comparison
for the saved offline-`val_acc` checkpoints.

## Interpretation

1. `wait225_mixed_rank` remains the best average-objective trade-off in the
   tested queue/wait neighborhood when paired with online-aware checkpoint
   selection.
2. The improvement is not a tail-safety result. P95 improves on average, but
   maximum wait worsens and only one of three training seeds favors `wait225`
   on max wait.
3. The selected epochs show that `val_acc`, top-1 regret, expected regret, and
   short-online reward can point to different checkpoints. This is consistent
   with the earlier diagnosis that offline checkpoint metrics are imperfect
   proxies for closed-loop behavior.
4. The current result is online-validation-tuned on the same flow and online
   seed used for the primary final result. It should be reported as a
   checkpoint-selection diagnostic and main-line candidate, not as a fully
   held-out online generalization claim.

## Decision

1. Keep `wait225_mixed_rank` as the main average-objective candidate under the
   online-checkpoint-selection protocol.
2. Do not claim a max-wait or hard tail-wait improvement from this campaign.
3. Treat the short-online selector as promising but unconfirmed relative to
   offline `val_acc` until a matched offline-selector final-online comparator
   is evaluated.

## Offline-ValAcc Comparator Follow-Up

Run root:
`records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823`

This follow-up reused the saved `best.offline_val_acc.pt` backups from the
short-online campaign and evaluated them with the same 3,600-second same-flow
online protocol. It completed 18/18 offline-`val_acc` rows with 0 incomplete
runs.

Primary online seed `2024`; values are mean +/- sample standard deviation over
three training seeds.

| Selector | Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---|---:|---:|---:|---:|---:|---:|
| `offline_val_acc` | `wait225_mixed_rank` | -626.249 +/- 4.190 | 175.372 +/- 1.765 | 44.852 +/- 1.637 | 280.295 +/- 1.166 | 181.667 +/- 6.351 | 554.667 +/- 60.044 |
| `offline_val_acc` | `wait2_mixed_rank` | -629.080 +/- 3.870 | 177.958 +/- 2.103 | 45.972 +/- 3.195 | 282.188 +/- 1.531 | 186.000 +/- 11.533 | 748.667 +/- 93.885 |
| `short_online` | `wait225_mixed_rank` | **-624.280 +/- 1.363** | **175.228 +/- 2.782** | **43.447 +/- 0.781** | 280.462 +/- 1.806 | **180.667 +/- 3.512** | 583.333 +/- 42.724 |
| `short_online` | `wait2_mixed_rank` | -628.029 +/- 7.148 | 177.247 +/- 5.323 | 45.517 +/- 0.686 | **281.203 +/- 3.369** | 184.333 +/- 4.726 | **548.667 +/- 30.567** |

The strict comparator resolves the earlier open question: short-online
selection gives small, noisy average gains over offline `val_acc`, but the
effect is not uniformly seed-stable. The strongest reliable signal is for
`wait2_mixed_rank` maximum wait, where short-online reduces mean max wait from
748.667 to 548.667 and wins 3/3 training seeds. The main-line conclusion still
favors `short_online + wait225_mixed_rank` for average objectives.

## Tail-Risk Selector Follow-Up

Run root:
`records/pipeline_runs/jinan_online_checkpoint_selection_tailrisk_20260823_r3`

The tail-risk selector reused the same 1,200-second short-online candidate
pool, but ranked checkpoints by
`short_reward - 0.10 * P95_wait - 0.02 * Max_wait`. It completed 18/18
tail-risk rows with 0 incomplete runs.

Primary online seed `2024`; values are mean +/- sample standard deviation over
three training seeds.

| Selector | Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---|---:|---:|---:|---:|---:|---:|
| `short_online` | `wait225_mixed_rank` | **-624.280 +/- 1.363** | **175.228 +/- 2.782** | **43.447 +/- 0.781** | **280.462 +/- 1.806** | 180.667 +/- 3.512 | **583.333 +/- 42.724** |
| `tail_risk` | `wait225_mixed_rank` | -627.212 +/- 3.915 | 177.692 +/- 4.327 | 43.707 +/- 0.602 | 281.756 +/- 2.363 | **177.000 +/- 4.000** | 593.333 +/- 53.529 |
| `short_online` | `wait2_mixed_rank` | **-628.029 +/- 7.148** | **177.247 +/- 5.323** | 45.517 +/- 0.686 | 281.203 +/- 3.369 | 184.333 +/- 4.726 | **548.667 +/- 30.567** |
| `tail_risk` | `wait2_mixed_rank` | -630.149 +/- 4.579 | 177.519 +/- 2.735 | **43.333 +/- 1.045** | **281.049 +/- 1.772** | **173.667 +/- 2.517** | 607.667 +/- 50.292 |

Tail-risk ranking should not replace short-online reward ranking as the default
selector. It reduces P95 wait, especially for `wait2_mixed_rank`, but it hurts
reward and does not reliably reduce final 3,600-second maximum wait. For
`wait225_mixed_rank`, tail-risk improves P95 but worsens reward, AQL, ATT, and
max wait. For `wait2_mixed_rank`, it improves AWT and P95 but makes max wait
worse than short-online. The result suggests that short-eval max-wait penalties
are too noisy for direct linear scalarization.

The next selector should therefore be constrained rather than fully
scalarized: first take the short-online Reward top-k candidates, then choose
among that restricted set using a tail-wait criterion such as
`P95_wait + 0.05 * Max_wait`.

## Constrained Selector Follow-Up

Run root:
`records/pipeline_runs/jinan_online_checkpoint_selection_constrained_20260823`

The constrained selector excludes mutable `best`/`last` aliases and uses only
stable `epoch_*.pt` checkpoints from the same 1,200-second short-online
candidate pool. It first keeps the short-online Reward top-3 candidates, then
selects the checkpoint with the lowest `P95_wait + 0.05 * Max_wait`. It
completed 18/18 constrained rows with 0 incomplete runs.

Primary online seed `2024`; values are mean +/- sample standard deviation over
three training seeds.

| Selector | Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---|---:|---:|---:|---:|---:|---:|
| `short_online` | `wait225_mixed_rank` | **-624.280 +/- 1.363** | **175.228 +/- 2.782** | 43.447 +/- 0.781 | **280.462 +/- 1.806** | 180.667 +/- 3.512 | **583.333 +/- 42.724** |
| `constrained` | `wait225_mixed_rank` | -626.522 +/- 4.438 | 175.694 +/- 5.140 | **42.970 +/- 0.779** | 280.638 +/- 2.596 | **173.667 +/- 3.055** | 607.667 +/- 30.501 |
| `short_online` | `wait2_mixed_rank` | -628.029 +/- 7.148 | 177.247 +/- 5.323 | 45.517 +/- 0.686 | 281.203 +/- 3.369 | 184.333 +/- 4.726 | **548.667 +/- 30.567** |
| `constrained` | `wait2_mixed_rank` | **-627.899 +/- 6.517** | **176.997 +/- 2.876** | **42.985 +/- 0.507** | **281.062 +/- 1.599** | **174.667 +/- 1.155** | 633.333 +/- 53.985 |

The constrained selector is a useful P95/AWT branch, but not a new default. It
lowers P95 and AWT for both arms relative to short-online reward ranking:
`wait225_mixed_rank` P95 drops from 180.667 to 173.667 and AWT from 43.447 to
42.970; `wait2_mixed_rank` P95 drops from 184.333 to 174.667 and AWT from
45.517 to 42.985. The cost is maximum wait: both arms have worse final max
wait than short-online, and constrained wins 0/3 training seeds on max wait for
both arms.

Compared with the linear tail-risk selector, constrained selection is more
balanced. It recovers reward and average metrics, especially for
`wait225_mixed_rank`, while retaining most of the P95/AWT benefit. The best
interpretation is: `short_online + wait225_mixed_rank` remains the main
average-objective line; `constrained + wait225_mixed_rank` is the cleaner
tail-latency branch if P95/AWT is the reporting target. Neither selector should
be claimed as a maximum-wait solution.

## Held-Out Selector-Seed Follow-Up

Run root:
`records/pipeline_runs/jinan_online_checkpoint_selection_selector_seed42_20260823`

This follow-up reused the trained `epoch_*.pt` candidates from the original
short-online selector campaign, but re-ran the 1,200-second short-online
candidate evaluations with selector seed `42` instead of selector seed `2024`.
It completed 18/18 final online rows and 90/90 short-eval rows with 0
incomplete runs.

Primary online seed `2024`; values are mean +/- sample standard deviation over
three training seeds.

| Selector | Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---|---:|---:|---:|---:|---:|---:|
| `short_online_seed2024` | `wait225_mixed_rank` | **-624.280 +/- 1.363** | **175.228 +/- 2.782** | **43.447 +/- 0.781** | 280.462 +/- 1.806 | **180.667 +/- 3.512** | 583.333 +/- 42.724 |
| `short_online_seed42` | `wait225_mixed_rank` | -624.292 +/- 1.345 | **175.228 +/- 2.782** | **43.447 +/- 0.781** | 280.462 +/- 1.806 | **180.667 +/- 3.512** | 583.333 +/- 42.724 |
| `short_online_seed2024` | `wait2_mixed_rank` | -628.029 +/- 7.148 | 177.247 +/- 5.323 | **45.517 +/- 0.686** | 281.203 +/- 3.369 | **184.333 +/- 4.726** | **548.667 +/- 30.567** |
| `short_online_seed42` | `wait2_mixed_rank` | **-626.338 +/- 4.761** | **175.264 +/- 3.349** | 45.637 +/- 0.813 | **279.858 +/- 2.244** | 185.467 +/- 6.217 | 589.000 +/- 44.531 |

The important result is not the small metric movement in the table, but the
selection stability. Selector seed `42` chose the same epoch as selector seed
`2024` in all 6 arm/seed cells: `wait225_mixed_rank` selected epochs 13, 15,
and 15 for training seeds 2024, 42, and 123; `wait2_mixed_rank` selected
epochs 12, 11, and 9. Candidate rankings under the two selector seeds had
Spearman correlation around 0.90-0.93 across cells.

This closes the selector-seed concern for the current evidence package:
short-online reward ranking is stable to this selector-seed change, but the
validation does not create a new better checkpoint. The main line remains
`short_online + wait225_mixed_rank`. Further selector scalarization is unlikely
to solve maximum wait; that needs an explicit vehicle-level max-wait protection
or an action-specific tail-wait teacher.

## Suggested Next Experiments

1. If the goal shifts back to maximum waiting time, add an explicit vehicle-level
   max-wait or protection mechanism instead of further selector scalarization.
2. If the goal shifts back to tail waiting, add an explicit action-specific
   tail-wait teacher instead of relying on mean AWT weight or short-online
   reward alone.
3. For the main paper line, freeze `short_online + wait225_mixed_rank` as the
   average-objective candidate and report the selector-seed42 validation as a
   robustness check, not as a new method variant.

## Source Artifacts

- Aggregate: `records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2/AGGREGATE.md`
- Raw online rows: `records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2/online_results.csv`
- Offline-`val_acc` comparator aggregate:
  `records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823/AGGREGATE.md`
- Tail-risk selector aggregate:
  `records/pipeline_runs/jinan_online_checkpoint_selection_tailrisk_20260823_r3/AGGREGATE.md`
- Constrained selector aggregate:
  `records/pipeline_runs/jinan_online_checkpoint_selection_constrained_20260823/AGGREGATE.md`
- Held-out selector-seed aggregate:
  `records/pipeline_runs/jinan_online_checkpoint_selection_selector_seed42_20260823/AGGREGATE.md`
- Incomplete audit: `records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2/incomplete_runs.csv`
- Provenance: `records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2/provenance.env`
- Selection reports: `records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2/arms/*/seed_*/artifacts/online_checkpoint_selection.json`
- Campaign script: `scripts/run_jinan_online_checkpoint_selection_campaign.sh`
