# Jinan Stage-1 × Stage-2 Capacity Factorial: Conclusion

Run root: `/home/code/LLMSTSC/records/pipeline_runs/jinan_capacity_stage1_factorial_20260822_r7`

Latest follow-ups: online checkpoint selection completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2`;
offline-`val_acc` comparator completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823`;
tail-risk selector completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_tailrisk_20260823_r3`;
constrained selector completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_constrained_20260823`;
held-out selector-seed validation completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_selector_seed42_20260823`.
See
`stage2_direct_urgency_innovation/anon_3_4_jinan_real/jinan_online_checkpoint_selection_20260822/RESULT_ANALYSIS.md`.

Status: **success**
Completion audit: **36/36 online rows completed, 0 incomplete**
Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, primary online seed `2024`
Training seeds: `314, 9001, 2718`
Arm: `wait225_mixed_rank`

> Important interpretation limit: the cross-architecture cells use
> `--allow_mismatched_align_projectors`, which copies overlapping projector
> weights when Stage 1 and Stage 2 tensor shapes differ. Therefore these cells
> test cross-architecture transfer with overlap-copy initialization, not a pure
> same-shape initializer replacement.

## Raw primary online summary

Primary online seed: `2024`; values are mean ± std over 3 training seeds.

Reward is better when higher. AQL, AWT, ATT, and P95 wait are better when lower.

| Cell | Stage 1 | Stage 2 | Reward | AQL | AWT | ATT | P95 wait |
|---|---|---|---:|---:|---:|---:|---:|
| `stage1_pt8_stage2_pt8` | `pt8_hd64` | `pt8_hd64` | **-629.235 ± 5.047** | **178.075 ± 3.733** | 45.254 ± 0.664 | **281.963 ± 2.376** | 184.483 ± 3.569 |
| `stage1_pt8_stage2_pt16` | `pt8_hd64` | `pt16_hd128` | -631.184 ± 7.471 | 178.467 ± 6.188 | 45.898 ± 4.048 | 282.629 ± 3.162 | 184.450 ± 17.249 |
| `stage1_pt16_stage2_pt8` | `pt16_hd128` | `pt8_hd64` | -634.480 ± 2.237 | 181.825 ± 2.949 | **44.720 ± 1.275** | 284.512 ± 1.552 | 178.400 ± 3.934 |
| `stage1_pt16_stage2_pt16` | `pt16_hd128` | `pt16_hd128` | -630.827 ± 1.660 | 179.650 ± 1.594 | 45.594 ± 0.586 | 283.025 ± 1.358 | **176.550 ± 3.493** |

## Paired causal comparisons

Metric deltas are candidate minus baseline over matched training seeds.
Positive reward is favorable; negative delay/queue/tail deltas are favorable.

| Comparison | Reward Δ | AQL Δ | AWT Δ | ATT Δ | P95 wait Δ | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Stage 2 capacity, Stage 1 fixed at `pt8_hd64`: `pt8/pt16 - pt8/pt8` | -1.949 ± 3.933 | +0.392 ± 2.455 | +0.644 ± 3.427 | +0.666 ± 1.066 | -0.033 ± 17.706 | Larger Stage 2 alone does not improve the main line. |
| Stage 1 capacity, Stage 2 fixed at `pt8_hd64`: `pt16/pt8 - pt8/pt8` | -5.245 ± 7.166 | +3.750 ± 6.395 | -0.534 ± 1.067 | +2.549 ± 3.872 | -6.083 ± 2.270 | Larger Stage 1 improves tail wait but hurts reward/AQL/ATT. |
| Stage 2 capacity, Stage 1 fixed at `pt16_hd128`: `pt16/pt16 - pt16/pt8` | +3.653 ± 3.891 | -2.175 ± 4.396 | +0.875 ± 1.387 | -1.487 ± 2.891 | -1.850 ± 7.364 | Larger Stage 2 partly repairs the high-Stage1/small-Stage2 mismatch. |
| Stage 1 capacity, Stage 2 fixed at `pt16_hd128`: `pt16/pt16 - pt8/pt16` | +0.357 ± 6.745 | +1.183 ± 4.598 | -0.304 ± 4.614 | +0.397 ± 1.969 | -7.900 ± 19.992 | High Stage 1 mainly contributes tail-wait reduction. |
| Both capacities vs baseline: `pt16/pt16 - pt8/pt8` | -1.592 ± 3.683 | +1.575 ± 2.144 | +0.341 ± 1.193 | +1.062 ± 1.041 | -7.933 ± 6.407 | Both-large is not the main-line winner, but improves P95. |

## Key findings

1. **Main-line winner remains `pt8_hd64` Stage 1 + `pt8_hd64` Stage 2.**
   It has the best primary reward, AQL, and ATT among the four cells.

2. **Increasing Stage 2 capacity alone does not improve the main result.**
   With Stage 1 fixed at `pt8_hd64`, moving Stage 2 from `pt8_hd64` to
   `pt16_hd128` makes reward, AQL, AWT, and ATT slightly worse on average.

3. **Increasing Stage 1 capacity does not improve reward/AQL/ATT, but it does
   improve tail wait.**
   With Stage 2 fixed at `pt8_hd64`, higher-capacity Stage 1 reduces P95 wait by
   about 6.08 while worsening reward by about 5.25 and AQL by about 3.75.

4. **The `pt16_hd128` Stage 1 + `pt16_hd128` Stage 2 cell is a tail-latency
   variant, not a main-line replacement.**
   Compared with `pt8/pt8`, it improves P95 wait by about 7.93, but reward,
   AQL, AWT, and ATT are all worse on average.

5. **There is a Stage 1 × Stage 2 interaction, but it does not overturn the
   baseline.**
   Larger Stage 2 helps when Stage 1 is already large, but this mostly recovers
   from the poor `pt16/pt8` setting. It still does not surpass `pt8/pt8` on
   main-line online metrics.

6. **Offline validation accuracy is not aligned with online reward.**
   `pt16/pt16` has the best offline validation accuracy mean, but it does not
   have the best online reward/AQL/ATT. This suggests that checkpoint selection
   by `val_acc` is not sufficient for the traffic-control objective.

## Answer to the research question

Current evidence **does not support improving the Stage 2 main-line result by
simply increasing Stage 1 or Stage 2 capacity**.

- For reward/AQL/ATT, keep the current `pt8_hd64` Stage 1 + `pt8_hd64` Stage 2
  as the main-line configuration.
- Treat higher-capacity Stage 1/Stage 2 as a possible **tail-wait optimization
  branch**, not as a main-line replacement.

## Suggested next steps

1. **Do not continue blind capacity scaling for the main line.**
   The next main-line experiments should target loss design and checkpoint
   selection rather than more memory or larger projectors.

2. **Online-aware checkpoint selection has now been tested.**
   The completed follow-up selected saved epoch checkpoints with a short
   1,200-second same-flow online reward probe. It changed the selected epoch in
   all six tested arm/seed cells and made `wait225_mixed_rank` the best
   average-objective candidate in that protocol. A matched offline-`val_acc`
   comparator is now complete: short-online selection gives small, noisy
   average gains over offline `val_acc`, and strongly reduces `wait2` maximum
   wait, but it is not uniformly seed-stable.

3. **Use constrained selection only as a P95/AWT branch, not the default.**
   The tail-risk selector
   `short_reward - 0.10 * P95_wait - 0.02 * Max_wait` reduces P95, especially
   for `wait2_mixed_rank`, but worsens reward and does not beat short-online on
   final maximum wait. The constrained selector, which first keeps Reward top-3
   and then minimizes `P95_wait + 0.05 * Max_wait`, is more balanced and lowers
   P95/AWT, but still worsens final max wait relative to short-online.

4. **Confirm the tail-wait finding cheaply.**
   Reuse existing checkpoints and run more online evaluation seeds for
   `pt8/pt8` and `pt16/pt16` to test whether the P95 improvement is stable.

5. **If keeping a tail-latency branch, report it as a trade-off.**
   The defensible statement is: higher capacity improves P95 wait but currently
   trades off average efficiency and reward.

## Online Checkpoint Selection Follow-Up

Run root:
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2`

Status: **success**
Completion audit: **18/18 online rows completed, 0 incomplete**
Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, primary online
seed `2024`
Short selector: 1,200-second same-flow online reward, selector seed `2024`
Training seeds: `2024, 42, 123`
Arms: `wait2_mixed_rank`, `wait225_mixed_rank`

> Important interpretation limit: the short selector and the primary final
> online result both use online seed `2024` on the same flow. This is an
> online-validation-tuned selector diagnostic, not a fully held-out online-test
> claim. A strict selector claim needs a matched offline-`val_acc` comparator
> and/or a held-out final online seed.

### Raw primary online summary

Primary online seed: `2024`; values are mean ± std over 3 training seeds.

| Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---:|---:|---:|---:|---:|---:|
| `wait225_mixed_rank` | **-624.280 ± 1.363** | **175.228 ± 2.782** | **43.447 ± 0.781** | **280.462 ± 1.806** | **180.667 ± 3.512** | 583.333 ± 42.724 |
| `wait2_mixed_rank` | -628.029 ± 7.148 | 177.247 ± 5.323 | 45.517 ± 0.686 | 281.203 ± 3.369 | 184.333 ± 4.726 | **548.667 ± 30.567** |

### Key findings

1. **`wait225_mixed_rank` is the best average-objective candidate under the
   short-online selector.**
   It improves mean reward, AQL, AWT, ATT, and P95 wait relative to `wait2`.
   The AWT benefit is seed-stable, winning 3/3 paired training seeds.

2. **The result is not a max-wait improvement.**
   `wait225_mixed_rank` has worse mean maximum wait than `wait2` and wins only
   1/3 paired training seeds on max wait.

3. **Checkpoint selection is an active factor.**
   The short-online selector chose a different epoch than offline `val_acc` in
   all six arm/seed cells. This supports the earlier diagnosis that pure
   `val_acc` is not sufficiently aligned with closed-loop behavior.

4. **The minimal next experiment is a selector comparator, not a new model.**
   Reuse the saved `best.offline_val_acc.pt` backups from this campaign and run
   the same 3,600-second online suite. That directly tests whether the
   short-online selector improves over offline `val_acc` checkpoint selection.

## Offline-ValAcc And Tail-Risk Selector Follow-Ups

The strict offline-`val_acc` comparator completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823`
with 18/18 offline rows and 0 incomplete. It confirms that short-online
selection is mildly helpful but noisy relative to offline `val_acc`: for
`wait225_mixed_rank`, Reward improves from -626.249 to -624.280, AWT improves
from 44.852 to 43.447, and P95 improves from 181.667 to 180.667, while max wait
worsens from 554.667 to 583.333. For `wait2_mixed_rank`, short-online strongly
reduces max wait from 748.667 to 548.667.

The tail-risk selector completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_tailrisk_20260823_r3`
with 18/18 tail-risk rows and 0 incomplete. Its rule was
`short_reward - 0.10 * P95_wait - 0.02 * Max_wait`. It should not replace
short-online reward ranking as the default selector. On `wait2_mixed_rank`, it
improves AWT from 45.517 to 43.333 and P95 from 184.333 to 173.667 relative to
short-online, but worsens Reward from -628.029 to -630.149 and max wait from
548.667 to 607.667. On `wait225_mixed_rank`, it improves P95 but worsens the
main average metrics.

The constrained selector completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_constrained_20260823`
with 18/18 constrained rows and 0 incomplete. It uses only stable `epoch_*.pt`
checkpoints, keeps short-online Reward top-3, then minimizes
`P95_wait + 0.05 * Max_wait`. Relative to short-online reward ranking, it lowers
P95/AWT for both arms: `wait225_mixed_rank` P95 drops from 180.667 to 173.667
and AWT from 43.447 to 42.970; `wait2_mixed_rank` P95 drops from 184.333 to
174.667 and AWT from 45.517 to 42.985. It does not solve maximum wait:
`wait225_mixed_rank` max wait worsens from 583.333 to 607.667, and
`wait2_mixed_rank` worsens from 548.667 to 633.333.

The held-out selector-seed validation completed at
`/home/code/LLMSTSC/records/pipeline_runs/jinan_online_checkpoint_selection_selector_seed42_20260823`
with 18/18 final online rows, 90/90 short-eval rows, and 0 incomplete. Changing
the short-online selector seed from `2024` to `42` selected the same epoch in
all 6 arm/seed cells. Candidate ranking was also stable: Spearman rank
correlation between selector-seed `2024` and selector-seed `42` was about
0.90-0.93 across cells. Therefore the selector-seed follow-up supports
checkpoint-selection stability, but it does not introduce a new better
checkpoint choice.

The current decision is therefore: keep `short_online + wait225_mixed_rank` as
the main average-objective line, keep `constrained + wait225_mixed_rank` only
as a P95/AWT tail branch, and stop tuning selector scalarizations for maximum
wait. Maximum-wait improvement needs an explicit vehicle-level protection or
tail-wait teacher mechanism rather than another checkpoint selector.
