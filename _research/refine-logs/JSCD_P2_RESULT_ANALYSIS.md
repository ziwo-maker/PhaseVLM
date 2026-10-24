# JSCD P2 Pilot Result Analysis

**Date:** 2026-08-20  
**Run:** `records/pipeline_runs/jscd_p2_pilot_20260819_141017/`  
**Verdict:** `P1 PASS; P2 NO-GO; P3 MUST NOT START.`

## Executive Conclusion

JSCD has a real teacher-level mechanism: replacing the old-policy continuation with the first-pass student changes the counterfactual decision substantially. On the primary `Hangzhou_real` flow, JSCD also has a favorable but unstable three-seed AWT signal: it improves over Static-PRCT by `2.80` AWT (`5.51%`) on average and over Frozen by `2.25` AWT (`4.47%`) on average.

This is **not** a validated performance result. JSCD improves over Frozen in only `2/3` paired training seeds, so it fails the pre-registered P2 `3/3` AWT direction gate. The exact two-sided Wilcoxon tests are non-significant (`p=0.5` for both JSCD-vs-Static-PRCT and JSCD-vs-Frozen AWT), and the small-sample 95% t intervals cross zero. Required `5816` evaluation and safety metrics were not produced. P3 must therefore not be launched.

## Scope and Integrity

- The run completed normally at `2026-08-20 03:41:31 CST`; all nine planned second-pass arm/seed runs have a complete `best.pt` and one completed online result.
- P1 used 32 fixed snapshots / 512 decision states. P2 used the disjoint remaining 86 snapshots / 1,376 decision states.
- Each P2 training job used `torchrun` over GPUs `0,1,2`, five epochs, identical mixed loss, `K=86`, and one deterministic `Hangzhou_real` online evaluation (`eval_seed=2024`). The independent statistical unit is the training seed, not the online seed.
- Per training seed, Static-PRCT (P) and JSCD (J) start from the same first-pass PRCT checkpoint. Their only second-pass teacher difference is the continuation controller: P uses current-code `pi_0`; J uses the matching first-pass `pi_1`. Frozen (F) is the matched two-pass frozen-teacher control initialized from its matching first-pass Frozen checkpoint.
- The primary runner was invoked once per seed. Consequently its top-level `config.json`, `status.json`, and `aggregate.json` were overwritten by the last (`tr123`) invocation. This analysis reads the per-seed result JSONs directly; the top-level aggregate files must not be cited as three-seed aggregates.

## P1: Self-Consistency Mechanism Gate

| Quantity | Result | Pre-registered threshold | Result |
|---|---:|---:|---|
| Top-1 action changes | `135 / 512 = 26.37%` decision states | `>= 20%` | Pass |
| Snapshots with any top-1 change | `31 / 32 = 96.88%` | Diagnostic | Material |
| Informative rankings | `493 / 512 = 96.29%` | `>= 50%` | Pass |
| Mean Kendall tau | `0.6789` | `<= 0.80` | Pass |

P1 shows that `pi_1` changes the counterfactual teacher enough to motivate JSCD. The P2 second-pass labels are also materially different: versus Static-PRCT, JSCD changes expert actions by `343/1376 (24.93%)`, `364/1376 (26.45%)`, and `337/1376 (24.49%)` for training seeds `2024`, `42`, and `123`, respectively. Thus the treatment is not a no-op.

### P1 Reproduction Defect

The planned `pi_0` replay equality check was not enforced before P2. Restricting the historical PRCT label file to the 512 P1 states gives:

- `12/512 = 2.34%` expert actions differ between historical PRCT and the current-code `pi_0` replay.
- `512/2048 = 25.00%` candidate score cells differ; maximum absolute score change is `0.20`.
- The discrepancies concentrate in spillback (`613` candidate-action metric cells) and, less often, AQL/AWT (`28` cells each), not in checkpoint identity or basic teacher provenance.

This is evidence of code/metric-version drift, not a random online seed effect. It is much smaller than the P1 `pi_0 -> pi_1` action change rate, so it does not erase the observed mechanism. It does, however, lower reproducibility confidence and means the strict P1 preflight condition was not fully satisfied.

## P2 Primary-Flow Results

All metrics are lower-is-better. Each row is one independent training seed, evaluated on `anon_4_4_hangzhou_real.json`.

| Training seed | Frozen ATT / AQL / AWT | Static-PRCT ATT / AQL / AWT | JSCD ATT / AQL / AWT |
|---:|---|---|---|
| 2024 | 316.11 / 64.03 / 55.62 | 315.19 / 63.42 / 50.66 | 315.54 / 64.28 / 50.98 |
| 42 | 311.39 / 60.33 / 44.24 | 314.39 / 63.23 / 51.02 | 311.65 / 60.48 / 45.82 |
| 123 | 315.58 / 63.73 / 50.96 | 312.12 / 61.54 / 50.80 | 310.97 / 60.56 / 47.29 |
| Mean +/- sample std | 314.36 +/- 2.59 / 62.69 +/- 2.06 / 50.27 +/- 5.72 | 313.90 +/- 1.60 / 62.73 +/- 1.03 / 50.83 +/- 0.18 | 312.72 +/- 2.47 / 61.77 +/- 2.17 / 48.03 +/- 2.66 |

### Paired Effects

`Delta = JSCD - reference`; negative is favorable.

| Comparison | ATT Delta | AQL Delta | AWT Delta | AWT direction |
|---|---:|---:|---:|---|
| JSCD - Static-PRCT | `-1.18` (`-0.38%`), 2/3 favorable | `-0.96` (`-1.53%`), 2/3 | `-2.80` (`-5.51%`), 2/3 | seed 2024 worsens by `+0.32` |
| JSCD - Frozen | `-1.64` (`-0.52%`), 2/3 favorable | `-0.92` (`-1.47%`), 1/3 | `-2.25` (`-4.47%`), 2/3 | seed 42 worsens by `+1.57` |
| Static-PRCT - Frozen | `-0.46` (`-0.15%`) | `+0.04` (`+0.06%`) | `+0.55` (`+1.10%`) | Static PRCT is not reliably better |

For AWT, the exact two-sided Wilcoxon `p` value is `0.5` for JSCD-vs-Static-PRCT and JSCD-vs-Frozen. The corresponding three-seed t intervals are `[-9.83, +4.23]` and `[-10.55, +6.06]`. These intervals are included only to show uncertainty; `n=3` is insufficient for a performance claim.

## Gate Decision

| P2 requirement | Observed result | Decision |
|---|---|---|
| JSCD beats Frozen in AWT for all 3 pairs | 2/3; seed 42 is worse | Fail |
| Mean JSCD-vs-Frozen AWT gain >= 3% | `4.47%` favorable | Pass alone, insufficient |
| JSCD beats Static-PRCT in at least 2/3 pairs | 2/3 | Pass |
| No >5% safety harm on both flows | Safety metrics absent; only one flow evaluated | Not evaluable / fail closed |
| Both `Hangzhou_real` and `5816` evaluated | `Hangzhou_real` only; zero `5816` artifacts | Not evaluable / fail closed |

**P2 is a NO-GO.** The positive mean is hypothesis-generating, but the direction and safety gates are deliberately stronger than a mean-only result. No P3 nine-seed extension is justified.

## What the Result Supports

1. **Mechanism claim, qualified:** continuation policy changes teacher rankings and labels at a material rate. This is the strongest valid JSCD finding.
2. **Performance hypothesis, not claim:** JSCD may improve AWT relative to Static-PRCT, but only two of three pairs do so and no inferential conclusion is possible.
3. **Boundary of the old PRCT story:** Static PRCT remains unstable in this two-pass setting, with mean AWT `+1.10%` worse than Frozen.

## What the Result Does Not Support

- JSCD improves deployment performance reliably.
- JSCD is safer, since p95/max movement wait, spillback, action entropy, and phase frequency were not measured.
- JSCD generalizes across flows, since `5816` was not evaluated.
- A P3 confirmation run, hyperparameter sweep, or paper-level performance claim.

## Required Next Action

1. **Do not start P3 or any loss/horizon/temperature sweep.** The pre-registered efficacy gate failed.
2. **Resolve the `pi_0` replay discrepancy first.** Version-lock the relabeler and score implementation; identify the spillback/AQL/AWT drift; then require exact action and candidate-score agreement on the P1 subset. If the current semantics are intentional, regenerate the first-pass PRCT teacher and retrain its `pi_1` checkpoints under the locked code before repeating a JSCD pilot.
3. **Keep any post-hoc work diagnostic, not confirmatory.** P0's nine existing `Hangzhou_real` re-evaluations remain incomplete. Evaluating the already trained P2 checkpoints on `5816` and adding safety extraction may characterize failure, but cannot rescue this failed P2 gate or justify P3.
4. **Before any future pilot, persist one immutable manifest per arm/seed** with checkpoint hash, label hash, source code revision, test flow, and metrics. Avoid relying on runner-level aggregate files that are overwritten in a per-seed launch loop.

## Citation-Safe Finding Statement

> In a three-training-seed, compute-matched pilot on `Hangzhou_real`, joint-policy continuation changed approximately one quarter of second-pass counterfactual labels and reduced mean AWT relative to static PRCT, but the effect was not consistent across all pairs, was not statistically decisive, and lacked required safety and second-flow validation. We therefore do not claim a deployment-performance improvement.
