# Future-Robust R001 Result Analysis

**Date:** 2026-08-20  
**Run:** `records/pipeline_runs/future_robust_hangzhou_audit_20260820_143959`  
**Verdict:** **NO-GO for student training**

## What Ran

- Teacher-only policy-responsive SimExpert audit; the deployed/student policy was not trained or changed.
- Four time-stratified Hangzhou-real snapshots (`t=1200, 1770, 2370, 3000`), one highest phase-queue target per snapshot.
- One exact nominal future plus five pre-registered future-only perturbations per snapshot: demand scale cycling through `0.85, 1.00, 1.15` and arrival jitter of +/-10 seconds.
- Four action rollouts per future, for `4 x 6 x 4 = 96` counterfactual rollouts.
- Acceptance required all-future SimExpert feasibility and a source-relative worst-tail (25%) composite-score advantage of at least `-0.10`.

## Integrity Checks

- `8/8` focused unit tests passed.
- `24/24` generated future-traffic/snapshot CityFlow restore checks passed before the valid GPU run.
- The nominal future is a byte-identical copy of the source traffic file.
- Every perturbed traffic file preserves the original list length and CityFlow `flow_i` identifiers. Under-scale samples are deferred outside the 120-second audit window; above-scale demand extends existing future-only flows rather than adding flow objects.
- The valid run completed normally, wrote four decisions and `summary.json`, and released GPU 0. Earlier startup directories are diagnostic failures only and are excluded from these results.

## Raw Decision Table

| Snapshot time | Target | Source action | Nominal winner | All-future winner | Robust action | Accepted | Best non-source CVaR delta | Nominal global AWT delta of that candidate |
|---:|---|---|---|---|---|---:|---:|---:|
| 1200 | `intersection_1_4` | ETWT | ETWT | ETWT in 6/6 | ETWT | No | NLSL: +1.764 | +502 s |
| 1770 | `intersection_2_3` | NTST | NTST | NTST in 6/6 | NTST | No | ETWT: +2.158 | +226 s |
| 2370 | `intersection_2_2` | ELWL | ELWL | ELWL in 6/6 | ELWL | No | NLSL: +2.171 | +207 s |
| 3000 | `intersection_4_4` | ETWT | ETWT | ETWT in 6/6 | ETWT | No | ELWL: +2.418 | +237 s |

Positive deltas are worse than the source action. CVaR is a source-relative teacher-ranking score, not a physically calibrated safety quantity.

## Aggregate Results

| Metric | Result |
|---|---:|
| Selected snapshots | 4 |
| Future realizations | 24 (4 snapshots x 6 futures) |
| Candidate rollouts | 96 |
| Robust non-source labels accepted | 0 / 4 |
| Nominal-to-perturbed top-1 flips | 0 / 20 perturbed comparisons |
| Mean per-snapshot top-1 flip rate | 0.0 |
| Non-source candidate/future pairs | 72 |
| Non-source pairs feasible in all futures | 0 / 72 |
| Constraint failure reason | `awt_exceeded` for all 72 / 72 |
| Non-source CVaR-delta range | +1.764 to +3.200 |
| Raw global-AWT-delta range across non-source future candidates | +183 s to +1237 s |

## Interpretation

1. **No deterministic-label instability was observed.** The nominal winner equals the source action at all four targets and remains the winner under every sampled perturbation.
2. **The robust gate is not merely rejecting a useful label.** All alternative first actions are already worse than the source action in the teacher's paired ranking and violate the source-relative AWT constraint in every future realization.
3. **R001 does not support the proposed mechanism on this pool.** There are no label changes to certify, so a robust-label student arm would reduce to the source policy. Training it cannot test the claimed benefit.

## Decision and Next Step

Do not launch R002/R003 or any student training from this run. The correct next research action, if this direction remains of interest, is a new teacher-only discovery audit over a deliberately disagreement-rich snapshot pool or a stronger/longer perturbation family, with both choices pre-registered before inspecting labels. Such a study must retain the same source-action baseline and matched random-rejection control before any distillation claim.

## Provenance

- Summary: `records/pipeline_runs/future_robust_hangzhou_audit_20260820_143959/summary.json`
- Decisions: `records/pipeline_runs/future_robust_hangzhou_audit_20260820_143959/future_robust_decisions.jsonl`
- Summary SHA-256: `219e5938010751900a00335a596de5b38bc8de8304f6f276c7778bc759f4985f`
- Decision SHA-256: `49359d86a216e7e513c5f302f0ac8db8ef8ea413cb4566f767f11c4604f7c277`
