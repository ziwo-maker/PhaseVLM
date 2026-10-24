# Future-Robust Counterfactual Teacher Audit Plan

**Date:** 2026-08-20  
**Status:** R001 implementation and teacher-only pilot approved by the user. No student training is authorized by this plan.

## Problem and Thesis

Existing policy-responsive H120 counterfactual distillation has a stable in-network AWT signal but no held-out improvement. The primary hypothesis is not that a new loss repairs this failure: it is that a deterministic future may assign an action label that is not robust to plausible future arrivals.

**Method thesis:** construct supervision only from candidate actions that retain a source-relative cost advantage and SimExpert feasibility over a pre-registered distribution of future traffic realizations; otherwise preserve the source action.

## Claim Map

| Claim | Minimum convincing evidence | R001 role |
|---|---|---|
| C1: deterministic H120 labels are materially unstable under future arrival perturbations | nontrivial winner flips/rank changes or a material deterministic-vs-CVaR disagreement | primary mechanism audit |
| C2: future-robust acceptance is more than reducing update count | later matched student study must beat random rejection at identical acceptance rate | explicitly deferred |
| Anti-claim: this is CRCD or generic robust RL renamed | source-relative paired candidate evaluation, future distribution fixed before results, deployment remains one student forward pass | provenance and controls |

## R001: Teacher-Only Future Audit

- **Setting:** Jinan-source pt32/hd256 checkpoint, Hangzhou `anon_4_4_hangzhou_real`, existing clean Stage-3 OOD state pool, policy-responsive continuation, H=120, global SimExpert score.
- **Sampling:** four time-stratified snapshots in `[1200, 3000]` seconds; one highest phase-queue target intersection per snapshot. This is a pilot sample, not independent online evaluation data.
- **Futures:** one nominal future plus five pre-registered perturbations. Perturbations use future-only demand scaling cycling through `0.85, 1.00, 1.15` and `+/-10s` arrival jitter. To preserve CityFlow snapshot identifiers, every original flow keeps its list index and the file length stays fixed: under-scale samples are deferred beyond the 120-second audit window, while above-scale samples extend future-only flow emission within the existing `flow_i` identifier. Vehicles starting at or before the snapshot remain byte-for-byte unchanged.
- **Decision:** each future evaluates all four candidate phases under the same future traffic file. Candidate cost is compared with the source-policy action within that future. The aggregate robust action minimizes the upper tail (worst 25%) of the relative cost; it is accepted only if feasible in all sampled futures and its CVaR advantage is at most `-0.10`.
- **Metrics:** nominal-vs-perturbed top-1 flip rate, per-action mean/CVaR relative cost, future-wise Pareto/constraint failures, accepted-label rate, and source-action fallback rate.
- **Safety scope:** R001 uses existing SimExpert AWT/spillback feasibility. It does not claim per-movement p95/max wait certification; that metric extractor is a mandatory prerequisite for student training.

## Decision Gate

- **GO to label-generation pilot:** winner/rank instability is material and at least one non-source label satisfies the pre-registered robust acceptance rule.
- **NO-GO:** winners are stable, source action dominates under perturbations, or feasibility rejects nearly all non-source actions. Report the deterministic teacher as adequate for this perturbation family and do not train a student.
- **No inference claim:** four snapshots and shared traffic realizations do not establish cross-flow generalization or a safe-policy theorem.

## Deferred Student Experiment

Only after R001 GO and per-movement tail metrics are available, compare `source/no-update`, deterministic H120 teacher, future-robust teacher, and random same-acceptance-rate rejection. Keep snapshot pool, source checkpoint, loss, training seeds, checkpoint selection, and both test flows matched. The random-rejection arm is required.
