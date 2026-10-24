# Research Contract: Policy-Responsive Counterfactual Teacher Construction

## Selected Idea

- **Description:** PRCT forces one local phase while non-target intersections
  re-plan with the source policy every 30 seconds during an H120 CityFlow
  rollout. PRCAT advantage weighting and source-action regularization are a
  separately testable robustness module.
- **Source:** `idea-stage/IDEA_REPORT.md`, Idea 1.
- **Selection rationale:** It is the only candidate rooted in a reproducible
  mechanism: frozen-neighbor long-horizon labels caused coordination collapse,
  while responsive H120 preserved action diversity and produced positive
  results where target headroom existed. PRCAT is retained only to test the
  remaining held-out robustness failure.

## Core Claims

1. Closed-loop policy-responsive counterfactual evaluation is a more faithful
   TSC teacher than a frozen-neighbor rollout.
2. PRCAT may prevent low-value counterfactual labels from moving a strong
   source policy unnecessarily, subject to isolated validation.
3. The method's benefit is conditional on source-policy headroom and must be
   evaluated under held-out demand shifts.

## Experiment Design

- **Datasets:** Jinan in-network and Jinan-2500 held-out demand; then a second
  road network.
- **Baselines:** zero-shot, frozen c30, unweighted responsive H120, PRCAT
  weight-only, PRCAT KL-only.
- **Metrics:** ATT, AQL, AWT, p95/max per-movement wait, action entropy and
  action distribution.
- **Budget:** Pilot: up to 3 GPU-hours. Confirmation: up to 24 GPU-hours.

## Current Results

| Method | Dataset | Metric | Score | Notes |
|---|---|---|---|---|
| Responsive H120 | Jinan real | AWT | 40.97 +/- 0.84 | K=118, 9 training seeds, existing result |
| Responsive H120 | Jinan-2500 held-out | AWT delta | -0.24 +/- 0.82 | K=118, not significant |
| PRCAT full | Jinan real | AWT | 42.48 +/- 0.22 | 1 training seed / 3 eval seeds; worse than matched responsive-H120 seed (39.44) |

## Key Decisions

- Rejected frozen-neighbor reliability gating, OOD acquisition and pure
  coreset selection because closed-loop experiments were negative.
- Do not claim generic advantage weighting as novel. Position PRCT around the
  policy-responsive multi-intersection counterfactual evaluator and its
  coordination-bias evidence; PRCAT is conditional.
- Freeze the conservative thresholds before comparing with the held-out flow.
- The full PRCAT pilot is negative versus the matched unweighted teacher, so
  do not spend the remaining budget on its seed expansion without a new,
  preregistered hypothesis.

## Status

- [x] Idea selected
- [x] Baseline reproduced
- [x] Main method implemented
- [x] Representative PRCAT dataset pilot complete (negative module result)
- [ ] Held-out dataset results
- [ ] Nine-seed confirmation and ablations
- [ ] Paper draft
