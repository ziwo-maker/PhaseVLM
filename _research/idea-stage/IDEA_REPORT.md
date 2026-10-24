# Idea Discovery Report: PhaseVLM Stage 3 Replacement

**Direction:** Replace the invalidated OOD/selection-centered Stage-3 story with
a falsifiable control mechanism for policy-responsive counterfactual distillation.

**Date:** 2026-08-18  
**Pipeline:** research-lit -> idea-creator -> novelty-check -> research-review -> research-refine-pipeline  
**Assurance:** same-family critical review is complete but provisional; the
Semantic Scholar search became rate-limited (HTTP 429), so a final pre-
submission search remains required.

## Executive Summary

The previous innovation claim should be replaced. OOD acquisition, diversity
selection, frozen-neighbor long-horizon labels, and cross-horizon reliability
gating have all failed in closed-loop evaluation. The strongest reproducible
mechanism is that **a counterfactual teacher must let non-target intersections
respond with the deployed policy**, otherwise long-horizon labels artificially
favor through phases and cause coordination collapse.

The recommended replacement is **Policy-Responsive Counterfactual Teacher
Construction (PRCT)**: force a candidate phase at one intersection while all
other intersections re-plan with the deployed source policy every 30 seconds
over a 120-second CityFlow rollout. This replaces the invalid frozen-neighbor
teacher and directly explains the observed action-collapse failure. **PRCAT**
(advantage-weighted conservative distillation) is retained only as a
conditional robustness module; its online benefit has not yet been established.

## Literature Landscape

### Retrieved Sources

Local paper libraries `papers/` and `literature/` were absent. Searches used
arXiv, Semantic Scholar (until rate-limited), and OpenAlex. The latter two had
weak relevance ranking for this specialized query; results are therefore used
only for metadata and broad prior-art checks, not evidence of absence.

| Work | Relevance to PRCAT |
|---|---|
| Foerster et al., COMA, AAAI 2018 | Counterfactual baselines for multi-agent credit assignment; does not construct a simulator teacher by re-planning neighbors during a forced local action. |
| Balakrishna et al., *On-Policy Robot Imitation from a Converging Supervisor*, 2019 | Relevant to learning from a policy-dependent/model-based supervisor, but not traffic coordination or per-action counterfactual costs. |
| A2PR, ICML 2024 (arXiv:2405.19909) | Advantage-guided conservative offline policy regularization; prevents novelty claims about generic advantage weighting. |
| Behaviour Preference Regression, AAAI 2025 (arXiv:2503.00930) | Preference-style policy fitting with behavior consistency; closest generic objective family. |
| Braun, *A Graph-Based Control Interface for Traffic Signals on Heterogeneous Road Networks*, arXiv:2607.21831 | Topology-general TSC interface, not a counterfactual teacher/distillation method. |
| Li et al., *Large-scale Regional TSC Based on Single-Agent RL*, arXiv:2503.09252 | Simulator-based regional TSC baseline; no policy-responsive teacher. |

### Structural Gap

No retrieved work combined all of: (1) a frozen VLM traffic policy as the
multi-intersection rollout policy, (2) a forced one-intersection candidate
action, (3) neighbor re-planning during the counterfactual horizon, and (4) a
source-relative trust rule that prevents low-value counterfactual labels from
moving a strong deployed policy. The novelty is **not** advantage weighting;
it is the closed-loop counterfactual evaluator and the measured coordination
bias it corrects.

## Ranked Ideas

### Idea 1: Policy-Responsive Counterfactual Teacher Construction (PRCT) - RECOMMENDED

**Method:**

1. From each source-policy state, force one candidate phase at one intersection.
2. Every other intersection executes the same source policy again at each
   30-second decision point during a 120-second CityFlow rollout.
3. Use all four resulting global costs to produce the target distribution and
   compare frozen versus policy-responsive teachers across horizons and label
   budgets.

**Hypothesis:** frozen-neighbor labels encode a coordination-inconsistent
counterfactual. Re-planning non-target intersections with the source policy
removes that bias, preserves phase diversity, and improves closed-loop traffic
control when target-flow headroom exists.

**Existing evidence:** responsive H120 avoided the frozen long-horizon
through-action collapse; it improved older Jinan-to-Hangzhou results and
Jinan-real in-network AWT by 6.8% over nine seeds. Its held-out Jinan-2500
gain was indistinguishable from zero, motivating the conservative component.

**Novelty:** Medium-high (7/10, provisional). The forced-local-action,
policy-responsive multi-intersection CityFlow teacher and its measured
frozen-neighbor coordination bias are the defensible delta; generic
counterfactual credit assignment and policy fitting are prior art.

**Risk:** Medium. It is a simulator-specific empirical contribution and must
replicate across the original cross-network setting and held-out demand shift.

### Idea 2: PRCAT Conservative Distillation - CONDITIONAL MODULE

PRCAT weights the PRCT loss by source-relative counterfactual advantage and
adds a source-action KL prior on low-advantage rows. Its valid `K=118`,
training-seed-2024 pilot reached `AWT=42.48 +/- 0.22` across three CityFlow
evaluation seeds. This improves on zero-shot (`43.98`) but is `3.04s` worse
than the matched unweighted responsive-H120 training seed (`39.44`). Because
AWR/A2PR/preference-style conservatism are prior art and the isolated pilot is
negative against the teacher baseline, PRCAT is not a core contribution and is
not being expanded to more training seeds.

### Idea 3: Topology-Conditioned Prefix Hypernetwork - DEFERRED

Potentially high ceiling, but it needs six or more diverse training topologies
and leave-one-topology-out tests. The current two-network evidence cannot
support it; it is deferred rather than pursued with a weak evaluation.

## Eliminated Ideas

| Idea | Evidence-based reason |
|---|---|
| OOD/regret active query | OOD selection did not beat random on equal query budgets. |
| Diversity/core-set selection | No demonstrated closed-loop gain; selection cannot repair a biased teacher. |
| Frozen multi-horizon reliability gate (CRCD) | Reliability filtering amplified through-action imbalance and was negative online. |
| Hard/soft/ranking loss substitution | Did not remove high-K degradation under the frozen evaluator. |
| Differentiable SimExpert surrogate | Discrete-action semantics and exploitation risk remain unresolved. |
| Preference optimization as headline | Pairwise cost ranking already exists in this codebase; it would be rebranding. |

## Novelty Verification

The closest generic families are COMA, model-based/dual-policy imitation,
advantage-weighted regression, A2PR, and behavior preference regression.
PRCAT must not claim a new generic conservative offline RL objective. The
defensible core delta is the policy-responsive, multi-intersection simulator
oracle that evaluates a local phase deviation against the source policy's
dynamic network response, together with causal evidence for the
frozen-neighbor coordination bias. Recent Semantic Scholar search was
incomplete due HTTP 429; rerun it before submission and search the final PRCT
and PRCAT terminology explicitly.

## External Critical Review

A same-family adversarial reviewer judged the teacher mechanism as the only
current paper-level contribution and rejected PRCAT as a proven replacement.
Its minimum credible package is: Jinan-to-Hangzhou and Jinan-to-Jinan-2500
shifts; `K=118` with nine training seeds plus `K=32`; zero-shot, frozen C30,
unweighted responsive H120, PRCAT weight-only, KL-only, and full PRCAT arms;
AWT primary with ATT/AQL, p95/max movement wait, spillback, and action
entropy/frequencies as safety outcomes. It also requires evidence that high
PRCAT advantage predicts later benefit rather than merely encoding a local
min-max ranking. Therefore PRCAT is a conditional module, there is no policy
improvement guarantee, and a one-training-seed result is exploratory only.

### Pilot Outcome

The PRCAT full-loss pilot completed without training or online-evaluation
failures. It used one training seed (`2024`) and three independent evaluation
seeds (`2024`, `42`, `123`) on Jinan real:

| Method | Training seeds | Eval seeds | ATT | AQL | AWT |
|---|---:|---:|---:|---:|---:|
| zero-shot (existing) | n/a | 3 | 268.94 | 157.88 | 43.98 |
| responsive H120 K=118 (existing) | 2024 | 3 | n/a | n/a | 39.44 |
| responsive H120 K=118 (existing) | 9 | 3 each | 269.98 +/- 0.68 | 159.27 +/- 1.10 | 40.97 +/- 0.84 |
| PRCAT full K=118 | 1 | 3 | 270.78 +/- 0.17 | 161.30 +/- 0.34 | 42.48 +/- 0.22 |

The pilot clears the zero-shot baseline but fails the relevant matched teacher
comparison. It is recorded as a negative module result, not evidence for a
new method. Reproducible JSON and logs are under
`records/pipeline_runs/prcat_pilot_20260818/prcat/k118/tr2024/evaluation/`.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/PRCAT_EXPERIMENT_PLAN.md`
- Active contract: `idea-stage/docs/research_contract.md`

## Next Steps

1. Use PRCT, not PRCAT, as the Stage-3 replacement claim.
2. Run the frozen-teacher versus PRCT causal matrix on the two required shifts,
   adding starvation/safety metrics.
3. Leave PRCAT frozen as a negative pilot unless a materially different,
   preregistered robustness hypothesis is proposed.
4. Update novelty search after Semantic Scholar rate limit resets.

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** PASS

All required stage records, artifacts, and report sections are present.
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->
