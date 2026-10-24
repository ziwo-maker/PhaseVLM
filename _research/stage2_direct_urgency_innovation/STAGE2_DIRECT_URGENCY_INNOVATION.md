# Stage 2 Innovation: Direct Phase-Urgency Ranking Distillation

**Status (2026-08-23):** the direct-urgency method is fixed. Jinan follow-up
experiments now show that blind capacity scaling does not improve the main
line, while a short-online checkpoint selector makes `wait225_mixed_rank` the
best average-objective candidate among the tested wait-weighted arms. The
offline-`val_acc` comparator confirms checkpoint selection matters, while a
linear tail-risk selector improves P95/AWT trade-offs but should not replace
short-online reward ranking. A constrained selector is a cleaner P95/AWT branch
but still not a max-wait solution. A held-out selector-seed validation
(`selector_seed=42`) selected the same epoch as the original short-online
selector in all 6 arm/seed cells, so the selector appears stable to this seed
change. This is not a max-wait or broad online-generalization claim.

## One-Sentence Claim

Stage 2 is a **direct phase-urgency ranking distillation** method: it retains
the existing rule-derived next-phase label as an action anchor, while using the
same decision's observed phase queue and phase waiting time to supervise the
relative ordering of all four legal phases. It requires no candidate-action
rollout, no counterfactual teacher, and no policy-responsive simulation.

## Problem

Hard behavior cloning converts every traffic state into one phase label. At a
decision point, however, the other three phases are not equally undesirable:
some serve an almost-as-urgent queue, while others leave a large and old queue
unserved. A single label discards this ordinal structure.

The historical Stage 2 log also identifies a practical failure mode: a local,
counterfactual, min-max-normalized teacher score can be misaligned with the
3,600-second closed-loop metrics. The present method deliberately avoids this
source of objective mismatch. Its cost fields are not predictions of what
would happen after taking an alternative action. They are direct measurements
of what demand is currently waiting behind that phase.

## Direct Urgency Signal

Let the legal phase set be

\[
\mathcal A=\{\mathrm{ETWT},\mathrm{NTST},\mathrm{ELWL},\mathrm{NLSL}\}.
\]

At a decision state \(s\), the environment already exposes a phase-level queue
\(q_a(s)\) and mean phase waiting time \(w_a(s)\) for every \(a\in\mathcal A\).
The versioned Stage 2 record stores, for every labelled state,

\[
m_a(s)=\{\mathrm{att}=0,\ \mathrm{aql}=-q_a(s),\ \mathrm{awt}=-w_a(s),\ \mathrm{spillback}=0\}.
\]

The training implementation min-max normalizes valid action values *within
that decision state* and forms a lower-is-better cost

\[
c_{\lambda}(a\mid s)=
  \operatorname{Norm}_{\mathcal A}[-q_a(s)]
  +\lambda\operatorname{Norm}_{\mathcal A}[-w_a(s)].
\]

Thus an action with a larger currently waiting queue, or with a larger current
waiting time when \(\lambda>0\), receives a lower urgency cost and should rank
above less urgent phases. Equal values remain ties; they are not artificially
ordered.

The three mixed-loss settings are a controlled wait-priority sweep:

| Arm | Queue weight | Wait weight \(\lambda\) | Interpretation |
|---|---:|---:|---|
| `equal_mixed_rank` | 1.0 | 1.0 | Equal queue/wait urgency |
| `wait2_mixed_rank` | 1.0 | 2.0 | Wait-sensitive urgency |
| `wait25_mixed_rank` | 1.0 | 2.5 | Stronger wait-sensitive urgency |

`hard_bc` is the matched hard-label-only baseline. All arms use the same data,
Stage 1 checkpoint, training budget, split strategy, and offline checkpoint
criterion.

## Learning Objective

For a rule-derived expert phase \(y(s)\), the mixed objective combines three
roles:

1. **Hard action anchoring.** Cross-entropy preserves the existing operational
   rule label, including its queue-gap and left-turn wait-protection logic.
2. **Soft urgency preference.** A temperature-scaled distribution derived from
   \(c_{\lambda}\) gives nonzero credit to phases whose observed urgency is
   close to the selected phase.
3. **Pairwise ordinal consistency.** A weighted pairwise ranking term penalizes
   the policy whenever it scores a strictly more urgent observed phase below a
   less urgent one.

The pre-registered mixed configuration uses soft-target coefficient `0.3`,
temperature `0.35`, ranking weight `0.05`, and a minimum normalized cost gap
of `0.02`. `att` and `spillback` are set to zero and their loss weights are
zero, so this experiment isolates queue/wait urgency rather than silently
mixing a proxy for another objective.

## What Is Actually New Here

The following is the defensible contribution boundary.

1. **Four-action, directly observed urgency supervision for a VLM traffic
   policy.** Rather than simulate each candidate phase, all legal actions get a
   comparable preference signal from the same observed traffic state. This is
   materially cheaper, deterministic conditional on the logged state, and
   avoids teacher-policy and rollout-horizon assumptions.
2. **Anchor-plus-ordering supervision.** The rule label answers “which phase
   should be selected now?” while direct queue/wait costs answer “how should
   the remaining legal phases be ordered?” Combining the two prevents the
   ordinal term from replacing the existing operational rule with an untested
   scalar objective.
3. **An explicit and testable queue--delay policy knob.** The wait coefficient
   is not buried in a simulator reward. It changes one interpretable preference
   trade-off, with matched `1:1`, `1:2`, and `1:2.5` ablations and fixed
   closed-loop evaluation.
4. **A reproducibility-first label contract.** The direct-cost data preserve
   all 30,480 temporal records, retain the 1,920 existing terminal labels, and
   add input/output hashes and an unambiguous provenance tag:
   `observed_phase_queue_wait_no_rollout`.

This is a useful systems-and-learning contribution even if a similar ranking
loss has appeared elsewhere. It should be presented as a **traffic-specific
supervision design and empirical improvement**, not as the invention of
cost-sensitive learning to rank.

## Theoretical Interpretation

### Relation to max-pressure and max-weight control

Max-pressure control provides the main control-theoretic motivation: serving a
high-pressure movement is a queue-stabilizing principle when pressure contains
the appropriate upstream/downstream queue differential. Direct urgency adopts
the *local service priority* part of this idea: a phase with more currently
queued demand receives higher priority. The wait term is an age-like correction
that raises priority for demand that has waited longer.

This supports an **urgency-regularization interpretation**, not a theorem that
the learned VLM policy inherits max-pressure stability. The experiment lacks
downstream queue differentials, saturation-flow calibration, and an exact
max-weight argmax controller. Therefore no throughput-optimality or network
stability guarantee is claimed.

### Relation to delay fairness

Increasing the wait coefficient makes the policy less indifferent between a
large new queue and a smaller but older queue. This is consistent with the
motivation of fairness-aware and waiting-time-constrained traffic control.
It is not a fairness guarantee: mean wait is only an urgency signal, while
fairness and safety require explicit tail constraints and vehicle-level
evidence. The hardened online protocol therefore reports p95/max of each
observed non-shadow vehicle's maximum continuous wait at action boundaries.
These are vehicle-level diagnostics, but still do not establish a full-trip
tail-wait or safety guarantee.

### Relation to cost-sensitive ranking

Cost-sensitive learning-to-rank establishes that a pairwise/ordinal loss can
encode unequal action errors. Our novelty is not the generic loss. It is the
construction of a direct, rollout-free, phase-level cost order that is coupled
to a hard operational teacher and evaluated in closed loop.

## Explicit Non-Claims

This Stage 2 direction does **not**:

- estimate counterfactual action values or claim causal action superiority;
- run alternative actions in a simulator to generate labels;
- use policy-responsive, future-robust, JSCD, PRCT, DAgger, or continuation
  teachers;
- prove max-pressure stability, no-starvation, fairness, or spillback safety;
- treat online realization seeds as independent statistical samples;
- call same-road-network demand tests cross-network or out-of-distribution generalization;
- select a checkpoint using final online test performance.

These boundaries are important. They make the evidence easier to audit and
separate a useful direct-urgency method from previously rejected
counterfactual-label directions.

## Jinan_real r2 Protocol And Audit Boundary

### Data and provenance

| Role | Artifact |
|---|---|
| Training input | `data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl` |
| Versioned r2 input | `records/pipeline_runs/jinan_real_direct_urgency_20260821_r2/data/train_1440_direct_phase_urgency_v1.jsonl` |
| Results | `records/pipeline_runs/jinan_real_direct_urgency_20260821_r2/aggregate.json` |

The versioned r2 input contains 22,860 temporal records and 1,440 terminal rule
labels. The training label distribution is ETWT 508, NTST 471, ELWL 237, and
NLSL 224. The direct-urgency builder changes no expert action, prompt,
movement history, or unlabelled temporal record. It only enriches labelled
decisions with four observed costs.

All 1,440 r2 labelled decisions contain complete raw
`candidate_action_metrics`, so the ranking path never used the legacy
`candidate_action_waits` fallback. That fallback nevertheless stored queue
cost in the wait field in r2; it has been corrected for successor datasets and
must not be used to reinterpret the archived r2 JSONL. The 108 r2 raw online
results also contain vehicle-tail diagnostics, but aggregate-v1/CSV omitted
them. Those historical artifacts remain immutable; hardened successor
aggregation preserves the tail fields.

Training-input SHA-256:
`da3a7170346a344a87edde9b509c5c48e4b2317f6a0f5b7fce17f1c6c285cffb`.
Versioned-output SHA-256:
`839eb6d97903b913b9ba95737add9ba19d9650523da1e3af801a285df630cd7b`.

### Executed matrix and result boundary

- Four arms: `hard_bc`, `equal_mixed_rank`, `wait2_mixed_rank`, and
  `wait25_mixed_rank`.
- Three training seeds: `2024`, `42`, and `123` (12 models).
- Common initializer:
  `jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt`.
- Common model/interface: 16-step, 224-pixel sequence image, 8 prefix tokens,
  hidden dimension 64, and `queue_wait_short` prompt.
- Primary closed-loop protocol: `anon_3_4_jinan_real.json`, 3,600 seconds,
  online seed 2024, and no max-phase override.
- Demand-shift checks: `anon_3_4_jinan_real_2000.json` and
  `anon_3_4_jinan_real_2500.json`, with online seeds `2024`, `42`, and `123`.

All three flows use the Jinan `3x4` road network. The primary flow is also the
source scenario for the logged training decisions. Therefore this matrix is
**same-network closed-loop validation plus demand-shift robustness**, not
cross-network transfer or out-of-distribution generalization. The primary
statistical unit is the training seed. Online seeds are realization checks and
must not be pooled as independent samples.

The r2 campaign used a random stratified offline split before this audit. Its
`val_acc` is suitable only for its narrow checkpoint-selection role, not as a
held-out temporal generalization result. The r2 primary deltas do not support
a net `wait2` improvement over hard BC: it lowers mean wait but worsens mean
reward, queue length, and travel time. `wait25` has an unstable seed with a
large congestion collapse. Those results are retained as negative evidence,
not reported as a performance gain.

### Hardened successor protocol

Subsequent Jinan campaigns use a blocked suffix validation split **per source
file, flow, and intersection trajectory**, with a 24-sample purge buffer before
each holdout boundary. This protects the 16-step observation histories despite
the source JSONL interleaving records from different intersections. Checkpoints
in the original hardened r3-style campaigns remain selected only by the
predeclared offline `val_acc`; final online metrics are never used for
selection. The later online-checkpoint-selection follow-up deliberately tests a
separate selector that uses a short 1,200-second same-flow online validation
probe before the final 3,600-second online evaluation. Each online seed is
evaluated with deterministic PyTorch settings, deterministic cuDNN, and TF32
disabled. This removes the prior fixed seed-to-GPU confound.

Every new campaign writes a `provenance.env` manifest containing data,
initializer, code, Git HEAD, and working-tree-diff hashes. Each online result
records the evaluated checkpoint and evaluator hashes. The aggregator is strict
by default and rejects results with an unexpected checkpoint, missing hash,
enabled override, wrong duration, or missing deterministic-runtime evidence;
historical r2 requires its explicit legacy flag. It also aggregates p95
and maximum observed vehicle wait alongside the four primary metrics.

## Jinan Online-Checkpoint Selection Follow-Up

Run root:
`records/pipeline_runs/jinan_online_checkpoint_selection_20260822_r2`

The follow-up tests whether checkpoint selection should remain pure offline
`val_acc` or use a small online validation probe. For each of `wait2_mixed_rank`
and `wait225_mixed_rank`, and for training seeds `2024`, `42`, and `123`, the
campaign saves epoch checkpoints, evaluates them on a 1,200-second same-flow
online probe with seed `2024`, promotes the best short-online-reward
checkpoint, and then runs the 3,600-second same-flow online suite.

Completion audit: 18/18 final online rows completed, 0 incomplete.

Primary online seed `2024`; values are mean ± std over 3 training seeds.

| Arm | Reward | AQL | AWT | ATT | P95 wait | Max wait |
|---|---:|---:|---:|---:|---:|---:|
| `wait225_mixed_rank` | **-624.280 ± 1.363** | **175.228 ± 2.782** | **43.447 ± 0.781** | **280.462 ± 1.806** | **180.667 ± 3.512** | 583.333 ± 42.724 |
| `wait2_mixed_rank` | -628.029 ± 7.148 | 177.247 ± 5.323 | 45.517 ± 0.686 | 281.203 ± 3.369 | 184.333 ± 4.726 | **548.667 ± 30.567** |

The short-online selector chose a different epoch than offline `val_acc` in all
six arm/seed cells. This confirms that checkpoint selection is a material
experimental variable. The current result supports `wait225_mixed_rank` as the
best average-objective candidate under this selector, but it worsens maximum
wait and is not a tail-safety result. Because the selector and the primary
final online result both use online seed `2024`, the next strict test is a
matched final-online comparison against the saved offline-`val_acc`
checkpoints, preferably with a held-out final online seed.

### Selector Comparator Updates

The matched offline-`val_acc` comparator completed at
`records/pipeline_runs/jinan_online_checkpoint_selection_valacc_comparator_20260823`
with 18/18 offline rows and 0 incomplete. It shows that short-online selection
has small, noisy mean gains over offline `val_acc`, but it strongly reduces
`wait2_mixed_rank` maximum wait from 748.667 to 548.667.

The tail-risk selector completed at
`records/pipeline_runs/jinan_online_checkpoint_selection_tailrisk_20260823_r3`
with 18/18 tail-risk rows and 0 incomplete. Its linear score
`short_reward - 0.10 * P95_wait - 0.02 * Max_wait` improves P95 for both arms
and improves `wait2_mixed_rank` AWT, but it worsens reward and does not beat
short-online on final maximum wait. This suggests that the short-eval max-wait
signal is too noisy for direct scalarization.

The next selector experiment is constrained rather than fully scalarized: rank
checkpoints by short-online Reward, keep the top 3, and then choose the
checkpoint with the lowest `P95_wait + 0.05 * Max_wait` inside that top-3 pool.

The constrained selector completed at
`records/pipeline_runs/jinan_online_checkpoint_selection_constrained_20260823`
with 18/18 constrained rows and 0 incomplete. It lowers P95/AWT relative to
short-online reward ranking, but still worsens final maximum wait. The main
average-objective line remains `short_online + wait225_mixed_rank`; the
constrained selector should only be reported as a P95/AWT tail branch.

The held-out selector-seed validation completed at
`records/pipeline_runs/jinan_online_checkpoint_selection_selector_seed42_20260823`
with 18/18 final online rows, 90/90 short-eval rows, and 0 incomplete. It
changed the short-online selector seed from `2024` to `42`, while keeping the
same trained `epoch_*.pt` candidates. The selected epoch was unchanged in all
6 arm/seed cells, and the selector-seed `2024` versus `42` candidate rankings
had Spearman correlation around 0.90-0.93. This supports robustness of the
short-online reward selector to this seed change. It does not change the main
decision: `short_online + wait225_mixed_rank` remains the average-objective
line, and maximum-wait improvement requires a new explicit protection or
tail-wait-teacher mechanism rather than more selector scalarization.

## Related Work and Positioning

| Work | What it establishes | Difference from direct urgency |
|---|---|---|
| Varaiya, 2013, *Transportation Research Part C* | Max-pressure can stabilize signalized networks under its assumptions. DOI: [10.1016/j.trc.2013.08.014](https://doi.org/10.1016/j.trc.2013.08.014) | The present method learns a local queue/wait ordinal preference; it does not implement or prove max-pressure control. |
| Wei et al., 2019, *PressLight*, KDD | Learns max-pressure signal control. DOI: [10.1145/3292500.3330949](https://doi.org/10.1145/3292500.3330949) | PressLight is an RL/max-pressure controller; here queue/wait are direct supervision for a pretrained VLM policy, anchored by a rule label. |
| McBride, Wang, Ren, and Li, 2019, AAAI | Generic cost-sensitive learning to rank. DOI: [10.1609/aaai.v33i01.33014570](https://doi.org/10.1609/aaai.v33i01.33014570) | The ranking loss is established machinery; the new element is its observed phase-urgency cost construction and closed-loop traffic validation. |
| Shirasaka, Kodama, and Harada, 2023, SICE ISCS | Fairness can be made explicit in distributed DRL traffic signal control. DOI: [10.23919/siceiscs57194.2023.10079200](https://doi.org/10.23919/siceiscs57194.2023.10079200) | Direct urgency emphasizes wait in supervision but does not claim its fairness objective or guarantee. |
| Iizuka and Seo, 2024, *International Journal of Intelligent Transportation Systems Research* | Waiting-time and spillback constraints matter in passenger-oriented signal control. DOI: [10.1007/s13177-024-00418-z](https://doi.org/10.1007/s13177-024-00418-z) | The present method does not optimize an explicit constrained program; it keeps spillback out of this isolated queue/wait ablation. |

## Reproducibility Artifacts

- Dataset builder: `scripts/build_direct_urgency_stage2_dataset.py`
- Per-arm launcher: `scripts/launch_jinan_direct_urgency_arm.sh`
- Campaign and startup-evidence logger:
  `scripts/run_jinan_direct_urgency_campaign.sh`
- Online suite: `scripts/run_jinan_direct_urgency_online_suite.sh`
- Result aggregator: `scripts/aggregate_jinan_direct_urgency.py`
- Executed r2 root: `records/pipeline_runs/jinan_real_direct_urgency_20260821_r2/`
- Online checkpoint selection campaign:
  `scripts/run_jinan_online_checkpoint_selection_campaign.sh`
- Offline-`val_acc` comparator:
  `scripts/run_jinan_online_checkpoint_selection_valacc_comparator.sh`
- Tail-risk selector comparator:
  `scripts/run_jinan_online_checkpoint_selection_tailrisk_comparator.sh`
- Constrained selector comparator:
  `scripts/run_jinan_online_checkpoint_selection_constrained_comparator.sh`
- Held-out selector-seed validation:
  `scripts/run_jinan_online_checkpoint_selection_selector_seed42_campaign.sh`
- Online checkpoint selection result analysis:
  `stage2_direct_urgency_innovation/anon_3_4_jinan_real/jinan_online_checkpoint_selection_20260822/RESULT_ANALYSIS.md`

For numerical claims, cite the generated `AGGREGATE.md`, `aggregate.json`,
`online_results.csv`, raw `result.json` files, and the campaign provenance
manifest. Do not claim an overall gain unless a future hardened campaign shows
consistent paired improvements across the four primary metrics and tail-wait
diagnostics.
