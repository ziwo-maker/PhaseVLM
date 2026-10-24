# A-Override Correction Distillation: Ablation Plan

Date: 2026-08-12

## Decision Summary

The current R1/R2 result is a strong single-run causal comparison, but it is not
yet enough for all four intended claims. The paper-critical ablations are:

1. Repeat the paired uncorrected-versus-distilled comparison with three genuine
   training seeds and held-out traffic realizations.
2. Compare distilled inference directly with runtime A-override using the same
   base checkpoints and an equivalence test, not a test for non-significance.
3. Add label-marginal-matched and shuffled-label negative controls to show that
   the gain comes from the state-conditional correction, rather than action
   rebalancing or generic label perturbation.
4. Measure a nested correction-ratio curve while holding the candidate
   correction pool fixed.
5. Ablate `gap` and left-turn protection on fixed states first, then confirm the
   important variants with fresh closed-loop data collection.
6. Run a 2x2 Stage-1-source factorial and a second target-network replication.

All headline evaluations must disable runtime A-override. Runtime override is
enabled only in the explicitly named teacher/equivalence cells.

## Experimental Contract

### Unit of analysis and pairing

- The experimental unit is a complete traffic episode, not an intersection,
  decision row, or time step.
- Use training seeds `2024`, `2025`, and `2026`. Pair variants by Stage 1
  checkpoint, data split, initialization seed, training schedule, and traffic
  realization.
- A different CLI seed is not a replication if the CityFlow demand and route
  realization remain deterministic. Create or identify genuinely different
  route/flow realizations and record their hashes.
- Report the paired effect for each metric with a 95% confidence interval. Use
  a hierarchical paired bootstrap over training seeds and traffic
  realizations; do not treat the 1,920 decisions as independent samples.
- Freeze the primary comparisons and analysis before seeing new test results.
  Apply Holm correction within each ablation family.

### Leakage and checkpoint selection

- Split by complete rollout/traffic realization before expanding a decision
  into history rows. No history from a held-out decision or episode may appear
  in training.
- The current dataset and evaluation use the same named Hangzhou traffic file.
  Reusing a deterministic trajectory would support imitation on that trajectory,
  but not the claim that the model learned a general correction. Use disjoint
  collection, validation, and test traffic realizations.
- Do not select checkpoints by the final online test. Because validation F1 has
  previously been anti-correlated with online control quality, either use the
  final epoch for every cell or select from a disjoint validation traffic set
  using one predeclared online metric. Apply the same rule to every cell.
- Keep architecture, optimizer, epochs, split, prompt, and Stage 1 checkpoint
  fixed within each paired comparison. The only changed field should be the
  ablated factor.

### Outcomes

Primary effectiveness outcomes:

- Episode queue, waiting time, and travel time, all lower-is-better.
- The existing `Sum = queue + wait + travel` for continuity with current
  results. Since it mixes quantities with different meanings, do not use Sum
  alone; also report each component and a predeclared baseline-normalized
  composite.

Safety and mechanism outcomes:

- Per-movement median, p95, and maximum wait; left-turn starvation events above
  60 s and 120 s; left-turn action rate.
- Max-phase match, average missed queue gap, strong misses, and action entropy.
- On held-out states, teacher agreement on trigger states, correction
  acquisition, and introduced-error rate on non-trigger states.
- Residual runtime-override rate when A-override is placed on top of the
  distilled model. A low residual rate is mechanistic evidence that the
  correction was internalized.

For the statement "matches runtime override," predeclare an equivalence margin.
A defensible starting margin is a paired 5% relative difference in normalized
composite/primary Sum, with no individual safety metric more than 10% worse.
Use TOST or an equivalent confidence-interval containment test. A non-significant
difference is not evidence of equivalence.

## Component and Causal Ablations

| ID | Name / change | What it tests | Expected if the component matters | Priority |
|---|---|---|---|---:|
| C0 | Remove all A-override corrections: R1, 0/319 labels replaced | Direct contribution of targeted corrected supervision | R1 is worse than R2 on held-out episode metrics, with lower teacher agreement and more strong misses | 1 |
| C1 | Four-way deployment comparison: uncorrected/no runtime, uncorrected/runtime, distilled/no runtime, distilled/runtime | Whether distillation reproduces the runtime teacher and how much correction remains | Distilled/no-runtime is equivalent to uncorrected/runtime; adding runtime override to distilled gives little extra gain and fires rarely | 1 |
| C2 | Action-marginal-matched random relabeling: alter the same number of labels so the final action histogram exactly matches R2, but choose unrelated states | Whether class rebalancing or generic label noise explains the gain | This control is substantially worse than state-conditionally targeted R2 despite identical correction count and label histogram | 1 |
| C3 | Shuffled correction targets: use the same 319 states and the same corrected-label histogram, but derange corrected labels across states and remove accidental no-ops | Whether the corrected action must be aligned with the state | The benefit disappears or reverses; introduced-error rate rises | 2 |
| C4 | Remove left-turn wait protection: `protect=0` (the implementation's special value for disabled protection) | Whether protection prevents unsafe over-clearing of through phases | Average queue may improve slightly, but left-turn p95/max wait, starvation count, or left-turn action rate worsens if protection matters | 1 |
| C5 | Indiscriminate max-queue relabeling: correct every model/max-queue disagreement, including gaps below 2 | Whether selective correction is better than "always imitate max queue" | More labels change, but wait tails and/or total performance worsen because low-gap decisions contain ambiguity/noise | 2 |
| C6 | Corrections-only fine-tuning: train on the 319 corrected decisions without unchanged anchor decisions | Whether unchanged examples prevent forgetting and action collapse | Trigger-state agreement rises but non-trigger retention, action balance, and closed-loop performance worsen | 3 |
| C7 | 2x2 Stage 1 source factorial: Jinan vs Hangzhou-native, each with uncorrected vs corrected labels | Whether the correction effect survives different representation initialization and whether historical source comparisons were confounded | A positive correction main effect appears under both sources; the source-by-correction interaction quantifies how Stage 1 quality changes effect size | 1 |
| C8 | Second target-network replication: paired uncorrected vs distilled training on Jinan target data | Whether the method, rather than one Hangzhou trajectory, generalizes across networks | The paired correction effect has the same direction on Jinan, although magnitude may differ | 1 |

Implementation note for C2/C3: preserve row count, changed-label count, train/val
split, and final action histogram. Report how many generated changes are true
label changes after excluding no-ops.

## Hyperparameter Sensitivity

Run these on one frozen set of collected states first. Recompute only the target
labels from saved queue/wait diagnostics. This isolates the rule parameter from
the state-distribution change caused by deploying a different rule.

| ID | Parameter / values | What it tests | Expected if the component matters | Priority |
|---|---|---|---|---:|
| H1 | Gap: `2`, `3`, `5` | Sensitivity to correction confidence/intervention strength | Correction count decreases monotonically with gap; gap 2 or 3 beats gap 5, with gap 3 possibly trading queue for wait | 1 |
| H2 | Protection threshold: disabled (`0`), `30`, `60`, `90` seconds | Whether 60 s is a robust safety/efficiency choice rather than a tuned point | Moderate protection retains left-turn safety; disabled/90 may over-clear through phases, while 30 may suppress useful corrections | 1 |
| H3 | Fraction of the fixed 319-label pool: `0`, `25%`, `50%`, `75%`, `100%` = approximately `0`, `80`, `160`, `239`, `319` labels (`0`, `4.2`, `8.3`, `12.4`, `16.6%` of decisions) | Sample efficiency and the "small fraction" claim | Performance improves before 100% and then saturates; if only 25-50% is sufficient, the small-targeted-set claim is stronger | 1 |
| H4 | Hard-to-soft correction strength `alpha`: `0`, `0.5`, `1.0`, where corrected rows use `(1-alpha)y_original + alpha y_override` | Whether hard replacement is needed or teacher uncertainty helps | If the rule is reliable, alpha 1 is best; if boundary corrections are noisy, alpha 0.5 may improve robustness | 2 |
| H5 | Corrected-row loss weight: `0.5`, `1`, `2`, `4` with hard targets | Whether rare corrections need importance weighting | A broad plateau supports robustness; a sharp optimum would weaken the simplicity claim | 3 |

For H3, use nested masks so larger fractions contain all labels from smaller
fractions. Stratify by intersection, source action, target action, and queue-gap
bin. Pair each of the three training seeds with an independently generated
stratified nested mask. As a secondary diagnostic, compare the 25% stratified
mask with the top-25%-by-gap mask; do not replace the main ratio curve with an
unreplicated hand-selected subset.

The existing `--target_mode mixed` is not a clean implementation of H4: it
constructs cost-based targets from candidate metrics and can affect rows beyond
the A-override corrections. H4 needs a per-row A-override target distribution
that is one-hot on unchanged rows and mixed only on corrected rows.

## Natural Design Alternatives

| ID | Name / change | What it tests | Expected if the current design matters | Priority |
|---|---|---|---|---:|
| D1 | Fixed-state relabeling versus fresh closed-loop collection for the winning gap/protection variants | Separates label-rule quality from the state distribution induced during collection | The qualitative ranking persists; a large reversal identifies rollout-distribution effects that must be described as part of the method | 1 |
| D2 | Replace labels versus append corrected duplicates while retaining original examples | Whether destructive replacement is better than contradictory augmented supervision | Replacement is cleaner; appending creates ambiguous targets and is no better unless carefully weighted | 3 |
| D3 | One-shot distillation versus a second collect-correct-train round | Whether residual systematic errors require iterative dataset aggregation | Round two has a smaller correction rate and diminishing gains if one-shot distillation captures the rule | 4 |
| D4 | Max-queue teacher versus a simple max-pressure/queue-plus-wait teacher at a matched correction rate | Whether gains are specific to the A-override rule or generic targeted teacher correction | A-override should remain competitive; a better alternative would narrow the paper's claim to targeted correction distillation rather than this teacher | 4 |

## Generalization Matrix

For each selected base/distilled pair, evaluate without runtime override on
traffic realizations disjoint from collection and validation:

| Axis | Minimum levels | Purpose |
|---|---|---|
| Target network | Hangzhou 4x4, Jinan | Supports method-level cross-network replication |
| Demand | `0.8x`, `1.0x`, `1.2x` | Tests low, nominal, and congested regimes |
| Traffic realization | 3 per demand level | Prevents a deterministic route file from masquerading as multi-seed evidence |
| Training seed | 3 | Measures optimization variance |

This matrix supports "the procedure works when applied on multiple networks and
loads." It does not by itself support zero-shot transfer of one distilled model
to an unseen topology. Make that stronger claim only if a separate zero-shot
experiment is run.

## Claim Coverage Assessment

| Intended claim | Required evidence | Status after plan |
|---|---|---|
| A-override distillation is effective | C0 with paired seeds/held-out traffic, plus C2/C3 negative controls | Direct causal effect and state-label specificity are covered |
| Distilled model matches runtime override | C1 equivalence test, held-out teacher agreement, residual override rate | Functional equivalence and internalization are covered; do not describe this as proof of identical policies |
| Only a small fraction needs correction | H3 ratio curve plus C5 indiscriminate relabeling | Shows the minimum useful fraction and whether selectivity beats more relabeling |
| Method is general | C7, C8, and the network/demand matrix | Covers source robustness, second target network, and load shift |
| Left-turn protection is necessary | C4/H2 with tail safety outcomes | Covers both average efficiency and starvation risk |
| `gap=2`, hard replacement are reasonable choices | H1/H4 and D1 | Covers local sensitivity and the natural soft-target alternative |

## Suggested Run Order

1. **Audit and freeze the protocol.** Hash source data/checkpoints, construct
   rollout-level splits, verify that traffic realizations differ, and declare
   checkpoint selection, metrics, equivalence margins, and correction masks.
2. **Core causal replication.** Run C0 and standard R2 for three paired training
   seeds. Stop and diagnose if the direction is inconsistent before expanding.
3. **Runtime equivalence.** Run C1 on those same checkpoints and test the
   predeclared equivalence margin. Compute held-out agreement and residual
   override diagnostics.
4. **Cheap causal controls.** Run C2 and C3. These provide more reviewer value
   than a broad hyperparameter sweep because they rule out class-distribution
   and generic-label-change explanations.
5. **Correction efficiency.** Run H3, followed by C5. Plot performance against
   both changed-label count and actual intervention rate.
6. **Rule components.** Run fixed-state H1 and H2/C4. Confirm only gap 3 and
   no-protection (or whichever cells change the conclusion) with D1 fresh
   closed-loop collection.
7. **Source and target generality.** Run C7, C8, then the held-out demand matrix.
8. **Label design.** Run the alpha 0.5 H4 cell first. Expand to alpha 0.25/0.75
   or H5 only if it is competitive with hard replacement.
9. **Nice-to-have alternatives.** Run C6/D2-D4 only if reviewer positioning or
   residual budget justifies them.

## Feasibility and Compute

Existing support:

- Hard R1/R2 training and no-override evaluation already exist in
  `scripts/run_a_override_distill_ablation.sh`.
- Runtime `gap` and protection flags are implemented in
  `projection_train/run_projection_online.py`; `protect=0` disables protection.
- The current dataset builder is hard-coded to R1/R2. C2/C3/C5 and H1-H3 need a
  deterministic parameterized builder with a manifest and assertions.
- H4 needs a small training/data-path change because current mixed targets are
  cost-based, not A-override-specific soft labels.

Budget accounting uses the supplied costs: 0.30 GPU-hours per training and
0.15 GPU-hours per evaluation/collection rollout.

| Must-run paper set | Training runs | Eval runs | Collection runs | Estimated GPU-hours |
|---|---:|---:|---:|---:|
| Hangzhou C0/C1/C7: 2 Stage 1 sources x 2 label conditions x 3 training seeds; no-runtime/runtime evaluation on 3 nominal held-out traffic realizations | 12 | 72 | 0 | 14.40 |
| C2/C3 causal controls: 2 variants x 3 training seeds x 3 held-out traffic realizations | 6 | 18 | 0 | 4.50 |
| H3 intermediate fractions plus C5: 4 new variants x 3 training seeds x 3 held-out traffic realizations | 12 | 36 | 0 | 9.00 |
| Fixed-state H1/H2 variants beyond the reused default: 5 variants x 3 training seeds x 3 held-out traffic realizations | 15 | 45 | 0 | 11.25 |
| Jinan target C8: 2 label conditions x 3 training seeds, evaluated at 3 demand levels x 3 held-out traffic realizations | 6 | 54 | 0 | 9.90 |
| **Must-run total** | **51** | **225** | **0** | **49.05** |

This leaves 0.95 GPU-hours for up to six 0.15-GPU-hour smoke/collection probes,
or one failed train-plus-eval retry. If the Jinan corrected dataset must be
recollected, take its collection rollout from this buffer.

The full plan exceeds 50 GPU-hours. The following cells are explicitly deferred,
not silently dropped: H4/H5, C6, and D1-D4, plus the Hangzhou low/high-demand
expansion. Promote H4 alpha 0.5 first if a must-run cell can be reused or
cancelled; promote D1 next if fixed-state versus fresh-collection confounding
becomes material. If failures exhaust the 0.95-hour buffer, cut C3 first and
state that generic perturbation is controlled only by the stronger
action-marginal-matched C2. Do not cut C0/C1, C2, H3, C4/H1, C7/C8, or held-out
traffic evaluation without narrowing the associated paper claim.

With three RTX 5090s, one two-GPU training can run alongside one one-GPU online
evaluation. When no training is active, independent online evaluations can use
one GPU each. Keep output directories, result JSON files, and traffic seeds
separate. All actual launches remain subject to the project's requirement to
show exact commands and receive fresh explicit approval.

## Unnecessary or Misleading Ablations

- Do not run an exhaustive integer sweep over gaps 1-10. Values 2/3/5 answer
  the sensitivity question; runtime gap 8 has already been shown to intervene
  only 7/1,920 times and adds little information.
- Do not repeat target/total queue-floor variants. Existing runtime experiments
  already show that these floors remove useful corrections and degrade all main
  outcomes.
- Do not use PPO variants, frozen-backbone removal, prefix length, prompt length,
  or optimizer sweeps as A-override component ablations. They answer different
  architecture/optimization questions.
- Do not relabel all 30,480 history rows. Only terminal decision labels define
  the treatment; modifying history labels changes the task.
- Do not treat repeated deterministic simulations under different nominal seeds
  as independent evidence. Identical metrics/trajectories are one realization.
- Do not claim "learned, not memorized" from evaluation on the same deterministic
  route file used to collect corrections. Held-out traffic is mandatory.
- Do not call hard versus the existing generic `target_mode=mixed` a clean label
  replacement ablation; it changes the target construction beyond corrected
  rows.
- Do not add many correction weights or soft-target temperatures before the
  causal controls and ratio curve are complete. They consume budget without
  addressing the strongest reviewer objections.

## Required Result Artifacts

For every cell, record the dataset/checkpoint hashes, training seed, traffic
realization, exact changed-label count and action histogram, best/final epoch,
all episode and safety metrics, and whether runtime override was enabled. Keep
negative results. Produce:

1. A main paired table for uncorrected, runtime, and distilled methods.
2. A correction-ratio curve with confidence intervals.
3. Gap/protection sensitivity plots including left-turn tail metrics.
4. A source x correction interaction table.
5. A network x demand generalization table.
6. A mechanism table with trigger-state agreement, introduced errors, and
   residual override rate.
