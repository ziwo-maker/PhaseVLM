# Experiment Results & Iteration Analysis — 2026-08-12

## Summary of All Experiments

### Phase 0: SimExpert Oracle Validation ✅
- 4,776 records (Jinan + Hangzhou), all checks PASSED
- Oracle is stable and reliable

### Phase 1: Bottleneck Diagnostic (4 configs × 2000 samples) ✅
| Config | Acc | ρ | ECE | Diagnosis |
|--------|-----|---|-----|-----------|
| Jinan→Jinan | 39.6% | 0.157 | 0.447 | representation_failure |
| Jinan→Hangzhou | 40.8% | 0.239 | 0.414 | representation_failure |
| Hangzhou→Hangzhou | 40.8% | 0.229 | 0.385 | representation_failure |
| Hangzhou→Jinan | 39.6% | 0.157 | 0.409 | representation_failure |

### Phase 2: Active Query (6 methods × 5 K × 3 seeds × 2000 states) ✅
- OOD consistently best: K=8 regret=0.43, agreement=75%
- Regret-aware/ensemble worst due to simulated MC dropout
- Core-set competitive at small K

### Phase 3: Temperature & Bias Analysis ✅
- **100% ETWT mode collapse confirmed** — all models predict ETWT always
- Temperature scaling (T=0.1 to 10.0) cannot help (logit ordering is flat)
- Logit bias correction only shifts collapse to a different action, doesn't fix ordering
- ρ is fundamentally limited by prefix capacity, not calibration

## Root Cause: Prefix Capacity Insufficient

**All existing checkpoints**: prefix_tokens=8, hidden_dim=64

The vision→prefix projector has only ~512K parameters. With only 8 prefix tokens, the model cannot encode enough state information to distinguish different traffic conditions. This causes:
1. **Mode collapse to ETWT** (most common action in data)
2. **Flat logits** — ρ values of 0.08-0.23 mean near-random ranking
3. **Temperature/bias can't fix** — the problem is in representation, not decoding

## Iteration Plan

### Must-Do: Architecture Retraining
| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| prefix_tokens | 8 | 32 | 4× more capacity for state encoding |
| hidden_dim | 64 | 256 | 4× larger projector hidden layer |
| max_length | 64 | 128 | Longer context window |

Estimated: ~6-8 GPU-hours × 3 seeds × 2 networks ≈ 36-48 GPU-hours on 1× RTX 5090

### Quick Wins (no retraining needed)
- Random small-budget SimExpert adaptation for cross-network transfer (validated at K=8)
- Action-specific temperature scaling
- Class-balanced evaluation metrics

### Gate Before Next Steps
- [ ] Retrain with prefix_tokens=32, hidden_dim=256
- [ ] Verify ρ > 0.4 on in-network test
- [ ] Then proceed to active query + ablation studies

---

## Phase 4: Stage 3 OOD-Guided Cross-Network Adaptation (2026-08-13)

### Setup and Validity

- Source policy: Jinan Stage-2 pt32/hd256 checkpoint (val_acc=73.5%); target: Hangzhou.
- Collection flow and evaluation flow are disjoint: `anon_4_4_hangzhou_real.json` vs. `anon_4_4_hangzhou_real_5816.json`.
- 120 collection snapshots were requested; 118 were usable after two corrupted zero-byte snapshots were removed.
- Each selected snapshot received full SimExpert counterfactual labels with yellow-transition emulation, Pareto filtering, and coverage verification.
- All final arms completed successfully. Earlier `training_seed` and corrupt-snapshot failures in `main.log` were from superseded attempts, not the final artifacts.

### Online Results

| Arm | ATT | AQL | AWT | Sum | Delta Sum vs. zero-shot |
|-----|----:|----:|----:|----:|----:|
| zero-shot (K=0) | 335.63 | 199.32 | 67.75 | 602.70 | -- |
| random K=8 | 330.14 | 189.78 | **61.73** | **581.65** | -21.05 |
| random K=32 | **327.50** | **190.53** | 66.19 | 584.22 | -18.48 |
| random K=118 | 337.86 | 206.08 | 81.99 | 625.93 | +23.23 |
| raw-OOD K=8 | 331.67 | 195.33 | 72.19 | 599.19 | -3.51 |
| raw-OOD K=32 | 328.69 | 192.01 | 67.07 | 587.77 | -14.93 |
| raw-OOD K=118 | 334.47 | 202.57 | 83.32 | 620.36 | +17.66 |

### Supported Conclusions

1. **Small-budget target adaptation works.** Random K=8 improves every online metric over zero-shot, including AWT by 8.9% (67.75 -> 61.73) and total Sum by 21.05.
2. **The current raw-OOD selector does not show an advantage over random selection.** At K=8, raw-OOD has worse AWT than random (72.19 vs. 61.73, +16.9%); at K=32 it is close but still worse on all three reported metrics. Thus the result is negative for this selector, not for target-domain SimExpert adaptation itself.
3. **Large-budget degradation is robust to the acquisition method.** Moving from K=32 to K=118 raises AWT by 23.9% for random and 24.2% for raw-OOD, while ATT and AQL also worsen. This reproduces the old high-K failure with full counterfactual labels and the larger pt32/hd256 architecture.
4. **The current result does not establish statistical superiority of random.** Each arm used one training seed (`2024`); the three reported seeds are online-evaluation seeds, not independent retraining repetitions. Most evaluation outputs are identical across those seeds. Therefore t-intervals here do not estimate selection/training variance. The defensible claim is: raw-OOD did not demonstrate an advantage in this run.

### Diagnostic Evidence

#### 1. Raw-OOD selects extreme demand states, not proven high-value labels

The selector computes a mean diagonal-Mahalanobis distance from 16 raw state features to a global Jinan source distribution: 8 movement counts, 4 phase queues, and 4 phase waits. It does not use policy uncertainty, policy-expert disagreement, counterfactual value, long-horizon return, road capacity, topology, or model representation.

- Raw-OOD K=8 chooses only times 810-1140s, a 330s span in two 15-minute windows. Random K=8 spans 510-2760s across all four 15-minute windows.
- Raw-OOD K=8 does target higher-load states: mean movement sum 16.71 vs. 13.23 and mean phase-wait sum 81.89 vs. 65.68 for random K=8.
- It also has larger **30-second** SimExpert AWT improvement over the source action (94.38 vs. 81.38), but gives worse **3600-second closed-loop** AWT. Therefore instantaneous congestion and local counterfactual gain are not valid proxies for adaptation value in this setting.
- Across all 118 usable snapshots, OOD score correlation with expert-action flip rate is -0.28, with expert margin is -0.18, and with local composite-cost gain is -0.23. High distance does not identify more corrective or more decisive labels.

#### 2. Batch redundancy is a direct selection failure mode

The selector ranks snapshots independently and takes top-K, without temporal, state-space, or topology-aware diversity. Consecutive high-demand snapshots are highly correlated but consume the full label budget. This explains why K=8 raw-OOD receives concentrated peak-state supervision whereas random covers the episode's demand regimes.

#### 3. The SimExpert training target is misaligned with the deployment objective

For each state-action candidate, SimExpert:

- freezes all non-target intersections at snapshot phases;
- simulates only 30 seconds;
- uses occupancy as an ATT proxy, queue length, accumulated vehicle-seconds as AWT, and a spillback proxy;
- min-max normalizes every metric across the four actions **within each individual snapshot** before forming a composite score.

The adaptation loss then uses these per-snapshot scores in both soft targets and pairwise ranking. More K therefore gives the short-horizon, locally normalized objective more influence. It need not optimize 3600-second online ATT/AQL/AWT, and this objective mismatch is the leading explanation for high-K degradation.

#### 4. Closed-loop distribution shift remains after offline adaptation

All labels come from one collection episode induced by the source policy. A fine-tuned policy changes future states but is not relabeled on its induced distribution. The adaptation command also sets `base_policy_kl_weight=0.0`, so there is no conservative anchor to the source policy. This is the standard behavioral-cloning covariate-shift failure mode in sequential control.

#### 5. Source-distribution estimate and oracle replay need stronger controls

- Source statistics use 4,320 complete records out of 68,580 input records (6.30%). Strict filtering avoids zero-imputation, but the remaining statistic is still a global, diagonal model over a small complete subset.
- The largest mean distance contributions are movement-count dimensions (`ET`, `ST`, `WT`, `NT`), indicating that the score mainly reflects demand-pattern mismatch rather than a policy-relevant representation mismatch.
- At K=118, random and raw-OOD selected byte-identical snapshot inputs and identical snapshot splits. Expert actions were identical for all 1,888 records, but 28 candidate-cost entries, 20 margins, and 3 feasibility masks differed across separate SimExpert replays. The differences are small and do not overturn the high-K degradation finding, but they invalidate the stronger claim that the two training datasets differed only in selected epoch. The oracle should be replay-tested and made bitwise deterministic before it is used as a variance-control baseline.

### Literature-Grounded Interpretation

- Batch active learning should balance informativeness with coverage/diversity; pure top-K ranking can select redundant regions. This is the motivation behind BADGE and core-set selection. See Ash et al., *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds* (arXiv:1906.03671), and Sener and Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach* (arXiv:1708.00489).
- Domain-adaptation discrepancy must be tied to the task loss; arbitrary input-space distance is not a general transfer objective. See Mansour, Mohri, and Rostamizadeh, *Domain Adaptation: Learning Bounds and Algorithms* (arXiv:0902.3430).
- In sequential imitation learning, deployed actions alter later observations, so i.i.d. offline behavioral cloning can fail under covariate shift. See Ross, Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 2011).

### Prioritized Next Experiments

1. **Measure the missing variance first:** at K=8 and K=32, run 3 selection seeds x 3 training seeds for random and raw-OOD. Treat independent trainings, not online simulator seeds, as statistical replicates.
2. **Test the selection diagnosis:** compare random, raw-OOD, and OOD plus temporal/state-space diversity under the same K. Enforce a per-time-window quota or use farthest-first selection after OOD candidate filtering.
3. **Test objective mismatch:** on the same selected datasets, compare hard BC only, mixed loss, mixed plus ranking, and longer SimExpert horizons (30/120/300s). A high-K failure limited to mixed/ranking or short horizons would identify the causal component.
4. **Address closed-loop shift:** use at least two DAgger-style collection/relabel/fine-tuning rounds; mix source data or add conservative base-policy KL to control destructive drift.
5. **Redefine OOD in task-relevant space:** condition by intersection/topology/capacity and use policy embeddings, predictive disagreement, or predicted counterfactual value rather than global raw-feature distance.
6. **Harden the oracle control:** replay identical snapshots in separate work directories, compare labels/costs bitwise, and define tolerance-aware deterministic handling for near-tied candidate costs.

### Decision

- Keep **random K=8 SimExpert adaptation** as the current operational Stage-3 baseline.
- Do **not** claim that OOD-guided acquisition is validated end-to-end.
- Treat raw-OOD as a negative but useful ablation: it demonstrates that input-statistic extremeness alone is insufficient for active target-domain adaptation.

---

## Phase 5: Variance, Diversity, and Oracle Hardening (2026-08-13)

This phase executes Prioritized Next Experiments **#6 (oracle hardening)**, **#1 (variance measurement)**, and **#2 (diversity selection)** on the same Jinan→Hangzhou Stage-3 adaptation task (pt32/hd256, K ∈ {8, 32}). Output directory: `records/pipeline_runs/stage3_var_diversity_20260813`.

Naming note: the `ood` method below is the same Mahalanobis selector labeled "raw-OOD" in Phase 4. `ood_diverse` = OOD candidate filtering (top-3K) followed by farthest-first selection; `ood_window_quota` = per-900s-window OOD top-K quota; `random_window_quota` was implemented but not run this phase.

### #6 — Oracle Hardening (result: resolved)

- Added `engine.set_random_seed(0)` to `relabel_simexpert.py` before each replay.
- New `scripts/verify_simexpert_determinism.py` replays identical snapshots N times into separate work dirs and compares labels/costs/margins/feasibility tolerance-aware. Result: `all_bitwise_deterministic = True` (previously 28/1888 cost entries, 20 margins, and 3 feasibility masks differed across replays).
- **Zero-shot baseline is now std=0 across the three rollout seeds** (ATT=335.63, AQL=199.32, AWT=67.75 identical to 2 decimals). The oracle and online rollout are deterministic, so the variance reported below is attributable to selection/training, not to the oracle.

### #1 — Variance Measurement (result: training-seed variance dominates)

`random` ran the full 3×3 matrix (3 selection seeds × 3 training seeds, n=9); deterministic `ood` methods collapse to 3 training seeds (n=3). Between-replicate mean±std (std = training-replicate std, not online-rollout std):

| Method | K | ATT | AQL | AWT | n |
|--------|---|-----|-----|-----|---|
| zero-shot | — | 335.63 | 199.32 | 67.75 | 3 (std=0) |
| random | 8 | 328.06 ± 2.35 | 189.44 ± 4.30 | 63.59 ± 4.61 | 9 |
| ood | 8 | **326.36 ± 3.58** | **186.10 ± 5.85** | 62.18 ± 1.56 | 3 |
| ood_diverse | 8 | 327.24 ± 2.50 | 187.72 ± 5.65 | 64.11 ± 6.85 | 3 |
| ood_window_quota | 8 | 328.83 ± 3.74 | 191.39 ± 7.32 | 63.36 ± 6.57 | 3 |
| random | 32 | 328.89 ± 3.01 | 191.52 ± 5.07 | 70.08 ± 6.51 | 9 |
| ood | 32 | 329.93 ± 1.66 | 193.17 ± 2.84 | 72.49 ± 5.47 | 3 |
| ood_diverse | 32 | 330.50 ± 4.27 | 193.24 ± 8.32 | 71.36 ± 10.97 | 3 |
| ood_window_quota | 32 | 330.60 ± 5.04 | 193.56 ± 8.05 | 70.54 ± 6.47 | 3 |

ANOVA decomposition of `random` (selection-seed MS vs. training-seed MS):

| Metric | K | MS_selection | MS_training |
|--------|---|-------------:|------------:|
| ATT | 8 | 3.23 | **6.27** |
| AQL | 8 | 12.28 | **20.53** |
| AWT | 8 | 17.26 | **22.54** |
| ATT | 32 | 0.57 | **11.85** |
| AQL | 32 | 0.11 | **34.17** |
| AWT | 32 | 4.83 | **54.83** |

Random snapshot selection is itself robust (selection variance is small and nearly vanishes at K=32); the dominant noise source is the **train/val split + torch seed**. To tighten estimates, run more training seeds, not more selection seeds.

### #2 — Diversity Selection (result: no gain over plain OOD; K=8 > K=32)

- At K=8, plain `ood` gives the best ATT (326.36) and AQL (186.10) and the lowest AWT variance (±1.56). `ood_diverse` and `ood_window_quota` do **not** improve the mean and add variance.
- **K=8 beats K=32** on AQL and AWT for every method; K=32 AWT (70.08–72.49) is worse than zero-shot (67.75), reproducing the high-K degradation with a deterministic oracle and multi-seed statistics.
- Diversity helps only relative to the concentrated raw-OOD K=32 failure mode; it does not make OOD dominate random at fixed K.

### Updated Conclusions vs. Phase 4

1. **Oracle determinism is now under control** (Phase 4 diagnostic item #5 resolved). The zero-shot std=0 control makes the between-replicate variance genuinely about selection/training.
2. **The Phase 4 "raw-OOD shows no advantage" finding was under-powered, not reversed.** With 3 training seeds and a deterministic oracle, `ood` K=8 now shows a *modest* mean advantage over `random` on all three metrics (ATT 326.36 vs 328.06; AQL 186.10 vs 189.44; AWT 62.18 vs 63.59). The old single-seed ood AWT of 72.19 was a noisy tail draw (new 3-seed value 62.18 ± 1.56). However, with n=3 the ood 95% CI is still wide (ATT [317.5, 335.3]), so **statistical superiority is still not established** — the defensible claim remains "no large, reliable advantage for OOD over random at K=8."
3. **Diversity-aware selection (experiment #2) is a negative result** at K=8/K=32: temporal/state-space diversity does not convert OOD's concentrated high-demand snapshots into better closed-loop performance, consistent with Phase 4's diagnosis that instantaneous congestion is not a valid proxy for adaptation value.
4. **K=8 remains the operational budget**; the high-K degradation is robust to oracle determinism and multi-seed averaging.

### Updated Decision

- Keep **random K=8** as the operational Stage-3 baseline (OOD K=8's advantage is not yet significant at n=3).
- Run **more training seeds (≥9) for `ood` K=8** before claiming any OOD advantage; selection seeds are not the bottleneck.
- Do **not** pursue diversity-aware selection further as an acquisition improvement at these budgets.
- Re-evaluate the objective-mismatch hypothesis (Prioritized experiment #3) next, since selection-side changes have now been exhausted without closing the gap.

---

## Phase 6: In-Network Architecture Comparison & ρ Diagnostic (2026-08-13)

This phase closes the two remaining Gates from the iteration plan: (a) verify the retrained pt32/hd256 model works in-network, and (b) verify ρ > 0.4 on the in-network test. Both runs use the **Jinan** source checkpoint; Hangzhou pt32/hd256 training was started but terminated early (stage1 reached only epoch 2, stage2 empty — `completion.status=success` is stale and must not be trusted), so Jinan is the only complete pt32 model.

### Setup (apples-to-apples)

- **pt32**: `records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt` (prefix_tokens=32, hidden_dim=256, max_length=64, prompt_mode=queue_wait_short).
- **pt8**: `records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt` (prefix_tokens=8, hidden_dim=64, max_length=64, prompt_mode=queue_wait_short). Same data, same prompt mode, same max_length as pt32 — the only difference is prefix/hidden capacity.
- Online eval: `run_projection_online.py`, dataset=jinan, traffic=`anon_3_4_jinan_real.json`, 3600 steps, seeds 2024/42/123. Outputs: `records/pipeline_runs/jinan_pt32_innetwork_20260813/`, `records/pipeline_runs/jinan_pt8_innetwork_20260813/`.
- ρ diagnostic: `bottleneck_diagnostic.py` on the 3 Jinan processed JSONL files (4296 valid records, max-samples=2000, seed=42, hidden_dim=256/prefix_tokens=32), prompt mode `queue_wait_short` (matches training). Output: `records/pipeline_runs/pt32_bottleneck_20260813/`.

### Result 1 — In-network online test: pt8 vs pt32 (std=0 across seeds)

| Metric | pt8 (8 tok, hd64) | **pt32** (32 tok, hd256) | Δ |
|--------|-------------------:|--------------------------:|----:|
| ATT | 1578.28 | **274.15** | −82.6% |
| AQL | 1729.36 | **166.16** | −90.4% |
| AWT | 1020.13 | **43.78** | −95.7% |
| Sum | 4327.77 | **484.09** | −88.8% |

pt8 is **catastrophically degraded in-network** (AWT ≈ 17 min average wait) — the online manifestation of the Phase 1 "mode collapse to ETWT" diagnosis. pt32 performs as a normal controller. Since the two checkpoints differ only in prefix/hidden capacity (same data, prompt, max_length), **the causal link from capacity to online behavior is established**.

### Result 2 — pt32 ρ diagnostic: still representation_failure (Gate NOT met)

| Metric | pt8-era Phase 1 (Jinan→Jinan) | **pt32 (this run)** |
|--------|-----:|-----:|
| Spearman ρ (mean) | 0.157 | **0.164** |
| Top-action accuracy | 0.396 | 0.395 |
| ECE | 0.447 | 0.517 |
| Diagnosis | representation_failure | **representation_failure** |

ρ barely moved (0.157 → 0.164); the Gate target ρ > 0.4 is **not** met.

### The Contradiction and Its Interpretation

The two results point in opposite directions:

- **Online**: pt32 improves over pt8 by an order of magnitude (AWT 1020 → 44, AQL 1729 → 166). Capacity 8→32 fixed the online mode collapse.
- **Offline ρ**: pt32 is still `representation_failure` (ρ=0.164). Policy logits ranking still does not correlate with SimExpert's 30s-local, per-snapshot min-max-normalized cost ranking.

Interpretation consistent with Phase 4's diagnostics: ρ measures alignment with **SimExpert's short-horizon, locally normalized composite cost**, which was already shown to diverge from 3600s closed-loop ATT/AQL/AWT. A low ρ therefore means the policy's ranking disagrees with that *proxy label*, not necessarily that the policy is broken — the 23× online AWT improvement is direct evidence the strategy is functioning. **ρ=0.164 is best read as label-criterion mismatch, not policy failure.** This strengthens the Phase 4 "objective mismatch" hypothesis and argues against continuing to optimize the SimExpert-ranking objective without first testing longer horizons (30/120/300s).

### Updated Decision

- **Gate (a) is closed**: pt32 works in-network; the capacity fix is real and causal.
- **Gate (b) is explicitly not met and is now deprioritized**: ρ vs the current SimExpert label does not move with capacity. Before investing in more architecture, test whether the *label* is the problem (objective-mismatch ablation, Prioritized experiment #3) — e.g. longer SimExpert horizons or online-supervised targets.
- **Report-level implication**: any paper claim framed as "policy ranking matches SimExpert counterfactual cost" is unsupported; claims framed as "larger prefix capacity eliminates in-network mode collapse" are now supported. Do not conflate the two.

---

## Phase 7: Research Direction — Reliable Cross-Horizon Counterfactual Distillation (proposed)

### Research Question

Why does a small number of SimExpert labels improve cross-network control, while adding more labels degrades closed-loop AWT? The evidence through Phases 4-6 points to a teacher-objective mismatch rather than an OOD-selection, diversity, oracle-determinism, or prefix-capacity problem.

The current SimExpert label for intersection `i` and candidate action `a` is produced by: (1) changing only `i`, (2) freezing every other intersection at its saved phase, (3) rolling out for 30 seconds, and (4) min-max normalizing each metric independently across the four actions of that one snapshot. This is useful as a cheap local diagnostic, but it is not a direct measurement of the 3600-second closed-loop objective.

### Proposed Main Method

**Reliable Cross-Horizon Counterfactual Distillation (CRCD)** turns SimExpert from a short-horizon hard-label source into a reliability-aware counterfactual teacher.

For a collected state, CRCD evaluates the source-policy action and each candidate action at multiple horizons, initially `H = {30, 120, 300}` seconds. It applies a one-step intervention at the target intersection, then lets the target and neighboring intersections resume policy-driven decisions at subsequent decision times. The supervision signal is each action's advantage relative to the source-policy action, not a per-snapshot min-max rank alone.

Only counterfactual evidence that is reliable is distilled. A state is reliable when:

1. The preferred action and the sign of its advantage are stable across horizons.
2. The action remains beneficial when neighboring intersections respond instead of remaining frozen at snapshot phases.
3. The advantage is larger than a predeclared practical margin and satisfies wait/spillback constraints.
4. The target policy is not asked to make a large change on an ambiguous or unstable state.

For reliable states, train with advantage-weighted soft targets or pairwise preferences. For unreliable states, retain the base policy using a nonzero KL anchor. Therefore, increasing the label budget adds only stable counterfactual evidence instead of amplifying short-horizon proxy artifacts.

### What Is Novel, and What Is Not

The intended contribution is **not** merely using a simulator, a longer rollout, DAgger, KL regularization, active selection, or preference/ranking loss. All of these have prior art or already exist in this codebase.

The intended contribution is the combination of:

1. A measurable diagnosis of short-horizon, local counterfactual-label unreliability in closed-loop multi-intersection control.
2. Policy-responsive, multi-horizon counterfactual advantages that test whether a local action remains useful after network response.
3. A reliability gate that decides whether counterfactual supervision should change the policy or preserve the source policy.
4. An empirical explanation and remedy for the observed high-budget adaptation degradation.

Literature boundary:

- COMA (Foerster et al., AAAI 2018) already uses counterfactual baselines for multi-agent credit assignment. CRCD does not claim novelty for counterfactual reasoning itself; it uses an executable traffic simulator to assess label reliability across horizons and policy responses.
- Counterfactual multi-agent actor-critic has already been studied for traffic signal control (Transportation Research Part C, 2024). CRCD must be compared against it conceptually and must not claim to be the first counterfactual TSC method.
- PlanLight (IEEE Access, 2020) already uses rollout demonstrations and iterative policy improvement for TSC. CRCD's distinction is reliability-calibrated supervision, not rollout imitation alone.
- DAgger explains closed-loop covariate shift and SPIBB/CQL motivate conservative policy improvement. Collection/relabeling rounds and the KL anchor are supporting controls, not headline innovations.

### Required Validation Before a Paper Claim

Run the following teacher study before full training:

1. Replay the existing 118 usable snapshots at 30/120/300 seconds. ✅ **done 2026-08-13** — see "Executed Validation #1" below.
2. Compare the current frozen-neighbor evaluator with a policy-responsive evaluator. (pending)
3. Report top-action flip rate, ranking correlation, advantage-sign flip rate, and wait/spillback constraint conflicts across horizons and evaluator modes. ✅ **done 2026-08-13** (flip rate, rank correlation, advantage-sign flip reported below; wait/spillback cross-horizon conflict breakdown still pending).
4. Test whether the reliable subset predicts 3600-second adaptation benefit better than the full label set.

Then evaluate, on identical selected snapshots and independent training seeds:

| Method | Purpose |
|---|---|
| Current 30s frozen SimExpert | Existing teacher baseline |
| 300s frozen SimExpert | Separates horizon from neighbor response |
| 30s policy-responsive SimExpert | Separates response from horizon |
| Multi-horizon without reliability gate | Tests whether consistency alone is enough |
| CRCD | Full proposed method |
| Reliability-count and action-marginal matched control | Verifies that gains are not caused only by fewer labels or class rebalance |

Use `K = {8, 32, 118}` and at least 9 independent training runs per principal arm where feasible. The primary success criterion is not only better K=8 performance: CRCD must prevent the K=32/K=118 AWT degradation while preserving or improving ATT/AQL, p95/max wait, and spillback relative to zero-shot and the current SimExpert teacher.

### Explicitly Deprioritized Directions

- Raw-feature Mahalanobis OOD acquisition: no reliable advantage over random has been established.
- OOD plus temporal/state-space diversity: negative at K=8 and K=32.
- DPO/CPO-style four-action preference optimization: the repository already has soft targets and pairwise cost ranking, so this would be a loss-function rename rather than a new method.
- Prefix-capacity scaling: pt32 fixed mode collapse but did not align policy logits with the current SimExpert ranking.
- A-override correction distillation: it is a useful baseline with positive historical evidence, but targeted correction itself is already an established project result rather than a new Stage-3 contribution.

---

### Executed Validation #1: Multi-Horizon Teacher Validity (2026-08-13)

Implements "Required Validation" items #1 and #3 (horizon replay + flip-rate/rank-correlation/advantage-sign report) on the 118 usable snapshots × 16 intersections (1888 states), Jinan→Hangzhou Stage-3. H ∈ {30, 120, 300}, frozen-neighbor mode, identical scoring (per-snapshot min-max, weights att/aql/awt/spillback = 1/1/1/0.2). The 30s file is the existing teacher (`stage3_ood_20260813/ood/k118/relabeled_states.jsonl`); 120/300 re-relabeled fresh with oracle determinism pinned. Output: `records/pipeline_runs/teacher_validity_20260813/` (summary JSON, per-state CSV, FINDINGS.md).

**Result: the short-horizon teacher is substantially unreliable.**

| Pair | raw-best agreement | expert-label agreement | mean Kendall τ | advantage-sign flip |
|------|-------------------:|-----------------------:|---------------:|--------------------:|
| 30s→120s | 0.713 | 0.712 | 0.686 | 0.219 (n=1603) |
| 30s→300s | 0.612 | 0.611 | 0.567 | 0.316 (n=1603) |
| all-3 stable | 0.602 | 0.601 | — | — |

- **39% of teacher labels change** between 30s and 300s; **32% of advantage-signs flip** (an action the 30s teacher ranks better than base becomes worse than base at 300s).

**The instability is structured, not random.**

- Label distribution collapses toward ETWT as horizon grows: 30s `ETWT 846 / NTST 717 / ELWL 144 / NLSL 181` → 300s `1291 / 577 / 9 / 11`.
- Left-turn labels are near-pure short-horizon artifacts — survival 30s→300s: **ETWT 87.8%, NTST 55.1%, ELWL 4.2%, NLSL 5.5%**. The dominant transition is `→ ETWT` (548 of 734 flips).
- **63% of flips revert to the base action** — the source-policy action was actually fine at 300s; the 30s "improvement" did not survive.
- 30s margin is only weakly protective: flip rate by margin quartile 71.5% → 54.9% → 35.2% → 27.5%, but **top-decile (margin>2.5) still 33.3%**. A reliability gate therefore cannot use the 30s margin alone.

**Not a normalization artifact.** Raw metric spread (max−min across the four actions) *grows*, not shrinks, with horizon: att 2.56→9.92, aql 2.86→10.31, **awt 85.7→3092.8 (36×)**, spillback 0.002→0.005. The 300s ranking rests on *larger* real differences; the 30s window is simply too short to distinguish actions. This is consistent with Phase 3's policy mode collapse: the long horizon genuinely prefers ETWT more often, so non-through 30s labels are partly noise.

**Implication for CRCD:** the multi-horizon reliability gate is the right lever. It should specifically distrust non-through (ELWL/NLSL) 30s labels and any label whose advantage sign is not confirmed at ≥120s. Phase 7's premise ("short-horizon, local counterfactual-label unreliability" — novelty point #1) is now a direct measurement, not an inference from Phase 4's selection-level diagnostics.

**Still open:** teacher-study item #2 (frozen vs policy-responsive neighbor), item #4 (reliable-subset vs 3600s online benefit), and the proxy-vs-real-metric gap (counterfactual `awt` is accumulated vehicle-seconds, not true waiting time) — the latter requires the LLM-in-the-loop online gold standard.

---

## Phase 7: CRCD Teacher Study — Result (2026-08-14)

### What Was Run

CRCD (Reliable Cross-Horizon Counterfactual Distillation) was implemented (`projection_train/crcd_relabel.py`, `scripts/run_phase7_crcd.py`) and evaluated as a teacher study on the Jinan→Hangzhou Stage-3 task (pt32/hd256), over the identical 118-snapshot pool (1888 states), frozen-neighbor SimExpert at H={30,120,300}, per-snapshot min-max composite scoring (att/aql/awt/spillback = 1/1/1/0.2).

Four teacher variants × K ∈ {8, 32, 118} × 3 training seeds (2024/42/123), each evaluated online on 3 simulator seeds (3600s, Hangzhou `anon_4_4_hangzhou_real_5816.json`):

- `c30` — 30s frozen labels (existing teacher baseline)
- `c300` — 300s frozen labels (longer horizon, no gate)
- `crcd_nogate` — multi-horizon majority vote, no gate, no KL
- `crcd` — reliability gate + KL-to-base anchor (weight 0.5, `expert_match`)

Reliability gate: `best_300` is a through action, `best_120 == best_300`, `adv_300 > 0.05`, `adv_120 > 0`, and Pareto-feasible. The gate marked 1049/1888 states reliable (55.6%).

### Online Results (mean over 3 training seeds; AWT is the primary success metric)

| Method | K | ATT | AQL | AWT |
|---|---|---|---|---|
| zero-shot | — | 335.63 | 199.32 | 67.75 |
| c30 | 8 | 327.66±3.19 | 188.58±5.09 | **61.81±4.49** |
| c30 | 32 | 332.90±1.75 | 198.50±4.69 | 80.39±8.11 |
| c30 | 118 | 335.37±1.72 | 201.95±3.64 | **83.24±14.13** |
| c300 | 8 | 330.64±3.36 | 194.82±5.96 | 78.68±12.33 |
| c300 | 32 | 337.03±2.71 | 205.95±4.04 | 94.38±7.16 |
| c300 | 118 | 366.93±5.57 | 267.05±12.51 | 236.00±18.52 |
| crcd_nogate | 8 | 326.86±1.34 | 188.15±4.23 | 68.33±3.29 |
| crcd_nogate | 32 | 334.36±3.25 | 200.14±7.94 | 83.57±11.68 |
| crcd_nogate | 118 | 350.73±8.13 | 233.71±17.07 | 165.48±44.49 |
| crcd | 8 | 328.97±2.00 | 192.68±3.27 | 68.05±3.35 |
| crcd | 32 | 334.91±1.04 | 203.10±2.11 | 97.01±10.08 |
| crcd | 118 | 349.89±10.46 | 233.21±20.15 | 155.85±34.37 |

### Verdict: CRCD Fails the Success Criterion (Negative Result)

The primary criterion (Phase 7 "Required Validation") was that CRCD must **prevent** K=32/K=118 AWT degradation while preserving or improving ATT/AQL relative to zero-shot and the c30 teacher. It does neither:

1. **CRCD does not prevent high-K degradation.** crcd K=118 AWT = 155.85, vs c30's 83.24 and zero-shot's 67.75. crcd K=32 (97.01) is worse than crcd_nogate K=32 (83.57) — adding the gate *hurt* at K=32.
2. **CRCD does not preserve K=8 gains.** crcd K=8 AWT = 68.05, worse than c30 K=8 (61.81) and not better than zero-shot (67.75).
3. **The 30s teacher is the best at every budget.** c30 AWT (61.81 / 80.39 / 83.24) dominates c300, crcd_nogate, and crcd at K=8, K=32, and K=118. The "short-horizon unreliability" CRCD was designed to fix is empirically the *least* harmful label source.

### Root Cause: Long-Horizon Frozen-Neighbor Labels Collapse to Through (ETWT), Destroying Action Diversity

Action distribution across horizons and the gate (1888 states):

| Source | ETWT | NTST | ELWL | NLSL |
|---|---|---|---|---|
| base_action (source policy) | 895 | 501 | 348 | 144 |
| best_30 | 847 | 715 | 145 | 181 |
| best_120 | 1173 | 638 | 38 | 39 |
| best_300 | 1290 | 577 | 10 | 11 |
| crcd reliable subset (gate) | 637 | 412 | **0** | **0** |

- The 30s labels preserve left-turn actions (ELWL+NLSL = 326, 17%); the 300s labels collapse to through (non-through = 21, 1.1%); the CRCD gate's `require_through` criterion makes its reliable subset **100% through**.
- This is the same ETWT mode collapse Phase 3 identified in-network, now induced out-of-network by long-horizon frozen-neighbor counterfactuals. All-through control removes the inter-intersection coordination that keeps queues bounded; closed-loop AWT explodes (c300 K=118 = 236).
- The frozen-neighbor assumption diverges from closed-loop reality as the horizon grows, systematically favoring the through action. The 30s label's instability is partly *protective* diversity, not the harm it was assumed to be.

### Revised Interpretation

The Phase 7 premise — "short-horizon, local counterfactual-label unreliability is the leading cause of high-K degradation" (novelty point #1) — is **rejected** by the online evidence. Executed Validation #1 correctly measured that 39% of labels flip 30s→300s and 32% of advantage-signs flip, but that instability does **not** translate into closed-loop harm: the 30s labels are the best available teacher precisely because their diversity resists the ETWT collapse that long-horizon labels and the through-requiring gate both amplify.

This closes the label-side search. OOD acquisition (Phase 4/5), diversity selection (Phase 5), and now horizon gating + reliability distillation (Phase 7) are all negative. The high-K degradation is a property of the **training objective** (per-snapshot min-max counterfactual cost) and the **frozen-neighbor evaluator**, not of label selection or horizon.

### Follow-up: Loss Ablation — hard-BC-only (Negative)

To test the loss lever directly (Prioritized #3), `c30` was retrained with `target_mode=hard` + `ranking_loss_weight=0` (hard CE only, no soft targets, no ranking) at K ∈ {8, 32, 118} × 3 training seeds, on the identical selected datasets (`records/pipeline_runs/phase7_loss_ablation_20260814`).

| c30 loss | K=8 AWT | K=32 AWT | K=118 AWT |
|---|---|---|---|
| zero-shot | 67.75 | — | — |
| mixed + ranking (original) | **61.81±4.49** | 80.39±8.11 | **83.24±14.13** |
| hard-BC only | 67.24±9.28 | **70.30±5.11** | 94.74±12.98 |

1. **Hard-BC does not remove the high-K degradation.** AWT still rises monotonically 67.24 → 70.30 → 94.74 from K=8 → K=32 → K=118; at K=118 hard-BC (94.74) is no better than mixed (83.24).
2. **The soft/ranking loss is not the culprit.** It is mildly helpful at K=8 (61.81 vs 67.24) and statistically indistinguishable at K=118. The mixed loss is a regularization on top of a label that is already misaligned, not the source of the misalignment.

**Conclusion:** the high-K degradation is a property of the **hard counterfactual label** (the SimExpert argmin action under frozen neighbors), not the soft/ranking loss, the selection method, the horizon, or the reliability gate. The only untested lever with a mechanistic claim to fix this is the **evaluator** — replacing the frozen-neighbor rollout with a policy-responsive rollout (Required Validation #2), because the frozen-neighbor assumption is what systematically biases long-horizon counterfactuals toward through actions and collapses action diversity.

### Follow-up: Policy-Responsive Counterfactual Evaluator (Mixed Result, K=32 Positive)

The policy-responsive evaluator was implemented to replace frozen-neighbor rollouts. At the first decision point, the source Stage-2 policy chooses actions for all intersections; only the target intersection is overridden by each counterfactual candidate. At subsequent `MIN_ACTION_TIME=30s` decision points, non-target intersections are re-controlled by the source policy instead of being frozen in their snapshot phases.

Implementation and outputs:

- Core evaluator: `projection_train/policy_responsive_simexpert.py`
- CLI: `scripts/relabel_policy_responsive.py`
- Phase-7 wrapper: `scripts/run_phase7_policy_responsive.py`
- Output: `records/pipeline_runs/phase7_policy_responsive_20260814_full`

Coverage was complete for both responsive horizons:

| Label file | Horizon | States | Snapshots | Candidate rollouts | Skipped |
|---|---:|---:|---:|---:|---:|
| responsive H30 | 30s | 1888 | 118/118 | 7552 | 0 |
| responsive H120 | 120s | 1888 | 118/118 | 7552 | 0 |

Teacher-validity diagnostics show that responsive H30 is a mild correction to the frozen 30s teacher, while responsive H120 is a much stronger intervention:

| Pair | Raw-best agreement | Expert-label agreement | Kendall tau | Advantage sign flips |
|---|---:|---:|---:|---:|
| frozen 30 -> responsive 30 | 0.919 | 0.941 | 0.917 | 0.0069 |
| frozen 30 -> responsive 120 | 0.599 | 0.628 | 0.601 | 0.1485 |

Unlike frozen long-horizon labels, responsive H120 does not collapse to through-only actions:

| Source | ETWT | NTST | ELWL | NLSL |
|---|---:|---:|---:|---:|
| frozen 30 expert labels | 846 | 717 | 144 | 181 |
| responsive 30 expert labels | 824 | 731 | 155 | 178 |
| responsive 120 expert labels | 632 | 617 | 288 | 351 |

Online H30 responsive training used K in `{8,32,118}` x 3 training seeds, each evaluated on 3 simulator seeds. Results:

| Method | K | ATT | AQL | AWT |
|---|---:|---:|---:|---:|
| zero-shot | -- | 335.63 | 199.32 | 67.75 |
| frozen c30 | 8 | 327.66±3.19 | 188.58±5.09 | **61.81±4.49** |
| responsive H30 | 8 | 332.60±5.51 | 196.72±11.79 | 70.85±11.85 |
| frozen c30 | 32 | 332.90±1.75 | 198.50±4.69 | 80.39±8.11 |
| responsive H30 | 32 | **326.12±1.48** | **185.43±1.96** | **60.38±1.57** |
| frozen c30 | 118 | 335.37±1.72 | 201.95±3.64 | 83.24±14.13 |
| responsive H30 | 118 | 331.51±0.75 | 194.37±1.10 | 76.10±3.29 |

Key interpretation:

1. **Responsive H30 fixes the K=32 failure.** AWT improves from frozen c30 K=32's 80.39 to 60.38, beating zero-shot by 7.37 and improving ATT/AQL at the same time.
2. **K=8 becomes unstable.** One training seed is strong (AWT 54.14), two are poor (78.05/80.35). This is a warning that responsive labels are not uniformly better under very small budgets.
3. **K=118 still degrades.** Responsive H30 reduces the frozen c30 K=118 damage (AWT 83.24 -> 76.10), but remains worse than zero-shot (67.75). The high-K problem is therefore only partly evaluator-side; over-supervision or snapshot-label weighting remains unresolved.
4. **H120 remains an open online question.** Its label distribution is much more different from frozen 30s, but it does not show the frozen long-horizon through-collapse. This justifies a focused K=32 online test before any wider H120 sweep.

Integrity note: during the first H30 run, the original eval scheduler launched more evaluation seeds than available GPUs, causing one same-GPU OOM. The scheduler was patched in `scripts/run_phase7_crcd.py` to evaluate in GPU-sized batches. Final status is `ok` for all nine H30 configurations and all final `eval_summary.json` files are present. One K=8/tr42 training log contains the interrupted pre-patch run; it later evaluated from an already written `best.pt`, so K=8 should be treated as less clean than the K=32/K=118 results.

### Supplementary Experiments Completed (2026-08-14)

Three focused supplements completed under `records/pipeline_runs/phase7_policy_responsive_supplement_20260814`:

| Experiment | Labels | K | Training seeds | GPU | Purpose |
|---|---|---:|---|---:|---|
| `h30_k32_more_a` | responsive H30 | 32 | 7, 11, 99 | 0 | Extend the positive H30 K=32 result toward n=9 |
| `h30_k32_more_b` | responsive H30 | 32 | 314, 2718, 9001 | 1 | Extend the positive H30 K=32 result toward n=9 |
| `h120_k32` | responsive H120 | 32 | 2024, 42, 123 | 2 | First online test of the stronger responsive horizon |

All nine supplementary training configurations finished with `status.json = ok`; no traceback/OOM/error string was found in the supplement logs, and all GPUs were idle after completion.

Raw completed supplementary means:

| Experiment | n training seeds | ATT | AQL | AWT | Δ AWT vs zero-shot |
|---|---:|---:|---:|---:|---:|
| responsive H30 K=32, main run | 3 | 326.12±1.48 | 185.43±1.96 | **60.38±1.57** | -7.37 |
| responsive H30 K=32, supplement A | 3 | 328.02±2.22 | 189.29±3.61 | 68.81±5.61 | +1.06 |
| responsive H30 K=32, supplement B | 3 | 328.20±1.58 | 190.48±3.10 | 69.48±7.02 | +1.73 |
| responsive H30 K=32, combined | 9 | 327.45±2.02 | 188.40±3.67 | 66.23±6.70 | -1.52 |
| responsive H120 K=32 | 3 | 326.57±4.82 | 185.16±9.64 | **56.25±6.79** | -11.50 |

Per-training-seed outputs:

| Arm | Training seed | ATT | AQL | AWT |
|---|---:|---:|---:|---:|
| H30 K=32 | 2024 | 326.52 | 184.77 | 58.63 |
| H30 K=32 | 42 | 324.15 | 183.42 | 62.45 |
| H30 K=32 | 123 | 327.70 | 188.09 | 60.07 |
| H30 K=32 | 7 | 329.69 | 192.71 | 74.45 |
| H30 K=32 | 11 | 324.88 | 184.30 | 61.17 |
| H30 K=32 | 99 | 329.50 | 190.85 | 70.82 |
| H30 K=32 | 314 | 330.37 | 194.84 | 79.40 |
| H30 K=32 | 2718 | 327.53 | 187.93 | 64.79 |
| H30 K=32 | 9001 | 326.69 | 188.67 | 64.25 |
| H120 K=32 | 2024 | 333.19 | 197.97 | 64.33 |
| H120 K=32 | 42 | 321.87 | 174.73 | 47.70 |
| H120 K=32 | 123 | 324.65 | 182.78 | 56.71 |

Success gates:

- H30 K=32 should remain below zero-shot AWT=67.75 after combining all 9 training seeds. **Passed narrowly**: combined AWT=66.23, only 1.52 below zero-shot; ATT/AQL remain clearly better.
- H120 K=32 is useful only if it beats responsive H30 K=32 on AWT without degrading ATT/AQL. **Passed in mean at n=3**: AWT=56.25 vs H30 n=9 AWT=66.23, with similar ATT and better AQL. However, H120 variance is high (`AWT std=6.79`, `AQL std=9.64`) and needs a 9-seed confirmation before it can supersede H30 as the operational baseline.

Interpretation:

1. **The initial H30 K=32 result was real but overestimated.** After increasing from 3 to 9 training seeds, H30 K=32 still improves zero-shot on all three metrics, but AWT gain shrinks from -7.37 to -1.52. The two added seed batches contain several AWT-worse-than-zero-shot runs (74.45, 70.82, 79.40), so H30 K=32 should be described as a modest, variance-sensitive improvement rather than a strong fix.
2. **Responsive H120 is now the most promising evaluator horizon.** It delivers the best observed mean AWT (56.25) and also strong ATT/AQL, unlike the frozen long-horizon teachers that collapsed to through-only actions. The policy-responsive rollout appears to remove the major long-horizon failure mode.
3. **Training-seed variance remains the bottleneck.** The same label file and K can produce AWT from 58.63 to 79.40 for H30 and 47.70 to 64.33 for H120. Any final claim should therefore be based on at least 9 training seeds per arm and should report training-seed variance, not simulator-seed variance.
4. **The high-K problem is not fully closed.** These supplements only test K=32. H30 K=118 remained worse than zero-shot; whether responsive H120 or replay/data weighting fixes K=118 is still unknown.

<!-- PHASE7_SUPPLEMENT_MONITOR_START -->

### Supplement Monitor (auto-updated)

- Last poll: `2026-08-14T18:13:46`
- Poll interval: 40 minutes
- Result directory: `/home/code/LLMSTSC/records/pipeline_runs/phase7_policy_responsive_supplement_20260814`

| Experiment | Labels | GPU | Status | Result JSONs | Completed training seeds | Current mean±std | Running PIDs | Notes |
|---|---|---:|---|---:|---|---|---|---|
| h30_k32_more_a | responsive H30 | 0 | done | 9/9 | 11, 7, 99 | ATT=328.02±2.22, AQL=189.29±3.61, AWT=68.81±5.61 | — | aggregate present |
| h30_k32_more_b | responsive H30 | 1 | done | 9/9 | 2718, 314, 9001 | ATT=328.20±1.58, AQL=190.48±3.10, AWT=69.48±7.02 | — | aggregate present |
| h120_k32 | responsive H120 | 2 | done | 9/9 | 123, 2024, 42 | ATT=326.57±4.82, AQL=185.16±9.64, AWT=56.25±6.79 | — | aggregate present |

GPU snapshot:

```
0, 2 MiB, 32607 MiB, 0 %
1, 2 MiB, 32607 MiB, 0 %
2, 2 MiB, 32607 MiB, 0 %
```

<!-- PHASE7_SUPPLEMENT_MONITOR_END -->

### H120 Follow-up Experiments Completed (2026-08-14)

The H120 follow-up completed under `records/pipeline_runs/phase7_policy_responsive_h120_followup_20260814`.

| Experiment | Labels | K | Training seeds | Status | Purpose |
|---|---|---:|---|---|---|
| `h120_k32_more_a` | responsive H120 | 32 | 7, 11, 99 | ok | Extend H120 K=32 from n=3 to n=6 |
| `h120_k32_more_b` | responsive H120 | 32 | 314, 2718, 9001 | ok | Extend H120 K=32 from n=6 to n=9 |
| `h120_k118` | responsive H120 | 118 | 2024, 42, 123 | ok | First high-K test: does responsive H120 fix K=118 degradation? |

Launch/monitor integrity:

- Launch check at `2026-08-14T18:21:02+0800`: all three formal runners were alive, entered training, and consumed the intended GPUs.
- Two short foreground probe processes were intentionally terminated before the formal launch because they would not have produced complete eval/aggregate files.
- 40-minute monitor script: `scripts/monitor_phase7_h120_followup.py`; first monitor notification was sent successfully at `2026-08-14T18:26:50+08:00`.
- Completion check: all three follow-up `status.json` files contain only `ok`; all GPUs were idle after completion; no traceback/OOM/error string was found in the follow-up logs.
- Final Feishu completion notification was sent manually after parsing the completed results; the 40-minute monitor process was then stopped to avoid duplicate completion notifications.

Aggregate results:

| Arm | n training seeds | ATT | AQL | AWT | Δ AWT vs zero-shot | Δ AWT vs H30 K=32 n=9 | Δ AWT vs H30 K=118 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | — | 335.63 | 199.32 | 67.75 | — | — | — |
| responsive H30 K=32 | 9 | 327.45±2.02 | 188.40±3.67 | 66.23±6.70 | -1.52 | — | -9.87 |
| responsive H30 K=118 | 3 | 331.51±0.75 | 194.37±1.10 | 76.10±3.29 | +8.35 | +9.87 | — |
| responsive H120 K=32, prior | 3 | 326.57±4.82 | 185.16±9.64 | 56.25±6.79 | -11.50 | -9.98 | -19.85 |
| responsive H120 K=32, more A | 3 | 324.64±0.98 | 182.52±2.31 | 60.16±2.51 | -7.59 | -6.07 | -15.94 |
| responsive H120 K=32, more B | 3 | 326.46±4.40 | 185.24±6.42 | 57.38±1.41 | -10.37 | -8.85 | -18.72 |
| responsive H120 K=32, combined | 9 | 325.89±3.91 | 184.31±6.93 | 57.93±4.57 | -9.82 | -8.30 | -18.17 |
| responsive H120 K=118 | 3 | **325.15±2.65** | **180.13±2.91** | **49.31±3.89** | **-18.44** | **-16.92** | **-26.79** |

Per-training-seed outputs:

| Arm | Training seed | ATT | AQL | AWT |
|---|---:|---:|---:|---:|
| H120 K=32 | 42 | 321.87 | 174.73 | 47.70 |
| H120 K=32 | 123 | 324.65 | 182.78 | 56.71 |
| H120 K=32 | 2024 | 333.19 | 197.97 | 64.33 |
| H120 K=32 | 7 | 324.68 | 182.99 | 61.58 |
| H120 K=32 | 11 | 325.81 | 185.07 | 62.27 |
| H120 K=32 | 99 | 323.42 | 179.48 | 56.63 |
| H120 K=32 | 314 | 328.52 | 189.38 | 57.94 |
| H120 K=32 | 2718 | 320.35 | 176.18 | 55.45 |
| H120 K=32 | 9001 | 330.52 | 190.18 | 58.76 |
| H120 K=118 | 42 | 327.24 | 181.03 | 46.79 |
| H120 K=118 | 123 | 326.79 | 183.16 | 54.81 |
| H120 K=118 | 2024 | 321.41 | 176.21 | 46.34 |

Decision criteria outcomes:

- **H120 K=32 promotion gate passed.** Combined 9-seed H120 K=32 has AWT=57.93, clearly below H30 K=32's 66.23, with better ATT (325.89 vs 327.45) and AQL (184.31 vs 188.40). H120 K=32 therefore supersedes H30 K=32 as the safer n=9 baseline.
- **H120 K=118 high-K gate passed at n=3.** It improves over zero-shot AWT=67.75 by 18.44 and does not reproduce H30 K=118 degradation (49.31 vs 76.10). ATT and AQL also improve over both zero-shot and H30 K=118.

### Final Decision (Phase 7, updated after H120 follow-up completion)

- CRCD (reliability-gated multi-horizon distillation) and hard-BC-only remain **negative**.
- The policy-responsive evaluator is now a **strong positive result**, not merely qualified:
  - H120 K=32 is validated at n=9 and is materially better than H30 K=32 on ATT/AQL/AWT.
  - H120 K=118 is the first arm that reverses the previously robust high-K degradation, with the best observed AWT so far.
- The current operational baseline should be **responsive H120 K=32** because it has n=9 evidence and strong margins over zero-shot and H30 K=32.
- The current best overall arm is **responsive H120 K=118**, but it has only n=3 training seeds. Before making it the default high-budget setting, extend it to n=9 with six additional training seeds.
- Mechanistic implication: the high-K degradation was not caused by label budget itself. It was caused by the **frozen-neighbor evaluator**. Once long-horizon labels are generated with a policy-responsive rollout, more labels can help rather than hurt.

### Best Arm Metadata

The best-performing configuration by AWT is **responsive H120 K=118**:

| Field | Value |
|---|---|
| Result directory | `records/pipeline_runs/phase7_policy_responsive_h120_followup_20260814/h120_k118` |
| Target/evaluation dataset | **Hangzhou**, `anon_4_4_hangzhou_real_5816.json` |
| Source policy | Jinan Stage-2 pt32/hd256 checkpoint, `records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt` |
| Label file | `records/pipeline_runs/phase7_policy_responsive_20260814_full/relabeled_policy_responsive_h120.jsonl` |
| Label/evaluator type | policy-responsive counterfactual evaluator, horizon = 120s |
| Label budget | K=118 snapshots |
| Training seeds | 2024, 42, 123, 7, 11, 99, 314, 2718, 9001 (n=9) |
| Eval seeds | 2024, 42, 123 |
| Loss | mixed: hard CE + soft target (`alpha=0.3`) + ranking loss (`weight=0.05`) |
| Online result | ATT=326.64±3.44, AQL=182.24±4.77, AWT=48.88±2.76 (n=9; n=3 was ATT=325.15±2.65, AQL=180.13±2.91, AWT=49.31±3.89) |

Architecture and parameter count:

| Component | Setting / Count |
|---|---:|
| Backbone | `/home/data/model/qwen-vl-8B` (frozen; loaded externally, not saved in checkpoint) |
| Sequence length | 16 |
| Input dimension | 8 |
| Image size | 224 |
| Hidden dimension | 256 |
| Prefix tokens | 32 |
| Prefix mode | prefix |
| Max text length | 64 |
| Trainable time-series projector parameters | 301,099 |
| Trainable vision-prefix projector parameters | 151,126,016 |
| **Total trainable parameters saved/fine-tuned** | **151,427,115 (~151.4M)** |

Interpretation: if "model size" means the frozen LLM backbone, the experiment uses the Qwen-VL **8B** model. If it means the actually fine-tuned parameters in this adaptation stage, the best arm fine-tunes **151.4M** projector parameters while keeping the 8B backbone frozen.

---

### H120 K=118 n=9 Confirmation (2026-08-14)

The best arm was extended from n=3 to n=9 training seeds (added 7, 11, 99, 314, 2718, 9001), reusing the same `relabeled_policy_responsive_h120.jsonl` labels, source checkpoint, and mixed loss. Two parallel runs on GPU 0/1 (`h120_k118_more_a`, `h120_k118_more_b`); all 6 new configs finished `ok`, no traceback/OOM.

| Metric | H120 K=118 n=9 | zero-shot | H120 K=32 n=9 | Δ vs zero-shot | Δ vs K=32 |
|---|---:|---:|---:|---:|---:|
| ATT | 326.64±3.44 | 335.63 | 325.89 | -8.99 | +0.75 |
| AQL | **182.24±4.77** | 199.32 | 184.31 | -17.08 | -2.07 |
| AWT | **48.88±2.76** | 67.75 | 57.93 | -18.87 | -9.05 |

Per-training-seed AWT (n=9, order 2024/42/123/7/11/99/314/2718/9001): 46.34, 46.79, 54.81, 48.66, 52.62, 47.43, 46.80, 49.02, 47.45.

**Promotion gate passed.** At n=9, responsive H120 K=118 remains the best arm: AWT is 18.87 below zero-shot and 9.05 below H120 K=32 n=9; AQL also improves; ATT is essentially tied with H120 K=32 (326.64 vs 325.89, within noise). The high-K degradation is fully reversed (frozen c30 K=118 AWT=83.24, c300 K=118=236 → responsive H120 K=118=48.88). Responsive H120 K=118 is now the confirmed default high-budget configuration.

Artifact: `records/pipeline_runs/phase7_policy_responsive_h120_followup_20260814/h120_k118_n9_aggregate.json`.
