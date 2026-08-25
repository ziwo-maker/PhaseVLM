# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-08-09 00:23 | /auto-review-loop | review-stage/AUTO_REVIEW.md | review | Round 1 adversarial artifact review of PhaseVLM and Stage 3 readiness |
| 2026-08-09 00:23 | /auto-review-loop | review-stage/REVIEW_STATE_20260809_002307.json | review | Timestamped Round 1 review state |
| 2026-08-09 00:23 | /auto-review-loop | review-stage/REVIEW_STATE.json | review | Latest Round 1 review state |
| 2026-08-09 00:33 | /auto-review-loop | review-stage/AUTO_REVIEW.md | review | Round 2 adversarial audit of cost ranking, reward shaping, and Stage 3 test protocol |
| 2026-08-09 00:33 | /auto-review-loop | review-stage/REVIEW_STATE_20260809_003331.json | review | Timestamped Round 2 review state |
| 2026-08-09 00:33 | /auto-review-loop | review-stage/REVIEW_STATE.json | review | Latest Round 2 review state |
| 2026-08-11 23:46 | /idea-creator | idea-stage/IDEA_REPORT_20260811_234648.md | idea-discovery | 11 ranked PhaseVLM Stage 3 ideas with falsifiable minimum experiments |
| 2026-08-11 23:46 | /idea-creator | idea-stage/IDEA_REPORT.md | idea-discovery | Latest PhaseVLM Stage 3 idea report |

## Auto-Review-Loop (2026-08-09 ~ 2026-08-10)

- `review-stage/AUTO_REVIEW.md` — 4-round review log: Stage 3 redesign from PPO to cross-network adaptation
- `review-stage/REVIEW_STATE.json` — Final state: round=4, score=5/10, status=completed
- `scripts/run_stage3_cross_network_adaptation.py` — Stage 3 pipeline (NEW)
- `scripts/run_stage3_cross_network_adaptation.sh` — Deployable launcher (NEW)
- `scripts/collect_online_states.py` — State collection in CityFlowEnv (NEW, deprecated by evaluator hook)
- `projection_train/run_projection_online.py` — Added `--collect_states_output` hook and `sequences` return
- `projection_train/train_projection_bc.py` — Added `--init_policy_ckpt` for source checkpoint init

## Experiment Bridge — 2026-08-12

### New Scripts
- `projection_train/sim_expert_validator.py` — SimExpert Oracle Validation (Step 0)
- `projection_train/bottleneck_diagnostic.py` — Bottleneck Diagnostic (Idea 3)
- `projection_train/regret_acquisition.py` — Acquisition functions (Idea 1)
- `projection_train/active_query_runner.py` — Active Query Experiment Runner (Idea 1)

### Results
- `sim_expert_validation/sim_expert_validation.json` — Oracle validation passed
- `bottleneck_diagnostic/bottleneck_diagnostic.json` — Representation failure (ρ=0.19)
- `active_query_results/active_query_results.json` — OOD best acquisition
- `refine-logs/EXPERIMENT_RESULTS.md` — Comprehensive results report
| 2026-08-13 12:00 | /experiment-bridge | refine-logs/STAGE3_OOD_PLAN.md | plan | Stage 3 OOD-guided adaptation plan (pt32/hd256 arch, OOD vs random, K=8/32/120) |
| 2026-08-13 12:00 | /experiment-bridge | projection_train/ood_selection.py | code | OOD snapshot selection: strict features, coverage-enforced source stats, Mahalanobis scoring |
| 2026-08-13 12:00 | /experiment-bridge | scripts/run_stage3_ood_adaptation.py | code | Stage 3 OOD pipeline: zero-shot → collect → select → SimExpert relabel → verify → snapshot-split → train → multi-seed eval |
| 2026-08-13 12:00 | /experiment-bridge | tests/test_ood_selection.py | test | 13 unit tests for OOD selection (all pass) |
| 2026-08-13 12:00 | /experiment-bridge | tests/test_relabel_simexpert.py | test | 4 unit tests for yellow-transition emulation (all pass) |
| 2026-08-13 12:00 | /experiment-bridge | scripts/relabel_simexpert.py | code | Fixed: yellow-transition emulation, Jinan grid (4,3), proxy-metric docs |
| 2026-08-13 12:00 | /experiment-bridge | projection_train/train_projection_bc.py | code | Fixed: infeasible-prediction penalty + invalid_action_rate in cost metrics |
| 2026-08-13 15:10 | /experiment-bridge | records/pipeline_runs/stage3_ood_20260813/ | experiment | Stage 3 OOD adaptation: 7/7 arms done (zero-shot, random×3, ood×3) |
| 2026-08-13 15:10 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS.md | results | Negative result: OOD ≤ random at all K; both degrade at K=118; best = random K=8 |
| 2026-08-13 15:10 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | tracker | Run-by-run table R001-R007 with status and notes |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/round-0-initial-proposal.md | implementation | Anchored initial PRCT method proposal |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/round-1-review.md | implementation | External review attempts recorded; service returned 503 twice |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/FINAL_PROPOSAL_20260818_142414.md | implementation | Timestamped PRCT final proposal |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/FINAL_PROPOSAL.md | implementation | Latest PRCT final proposal |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/EXPERIMENT_PLAN_20260818_142414.md | implementation | Timestamped claim-driven PRCT experiment plan |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/EXPERIMENT_PLAN.md | implementation | Latest PRCT experiment plan |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/EXPERIMENT_TRACKER_20260818_142414.md | implementation | Timestamped PRCT execution tracker |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest PRCT execution tracker |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/PIPELINE_SUMMARY_20260818_142414.md | implementation | Timestamped integrated PRCT summary |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/PIPELINE_SUMMARY.md | implementation | Latest integrated PRCT summary |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/REVIEW_SUMMARY_20260818_142414.md | implementation | Timestamped review summary and reviewer-service gap |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/REVIEW_SUMMARY.md | implementation | Latest review summary and reviewer-service gap |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/REFINEMENT_REPORT_20260818_142414.md | implementation | Timestamped refinement report |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/REFINEMENT_REPORT.md | implementation | Latest refinement report |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/REFINE_STATE_20260818_142414.json | implementation | Timestamped completed refinement state |
| 2026-08-18 14:24 | /research-refine-pipeline | refine-logs/REFINE_STATE.json | implementation | Latest completed refinement state |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/round-0-stage2-prct-proposal.md | implementation | Full anchored Stage-2 PRCT refactor proposal |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/round-1-stage2-prct-review.md | implementation | External method-review request record; reviewer service returned 503 |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/FINAL_PROPOSAL_20260818_205337.md | implementation | Timestamped final Stage-2 PRCT proposal |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/FINAL_PROPOSAL.md | implementation | Latest final Stage-2 PRCT proposal |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/REVIEW_SUMMARY_20260818_205337.md | implementation | Timestamped review summary and evidence gate |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/REVIEW_SUMMARY.md | implementation | Latest review summary and evidence gate |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/REFINEMENT_REPORT_20260818_205337.md | implementation | Timestamped Stage-2 refactor report |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/REFINEMENT_REPORT.md | implementation | Latest Stage-2 refactor report |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/EXPERIMENT_PLAN_20260818_205337.md | implementation | Timestamped 24-GPU-hour gated PRCT experiment plan |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/EXPERIMENT_PLAN.md | implementation | Latest gated PRCT experiment plan |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/EXPERIMENT_TRACKER_20260818_205337.md | implementation | Timestamped PRCT run tracker |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest PRCT run tracker |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/PIPELINE_SUMMARY_20260818_205337.md | implementation | Timestamped integrated Stage-2 PRCT summary |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/PIPELINE_SUMMARY.md | implementation | Latest integrated Stage-2 PRCT summary |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/score-history_20260818_205337.md | implementation | Timestamped score history with unavailable external review |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/score-history.md | implementation | Latest score history with unavailable external review |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/REFINE_STATE_20260818_205337.json | implementation | Timestamped completed Stage-2 PRCT refinement state |
| 2026-08-18 20:53 | /research-refine-pipeline | refine-logs/REFINE_STATE.json | implementation | Latest completed Stage-2 PRCT refinement state |
| 2026-08-19 13:54 | /research-refine-pipeline | refine-logs/JSCD_NEXT_STAGE_PLAN_20260819_135445.md | implementation | Timestamped JSCD next-stage plan based on corrected PRCT evidence; planning only, no experiment launched |
| 2026-08-19 13:54 | /research-refine-pipeline | refine-logs/JSCD_NEXT_STAGE_PLAN.md | implementation | Latest JSCD next-stage plan |
| 2026-08-20 08:44 | /analyze-results | refine-logs/JSCD_P2_RESULT_ANALYSIS_20260820_084458.md | implementation | Timestamped analysis of the JSCD P2 pilot; P1 passed, P2 no-go, P3 prohibited |
| 2026-08-20 08:44 | /analyze-results | refine-logs/JSCD_P2_RESULT_ANALYSIS.md | implementation | Latest JSCD P2 result analysis |
| 2026-08-20 09:49 | /research-refine-pipeline | refine-logs/JSCD_MIXED_5816_SCALE_EXPERIMENT_20260820_094949.md | implementation | Timestamped design for balanced Hangzhou_5816 supplement and JSCD data-scale experiment |
| 2026-08-20 09:49 | /research-refine-pipeline | refine-logs/JSCD_MIXED_5816_SCALE_EXPERIMENT.md | implementation | Latest JSCD mixed-5816 data-scale experiment design |
| 2026-08-20 12:30 | /analyze-results | refine-logs/JSCD_MIXED_5816_SCALE_RESULT_ANALYSIS_20260820_123002.md | implementation | Timestamped analysis of 9/9 completed JSCD mixed-5816 data-scale runs; rule-teacher mixing worsened AWT and did not restore JSCD efficacy |
| 2026-08-20 12:30 | /analyze-results | refine-logs/JSCD_MIXED_5816_SCALE_RESULT_ANALYSIS.md | implementation | Latest analysis of completed JSCD mixed-5816 data-scale experiment |
| 2026-08-20 14:19 | /experiment-plan | refine-logs/FUTURE_ROBUST_EXPERIMENT_PLAN_20260820_141644.md | plan | Pre-registered future-perturbed counterfactual teacher-audit plan |
| 2026-08-20 14:19 | /experiment-plan | refine-logs/FUTURE_ROBUST_EXPERIMENT_TRACKER_20260820_141644.md | tracker | Execution tracker for future-robust label certification |
| 2026-08-20 14:19 | /run-experiment | projection_train/future_robust_simexpert.py | code | Future-only traffic perturbation and robust SimExpert label aggregation |
| 2026-08-20 14:19 | /run-experiment | scripts/audit_future_robust_counterfactual.py | code | CLI entry point for future-robust teacher audit |
| 2026-08-20 14:19 | /run-experiment | tests/test_future_robust_simexpert.py | test | Unit coverage for future traffic provenance and robust-label rules |
| 2026-08-20 14:47 | /run-experiment | records/pipeline_runs/future_robust_hangzhou_audit_20260820_143959/ | experiment | Validated R001 future-robust teacher audit: 4 snapshots, 6 futures each, 96 rollouts; no robust label accepted |
| 2026-08-20 14:47 | /analyze-results | refine-logs/FUTURE_ROBUST_R001_RESULT_ANALYSIS_20260820_144721.md | results | R001 NO-GO analysis: 0/4 labels accepted, 0.0 future top-1 flip rate, all alternatives AWT-infeasible |
| 2026-08-20 14:47 | /analyze-results | refine-logs/FUTURE_ROBUST_R001_RESULT_ANALYSIS.md | results | Latest R001 future-robust audit result analysis |
| 2026-08-20 14:49 | /run-experiment | projection_train/future_robust_simexpert.py; scripts/audit_future_robust_counterfactual.py; tests/test_future_robust_simexpert.py | retirement | Removed after validated R001 NO-GO: no robust labels accepted and no future-rank instability; raw audit evidence retained |
