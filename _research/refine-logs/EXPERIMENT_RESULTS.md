# Initial Experiment Results — Stage 3 OOD-Guided Adaptation

**Date**: 2026-08-13
**Plan**: refine-logs/STAGE3_OOD_PLAN.md
**Status**: ✅ All 7 arms complete (status.json: no FAILED)
**Output**: `records/pipeline_runs/stage3_ood_20260813/`

---

## Setup (unchanged across arms)

- Source: Jinan pt32/hd256 Stage-2 policy (`stage2_pt32_hd256_20260812/best.pt`, val_acc=73.5%), target Hangzhou
- Collection flow: `anon_4_4_hangzhou_real.json`; test flow: `anon_4_4_hangzhou_real_5816.json` (flow-disjoint)
- 120 collected snapshots → 118 usable (2 corrupted 0-byte snapshot files dropped by pre-filter); **K=120 budget → K_eff=118** (recorded in metadata)
- Full SimExpert counterfactual relabeling with yellow-transition emulation, Pareto + constraint filters, 100% coverage enforced
- Snapshot-level 90/10 train/val split; BC warm-start from source checkpoint (mixed loss); best-pt selection on `val_top1_cost_regret`
- 3 eval seeds (2024/42/123), parallel on 3 GPUs, 95% t-intervals

## Results by Arm (3-seed means)

| Arm | ATT | AQL | AWT | Sum |
|-----|----:|----:|----:|----:|
| zero-shot (K=0) | 335.63 | 199.32 | 67.75 | 602.70 |
| random K=8 | 330.14 | 189.78 | **61.73** | **581.65** |
| random K=32 | **327.50** | **190.53** | 66.19 | 584.22 |
| random K=118 | 337.86 | 206.08 | 81.99 | 625.93 |
| ood K=8 | 331.67 | 195.33 | 72.19 | 599.19 |
| ood K=32 | 328.69 | 192.01 | 67.07 | 587.77 |
| ood K=118 | 334.47 | 202.57 | 83.32 | 620.36 |

Note: seed-level std ≈ 0 for most arms (CityFlow is near-deterministic under fixed seeds); the meaningful variance is training-run noise (see K=118 note below).

## Success Criteria vs Results

| Criterion | Result |
|-----------|--------|
| OOD ≥ random at equal K | ❌ Random ≥ OOD at all K |
| AWT does not degrade with K | ❌ Both methods degrade at K=118 (random: 61.7→82.0; ood: 72.2→83.3) |
| OOD K=120 best overall | ❌ Best arm is random K=8 (Sum 581.7) |

## Key Findings (honest negative result)

1. **OOD-guided selection does not beat random at any K.** At K=8 random is clearly better on AWT (61.7 vs 72.2); at K=32 the two tie within training noise. This matches the negative-outcome hypothesis anticipated in IDEA_REPORT.md: target-network collection states are roughly equally informative for this adaptation setup.
2. **Large-K degradation is acquisition-independent.** Both arms degrade at K=118 vs K=32 on ALL metrics. The old pipeline's "AWT degrades at higher K" problem is reproduced even with full SimExpert labels and the new architecture → the cause is NOT state selection. Candidate causes (for the review loop): (a) SimExpert composite-cost objective (proxies) diverges from online ATT/AQL/AWT as more data is fit; (b) overfitting to a single collection episode (K=118 ≈ the whole flow).
3. **Small-K adaptation helps.** random K=8 improves all three metrics over zero-shot (Sum 602.7→581.7). Both methods beat zero-shot at K=8/32.
4. **Training-noise control at K=118:** both arms select all 118 usable snapshots → identical train sets; the ood/random difference (334.5 vs 337.9 ATT, 83.3 vs 82.0 AWT) comes from best-epoch selection on a noisy 12-snapshot val set (ood picked epoch 2, random epoch 3). This bounds run-to-run variance: ~1% ATT, ~3% AWT.

## Summary

- 7/7 must-run arms completed; zero pipeline failures at final run.
- Main result: **negative for OOD-guided acquisition**; small-K SimExpert adaptation (random K=8) is the best Stage 3 configuration so far.
- Ablation planning skipped per protocol (main result negative/inconclusive).
- Ready for Workflow 2: **YES** (with the negative-result framing).

## Next Step

→ `/auto-review-loop "Stage 3 OOD-guided adaptation: OOD does not beat random; both methods degrade at K=118 — diagnose and redesign"`
