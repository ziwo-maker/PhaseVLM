# Pipeline Summary

**Problem：** frozen-neighbor counterfactual labels do not preserve the multi-intersection response that a deployed Stage-2 traffic policy encounters.  
**Final Method Thesis：** Construct the Stage-2 teacher by forcing only the target's first candidate action, then rolling the full network forward under the frozen source policy, and distill its existing four-action costs with the unchanged student.  
**Final Verdict：** `REVISE`  
**Date：** 2026-08-18

## Final Deliverables

- Proposal：`refine-logs/FINAL_PROPOSAL.md`
- Review summary：`refine-logs/REVIEW_SUMMARY.md`
- Experiment plan：`refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker：`refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- **Dominant contribution：** deployment-consistent policy-responsive counterfactual teacher construction for Stage 2.
- **Supporting implementation：** reuse of existing four-action schema and supervised trainer; it is not a second contribution.
- **Explicitly rejected complexity：** PRCAT, CRCD, OOD selection, PPO, new VLM modules and test-time planning.

## Must-Prove Claims

- Same-H120 frozen and policy-responsive teachers differ because of the non-target controller, not horizon or loss.
- In a target with source-policy headroom, the PRCT teacher produces safer, lower-AWT closed-loop Stage-2 adaptation than frozen-H120.
- The method is bounded: no headroom can correctly yield no gain.

## First Runs to Launch

1. P001/P002: audit metadata, snapshot schema and deterministic PRCT replay.
2. P003/P004: frozen-H120 vs PRCT-H120 handler-only teacher audit.
3. P005/P006: three training-seed paired causal gate at Jinan -> Hangzhou, K=118.

## Main Risks

- **Matched control may erase the apparent benefit：** then demote PRCT to a diagnostic observation.
- **Safety may worsen despite mean AWT：** stop on tail wait/spillback/action-entropy guardrail failure.
- **Three seed pilot is insufficient for a paper claim：** only request nine-seed/second-network confirmation after the <=24 GPUh gate passes.
- **External novelty not yet verified：** no "first" statement until a direct related-work audit is complete.

## Next Action

- Proceed to `/run-experiment` only for M0 -> M2. M4 requires a new compute decision.

