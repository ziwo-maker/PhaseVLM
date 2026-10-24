# DUCPD Robustness Execution Plan

**Problem**: Validate whether direct observed queue/wait urgency with `AWT=2` remains preferable to hard behavioral cloning beyond the initial three-seed, single-flow result.

**Method thesis**: A calibrated waiting-time term should preserve queue-clearing behavior while reducing prolonged observed vehicle waiting; excessive waiting emphasis already failed in the `AWT=2.5` ablation.

## Claim Map

| Claim | Minimum convincing evidence | Execution block |
|---|---|---|
| C1: `AWT=2` improves the primary Stage 2 comparison | Paired hard-BC vs DUCPD results across nine independently trained seeds on the fixed target full flow | B1 |
| C2: the effect is not confined to one temporal intensity pattern | Same frozen checkpoints evaluated on multiple demand files and three non-overlapping target-flow time windows | B2 |
| C3: mean waiting improvement does not conceal observed long waits | Vehicle-level p95/max continuous waiting metrics are extracted by the online evaluator | B3 |

## Fixed Protocol

- Training arms: `hard_bc` and `wait2_mixed_rank` only. The equal and `AWT=2.5` arms remain the completed mechanism ablation, not part of the nine-seed expansion.
- Training seeds: existing `{2024, 42, 123}` plus new `{7, 11, 99, 314, 2718, 9001}`.
- Training: 15 epochs, fixed Stage 1 initialization, 3-GPU DDP, validation accuracy checkpoint selection. Online evaluation never selects checkpoints.
- Evaluation scenarios: full target `Hangzhou_real`; training-source-demand reference `5816`; synthetic high-demand stress; and target-real windows `[0,1200)`, `[1200,2400)`, `[2400,3600)`.
- Unit of inference: training seed. The CityFlow online seed is fixed at 2024 and is not a statistical replicate.

## Tail Metric Definition

For each non-shadow vehicle ever observed in `waiting_vehicle_list` at an action boundary, retain its maximum CityFlow continuous waiting time. Report p95 and max over these one-value-per-vehicle maxima. This is a vehicle-level observed queueing tail metric, not a completed-trip delay distribution or a formal safety guarantee.

## Execution Order

1. Build and hash the three target-flow time windows.
2. Run twelve additional serial 3-GPU DDP training jobs (two arms x six seeds), with five-minute startup observation for every training job.
3. Evaluate all 18 checkpoints on six scenarios. The 108 frozen-checkpoint evaluations are distributed one worker per GPU.
4. Aggregate paired seed deltas and preserve all incomplete tasks rather than silently excluding them.

## Scope Boundaries

- All outcomes are CityFlow closed-loop simulation outcomes, not deployed Hangzhou intersection results.
- The supervision remains existing rule-derived labels plus direct observed queue/wait urgency; this campaign does not run candidate rollouts or counterfactual relabeling.
- The `5816` full-flow and its training data are related. It is a demand-reference stress condition, not an independent test set.
