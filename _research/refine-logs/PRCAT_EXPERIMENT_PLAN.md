# PRCAT Experiment Plan

## Pilot Outcome And Stop Decision

The valid full-PRCAT `K=118` pilot used training seed 2024 and CityFlow
evaluation seeds 2024, 42, and 123. It reached `AWT=42.48 +/- 0.22`, versus
zero-shot `43.98`, but worse than the matched unweighted responsive-H120
training seed (`39.44`) and its nine-training-seed mean (`40.97 +/- 0.84`).
The queued training runs for seeds 42 and 123 were invalidated by a process
collision and are excluded. The pilot does not satisfy the planned health gate
for seed expansion; retain it as a negative module result and do not launch
the remaining PRCAT arms without a new preregistered hypothesis.

## Frozen Protocol

| Item | Value |
|---|---|
| Source policy | Jinan pt8/hd64, queue_wait_short, max_length 64 |
| Teacher | policy-responsive CityFlow, H120, decision interval 30s |
| Main budgets | K=32, K=118 |
| Training seeds | 2024, 42, 123, 7, 11, 99, 314, 2718, 9001 |
| Evaluation seeds | 2024, 42, 123 |
| Main loss | mixed CE + cost soft target 0.3 + rank loss 0.05 |
| PRCAT | min weight 0.15; source prior KL 0.5 when weight <=0.30 |

## Ordered Runs

1. Complete PRCAT `K=118` pilot seed 2024 and three online evaluation seeds.
2. Repeat PRCAT `K=118` for seeds 42 and 123 only after the pilot is healthy.
3. Compare the three-seed PRCAT pilot to the exact unweighted responsive-H120
   split and source checkpoint.
4. If PRCAT does not worsen AWT or safety metrics, extend to nine training
   seeds and `K=32`.
5. Run held-out Jinan-2500 and a second-network replication. Do not select
   weights using either final test flow.

## Ablations

| Arm | Isolates |
|---|---|
| responsive H120, unweighted | policy-responsive teacher alone |
| PRCAT weight only | advantage-weighted counterfactual supervision |
| PRCAT KL only | conservative source-action prior |
| PRCAT full | interaction of trust weight and prior |
| frozen c30 | frozen-neighbor control |

## Stop Rules

- Stop an arm on traceback, OOM, or absent GPU allocation during its 5-minute
  startup monitor; repair the launcher before retrying.
- Kill the PRCAT line if three training seeds degrade AWT by more than 5% and
  reveal no safety or held-out-transfer benefit.
- Do not elevate a one-seed result to the paper claim.
