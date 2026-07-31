# Hangzhou 5816 Stage 2 Online Diagnosis

## Run

- Remote run: `records/pipeline_runs/exp_hangzhou_5816/stage2_from_stage1temp007_lr1e4_a025_bal_bs9ga7_ep14_trainonly_20260521_165725`
- Local artifact copy: `analysis/hangzhou5816_stage2_online_20260521/`
- Best checkpoint: epoch 12
- Best offline metrics: train F1 0.6540, val F1 0.5124
- Final epoch metrics: train F1 0.6398, val F1 0.4927

## Online Result

| Source | Reward | Avg Queue | Avg Wait | Avg Travel | Final Queue |
|---|---:|---:|---:|---:|---:|
| Local expert data log | -554.22 | 159.44 | 35.00 | 317.11 | 19133 |
| Stage 2 online test | -666.64 | 188.30 | 56.73 | 333.49 | 22596 |
| Delta | -112.42 | +28.86 | +21.73 | +16.38 | +3463 |

The online queue trace is monotonic after the early phase: first 10 queue points are `0, 0, 5, 15, 21, 43, 63, 91, 119, 159`; last 10 are `18186, 18568, 19027, 19455, 19921, 20424, 20946, 21461, 22005, 22596`.

## Data Signals

- Training groups: 1335; validation groups: 165; test groups: 165.
- Only one expert data-generation candidate was used: Hangzhou 5816, horizon 60, weights ATT=1, AQL=1, AWT=2, spillback=0.2.
- Label distribution: ETWT 36.03%, NTST 34.23%, ELWL 17.08%, NLSL 12.66%.
- Class balancing changed effective hard-loss weights to ETWT 0.571, NTST 0.601, ELWL 1.204, NLSL 1.624.
- Expert score margins are weak for left-turn labels: median margin ETWT 2.559, NTST 1.758, ELWL 0.755, NLSL 0.623.
- Traffic coverage is skewed: east-west dominant 891/1335 groups, north-south dominant 307/1335, balanced 137/1335; through-dominant 1023/1335, left-dominant only 60/1335.

## Policy Behavior

- Online action distribution stayed close to training labels: ETWT 39.11%, NTST 32.60%, ELWL 16.35%, NLSL 11.93%.
- This means the failure is not a simple global action-count collapse.
- The failure is closed-loop state mismatch: as queues grow, the policy does not adapt enough to the currently worst phase.
- Chosen action matched the current max-queue phase only 56.25% of decisions.
- Mean missed queue gap increased over the episode: 0.423 -> 1.212 -> 2.290 -> 3.529 by quarter.
- There were 79 decisions where another phase had at least 10 more queued vehicles than the chosen phase.

## Likely Root Cause

The model learned the marginal expert action distribution from a small single-trajectory dataset, but not a robust state-to-action control law. Offline val F1 around 0.51 means roughly half of held-out expert labels are still wrong, and in closed loop those errors compound: the state distribution drifts into congested states that are poorly covered by the training data. Class balancing and soft targets help rare actions appear, but the rare left-turn labels have the weakest score margins, so the model is being asked to upweight the noisiest labels. The online action histogram looks superficially correct, yet queue-aware decisions degrade over time.

## Next Checks

1. Run checkpoint-level online eval for epoch 5, 10, 12, and the no-class-balance run's best checkpoint; select by online queue/wait, not val F1.
2. Regenerate expert data with multiple seeds/horizons and more congested states; the current "best" data comes from only one candidate.
3. Add a blocked/time-based validation split and report online proxy metrics, because random same-trajectory val F1 is not predictive enough.
4. Reduce or remove class balancing for left-turn classes, or weight labels by expert score margin so ambiguous left-turn labels do not dominate.
5. Add online short-rollout selection during Stage 2, similar to Stage 3 guardrails.
