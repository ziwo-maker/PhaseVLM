# Hangzhou 5816 Online vs Train Traffic Diagnosis

## Compared Artifacts

- Online test: `stage2_from_stage1temp007_lr1e4_a025_bal_bs9ga7_ep14_trainonly_20260521_165725/online_test_best/anon_4_4_hangzhou_real_5816.json_05_21_17_53_28`
- Local train/expert run: `records/sim_expert_validation_search/probe_hangzhou_5816_container/best/work`
- Raw traffic file md5: both `a840c69aa266b6f60a914f3d2863ce96`
- Roadnet md5: both `49cfccccd1e3451c6d1fff36fd98c229`

The raw demand file and roadnet are identical. The differences below are therefore not caused by testing on a different traffic JSON. `traffic_counts_*` are simulator lane-occupancy snapshots, so their differences reflect how the different policies change congestion and vehicle positions over time.

## Metric Gap

| Source | Avg queue | Avg wait | Avg travel | Final queue | Reward |
|---|---:|---:|---:|---:|---:|
| Train/expert log | 159.44 | 35.00 | 317.11 | 19133 | -554.22 |
| Online Stage 2 | 188.30 | 56.73 | 333.49 | 22596 | -666.64 |
| Online - train | +28.86 | +21.73 | +16.38 | +3463 | -112.42 |

## Where Traffic Occupancy Gets Worse

### By Phase Group

| Phase group | Train | Online | Delta | Pct |
|---|---:|---:|---:|---:|
| ETWT | 409032 | 403555 | -5477 | -1.3% |
| NTST | 163992 | 190371 | +26379 | +16.1% |
| ELWL | 79450 | 90927 | +11477 | +14.4% |
| NLSL | 47183 | 62361 | +15178 | +32.2% |

The online run is not worse on east-west through traffic. The load increase is mainly north-south through and left-turn traffic.

### By Movement

| Movement | Train | Online | Delta | Pct |
|---|---:|---:|---:|---:|
| NT | 92143 | 104736 | +12593 | +13.7% |
| ST | 71849 | 85635 | +13786 | +19.2% |
| SL | 24111 | 35138 | +11027 | +45.7% |
| NL | 23072 | 27223 | +4151 | +18.0% |
| EL | 35905 | 41520 | +5615 | +15.6% |
| WL | 43545 | 49407 | +5862 | +13.5% |
| ET | 213021 | 211878 | -1143 | -0.5% |
| WT | 196011 | 191677 | -4334 | -2.2% |

### Top Intersections

| Intersection | Train | Online | Delta | Pct |
|---|---:|---:|---:|---:|
| intersection_1_1 | 60343 | 71382 | +11039 | +18.3% |
| intersection_4_4 | 139966 | 148559 | +8593 | +6.1% |
| intersection_1_4 | 116808 | 122201 | +5393 | +4.6% |
| intersection_1_2 | 46493 | 50246 | +3753 | +8.1% |
| intersection_4_1 | 19519 | 22987 | +3468 | +17.8% |
| intersection_4_3 | 29631 | 33032 | +3401 | +11.5% |

### Top Intersection-Movement Increases

| Intersection | Movement | Train | Online | Delta | Pct |
|---|---|---:|---:|---:|---:|
| intersection_1_4 | NT | 16912 | 23822 | +6910 | +40.9% |
| intersection_1_1 | ST | 9393 | 15789 | +6396 | +68.1% |
| intersection_1_1 | NT | 8474 | 13424 | +4950 | +58.4% |
| intersection_4_4 | SL | 4167 | 9023 | +4856 | +116.5% |
| intersection_4_4 | ST | 9012 | 13296 | +4284 | +47.5% |
| intersection_1_3 | EL | 1661 | 3880 | +2219 | +133.6% |

## When It Diverges

| Window | Train avg occupancy | Online avg occupancy | Delta | Pct |
|---|---:|---:|---:|---:|
| 00-15m | 122.16 | 132.71 | +10.55 | +8.6% |
| 15-30m | 251.28 | 268.22 | +16.93 | +6.7% |
| 30-45m | 438.99 | 465.07 | +26.08 | +5.9% |
| 45-60m | 742.36 | 794.48 | +52.12 | +7.0% |

Sustained online-minus-train occupancy gap first exceeds:

- 25 vehicles at 838s, about 14.0m.
- 50 vehicles at 2558s, about 42.6m.
- 75 vehicles at 3312s, about 55.2m.

This is a cumulative closed-loop degradation, not an immediate startup mismatch.

## Action and Queue Diagnosis

### Action Distribution

| Source | ETWT | NTST | ELWL | NLSL |
|---|---:|---:|---:|---:|
| Train/expert | 784 (40.83%) | 618 (32.19%) | 302 (15.73%) | 216 (11.25%) |
| Online model | 751 (39.11%) | 626 (32.60%) | 314 (16.35%) | 229 (11.93%) |

The online action histogram is close to the train/expert action histogram. The failure is not global action collapse.

### State-Conditional Behavior

| Source | Avg total queue | Max-phase match | Avg missed gap | Strong misses |
|---|---:|---:|---:|---:|
| Train/expert | 9.719 | 65.31% | 1.264 | 70 |
| Online model | 11.448 | 54.69% | 1.864 | 79 |

Online has more queue, lower match to the currently largest queued phase, and larger missed gaps.

### Avg Queue by Phase Group

| Phase group | Train | Online | Delta | Pct |
|---|---:|---:|---:|---:|
| ETWT | 5.417 | 5.330 | -0.087 | -1.6% |
| NTST | 1.922 | 2.823 | +0.901 | +46.8% |
| ELWL | 1.309 | 1.717 | +0.408 | +31.2% |
| NLSL | 1.070 | 1.578 | +0.508 | +47.4% |

The queue increase matches the traffic-count signal: north-south and left-turn phases accumulate.

### Online Action vs Current Max-Queue Phase

| Current max phase | ETWT chosen | NTST chosen | ELWL chosen | NLSL chosen |
|---|---:|---:|---:|---:|
| ETWT | 503 | 112 | 90 | 71 |
| NTST | 170 | 387 | 83 | 59 |
| ELWL | 40 | 84 | 94 | 33 |
| NLSL | 38 | 43 | 47 | 66 |

When the largest queue is `NTST`, the model still chooses `ETWT` 170 times. When the largest queue is `NLSL`, it chooses `NLSL` only 66 out of 194 times.

### Worst Same-Index Cases

| Intersection | Step | Train q | Online q | Delta q | Train action | Online action | Online max | Online missed gap |
|---|---:|---:|---:|---:|---|---|---|---:|
| intersection_1_1 | 118 | 73 | 121 | +48 | ELWL | ETWT | NTST | 23 |
| intersection_1_1 | 105 | 24 | 66 | +42 | NTST | ETWT | NTST | 26 |
| intersection_1_1 | 119 | 95 | 131 | +36 | ETWT | ELWL | NTST | 48 |
| intersection_1_1 | 109 | 43 | 79 | +36 | NTST | NTST | NTST | 0 |
| intersection_1_4 | 118 | 92 | 124 | +32 | ETWT | ELWL | ETWT | 10 |

## Conclusion

The raw traffic demand is the same. The online Stage 2 policy creates a worse closed-loop trajectory: it lets north-south and left-turn queues build up, especially around `intersection_1_1`, `intersection_1_4`, and `intersection_4_4`. Its global action distribution looks normal, but the timing is wrong: it does not reliably choose the phase with the current largest queue once congestion drifts away from the training trajectory. The degradation starts mildly before 15 minutes and becomes large in the last 15 minutes.

Most likely root cause: the model learned the marginal expert action distribution better than the state-to-action control law. Offline train/validation labels from one expert trajectory are not enough to cover the late congested states reached by the online model, so small action-timing errors compound into north-south and left-turn accumulation.

## Suggested Next Checks

1. Re-run online evaluation with checkpoint selection by online queue/wait, not offline F1.
2. Add short online rollouts for Stage 2 model selection on Hangzhou 5816.
3. Generate more expert data covering late congested states, especially `intersection_1_1`, `1_4`, `4_4`, `NTST`, and `NLSL`.
4. Add a validation proxy that scores max-phase missed gap and phase-specific queue, not only action F1.
5. Consider margin-weighted labels or reduced balancing for weak left-turn labels so ambiguous rare classes do not dominate timing.
