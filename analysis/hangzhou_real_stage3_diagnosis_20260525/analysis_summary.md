# Hangzhou Real Stage 3 Diagnosis

## Latest Remote Result

| Run | Reward | Avg queue | Avg wait | Avg travel | Queuing vehicles |
|---|---:|---:|---:|---:|---:|
| Latest Stage3 2026-05-24 16:18 -> test 17:08 | -625.170 | 70.717 | 55.126 | 325.890 | 8486 |

This is the newest Hangzhou-real Stage 3 run found under remote `/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real`.

### Context Against Recent Hangzhou-Real Runs

| Run | Reward | Avg queue | Avg wait | Avg travel |
|---|---:|---:|---:|---:|
| Latest Stage3 | -625.170 | 70.717 | 55.126 | 325.890 |
| 5816 Stage2 checkpoint -> Hangzhou real | -632.374 | 69.058 | 55.523 | 322.516 |
| Hangzhou-real Stage2 W2 epoch10 | -687.958 | 87.350 | 209.328 | 342.320 |
| Hangzhou-real Stage2 main best epoch9 | -833.031 | 131.033 | 325.256 | 388.489 |

### Stage3 PPO Selection

- `best.pt` was selected at PPO update `0`.
- This means the final tested checkpoint is the guarded baseline/start checkpoint for this Stage3 run, not an accepted later PPO update.

| Eval update | Reward | Avg queue | Avg wait | Avg travel | Score |
|---:|---:|---:|---:|---:|---:|
| 0 | -1636.369 | 68.867 | 54.951 | 322.916 | -1.000 |
| 2 | -1889.931 | 71.692 | 57.874 | 327.906 | -1.038 |
| 4 | -1806.812 | 71.567 | 56.825 | 326.885 | -1.028 |

Rejected updates:
- update 2: eval_queue_ratio=1.0410 > 1.0200; eval_wait_ratio=1.0532 > 1.0300; eval_travel_ratio=1.0155 > 1.0100
- update 4: eval_queue_ratio=1.0392 > 1.0200; eval_wait_ratio=1.0341 > 1.0300; eval_travel_ratio=1.0123 > 1.0100

## Traffic Log Comparison

- Local train traffic log: `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl` (20080 rows, compact sampled train windows).
- Remote Stage3 closed-loop log: `analysis/hangzhou_real_stage3_diagnosis_20260525/remote_traffic_counts_2s.full.jsonl` (28800 rows, full 60-minute 16-intersection rollout).
- Mean occupancy per row: train 11.852, online 12.310, delta 0.458 (3.9%).
- Direction share train -> online: EW 65.7% -> 67.8%, NS 34.3% -> 32.2%, left 21.1% -> 20.9%.

### By Phase Group

| Phase | Train per row | Online per row | Delta | Pct | Train share | Online share |
|---|---:|---:|---:|---:|---:|---:|
| ETWT | 6.206 | 6.722 | 0.517 | 8.3% | 52.4% | 54.6% |
| NTST | 3.142 | 3.016 | -0.126 | -4.0% | 26.5% | 24.5% |
| ELWL | 1.575 | 1.626 | 0.051 | 3.2% | 13.3% | 13.2% |
| NLSL | 0.929 | 0.946 | 0.017 | 1.8% | 7.8% | 7.7% |

### Biggest Movement Increases

| Movement | Train per row | Online per row | Delta | Pct | Share delta |
|---|---:|---:|---:|---:|---:|
| WT | 3.494 | 3.755 | 0.261 | 7.5% | 1.0 pp |
| ET | 2.712 | 2.967 | 0.255 | 9.4% | 1.2 pp |
| EL | 0.626 | 0.688 | 0.062 | 9.9% | 0.3 pp |
| SL | 0.460 | 0.472 | 0.012 | 2.6% | -0.0 pp |
| NL | 0.469 | 0.474 | 0.005 | 1.0% | -0.1 pp |
| WL | 0.949 | 0.938 | -0.011 | -1.2% | -0.4 pp |
| NT | 1.673 | 1.620 | -0.054 | -3.2% | -1.0 pp |
| ST | 1.469 | 1.396 | -0.072 | -4.9% | -1.0 pp |

### Biggest Intersection Increases

| Intersection | Train per row | Online per row | Delta | Pct |
|---|---:|---:|---:|---:|
| intersection_1_4 | 23.618 | 26.889 | 3.270 | 13.8% |
| intersection_4_4 | 17.981 | 20.867 | 2.885 | 16.0% |
| intersection_1_3 | 9.671 | 11.591 | 1.920 | 19.9% |
| intersection_4_2 | 9.067 | 10.597 | 1.530 | 16.9% |
| intersection_4_1 | 6.608 | 7.371 | 0.764 | 11.6% |
| intersection_2_3 | 8.122 | 8.819 | 0.697 | 8.6% |
| intersection_1_2 | 11.414 | 12.014 | 0.600 | 5.3% |
| intersection_1_1 | 16.511 | 17.054 | 0.543 | 3.3% |

### Biggest Intersection-Movement Increases

| Intersection | Movement | Train per row | Online per row | Delta | Pct |
|---|---|---:|---:|---:|---:|
| intersection_1_4 | WT | 11.564 | 14.442 | 2.878 | 24.9% |
| intersection_4_4 | ET | 8.825 | 11.224 | 2.399 | 27.2% |
| intersection_2_4 | WT | 6.630 | 7.551 | 0.922 | 13.9% |
| intersection_1_3 | NT | 3.719 | 4.501 | 0.782 | 21.0% |
| intersection_4_4 | SL | 1.096 | 1.838 | 0.742 | 67.8% |
| intersection_4_1 | ST | 2.406 | 3.148 | 0.742 | 30.8% |
| intersection_1_3 | EL | 0.866 | 1.539 | 0.674 | 77.8% |
| intersection_1_3 | ET | 2.466 | 3.102 | 0.636 | 25.8% |
| intersection_2_2 | WT | 2.598 | 3.181 | 0.582 | 22.4% |
| intersection_4_3 | ET | 3.249 | 3.818 | 0.569 | 17.5% |

### Time Windows

| Window | Train avg occupancy | Online avg occupancy |
|---|---:|---:|
| 00-15m | 10.987 | 10.329 |
| 15-30m | 14.136 | 16.077 |
| 30-45m | 11.043 | 11.997 |
| 45-60m | 10.830 | 10.838 |

## Action And Queue Diagnosis

| Source | ETWT | NTST | ELWL | NLSL |
|---|---:|---:|---:|---:|
| Local train labels | 505 (40.2%) | 447 (35.6%) | 177 (14.1%) | 126 (10.0%) |
| Stage3 online model actions | 709 (36.9%) | 649 (33.8%) | 299 (15.6%) | 263 (13.7%) |

- The remote prompt JSONL contains 1920 action-tagged decision rows, matching the 1920 entries in `state_action`.
- Avg total queue proxy from `state_action`: 4.388.
- Max-queue phase match: 55.2%.
- Avg missed queue gap: 0.738; strong misses gap>=10: 7.

### Avg Queue By Phase

| Phase | Avg queue | Avg wait proxy | Current max-phase rate | Action rate |
|---|---:|---:|---:|---:|
| ETWT | 1.947 | 9.694 | 56.2% | 36.9% |
| NTST | 0.954 | 7.375 | 21.6% | 33.8% |
| ELWL | 0.866 | 25.804 | 15.4% | 15.6% |
| NLSL | 0.620 | 21.723 | 6.8% | 13.7% |

### Online Action Given Current Max-Queue Phase

| Current max phase | ETWT chosen | NTST chosen | ELWL chosen | NLSL chosen |
|---|---:|---:|---:|---:|
| ETWT | 595 | 226 | 128 | 130 |
| NTST | 52 | 297 | 27 | 39 |
| ELWL | 41 | 86 | 121 | 47 |
| NLSL | 21 | 40 | 23 | 47 |

### Worst Intersections By Queue Proxy

| Intersection | Avg total queue | Match | Avg missed gap | Strong misses | Actions | Max phases |
|---|---:|---:|---:|---:|---|---|
| intersection_1_4 | 10.275 | 63.3% | 1.017 | 2 | ETWT 54 (45.0%), NTST 35 (29.2%), ELWL 18 (15.0%), NLSL 13 (10.8%) | ETWT 62 (51.7%), NTST 29 (24.2%), ELWL 19 (15.8%), NLSL 10 (8.3%) |
| intersection_4_4 | 7.508 | 55.0% | 1.217 | 1 | ETWT 48 (40.0%), NTST 36 (30.0%), ELWL 29 (24.2%), NLSL 7 (5.8%) | ETWT 76 (63.3%), NTST 20 (16.7%), ELWL 17 (14.2%), NLSL 7 (5.8%) |
| intersection_1_1 | 6.475 | 60.0% | 0.842 | 1 | ETWT 43 (35.8%), NTST 43 (35.8%), ELWL 16 (13.3%), NLSL 18 (15.0%) | ETWT 59 (49.2%), NTST 31 (25.8%), ELWL 14 (11.7%), NLSL 16 (13.3%) |
| intersection_3_4 | 4.933 | 65.8% | 0.742 | 1 | ETWT 52 (43.3%), NTST 36 (30.0%), ELWL 28 (23.3%), NLSL 4 (3.3%) | ETWT 81 (67.5%), NTST 13 (10.8%), ELWL 22 (18.3%), NLSL 4 (3.3%) |
| intersection_1_3 | 4.775 | 45.0% | 1.050 | 0 | ETWT 25 (20.8%), NTST 45 (37.5%), ELWL 5 (4.2%), NLSL 45 (37.5%) | ETWT 53 (44.2%), NTST 46 (38.3%), ELWL 21 (17.5%), NLSL 0 (0.0%) |
| intersection_2_4 | 4.358 | 67.5% | 0.625 | 1 | ETWT 51 (42.5%), NTST 30 (25.0%), ELWL 31 (25.8%), NLSL 8 (6.7%) | ETWT 76 (63.3%), NTST 17 (14.2%), ELWL 22 (18.3%), NLSL 5 (4.2%) |
| intersection_1_2 | 4.175 | 55.0% | 0.583 | 0 | ETWT 43 (35.8%), NTST 48 (40.0%), ELWL 11 (9.2%), NLSL 18 (15.0%) | ETWT 67 (55.8%), NTST 25 (20.8%), ELWL 17 (14.2%), NLSL 11 (9.2%) |
| intersection_4_2 | 3.792 | 53.3% | 0.692 | 0 | ETWT 46 (38.3%), NTST 38 (31.7%), ELWL 18 (15.0%), NLSL 18 (15.0%) | ETWT 67 (55.8%), NTST 20 (16.7%), ELWL 17 (14.2%), NLSL 16 (13.3%) |

## Diagnosis

1. The latest Stage3 result is not the bad Hangzhou-real result seen in earlier Stage2 runs; it is slightly better than the previous 5816-to-real online result and far better than the Hangzhou-real Stage2 main/best guarded runs.
2. PPO did not improve this run: update 2 and update 4 were worse than update 0 on queue/wait/travel and were rejected by the guardrails. So the latest result is good mainly because checkpoint selection preserved the start checkpoint, not because the Hangzhou-real PPO adaptation succeeded.
3. The remaining gap versus the local generated train traffic log is modest but real: online mean occupancy is 3.9% higher, concentrated in ET/WT through traffic and high-load intersections `intersection_1_4`, `intersection_4_4`, and `intersection_1_3`.
4. Stage3 avoids the severe NLSL collapse seen in the bad Stage2 run. NLSL action rate is 13.7%, above the 10.0% local train label rate. The residual problem is now ETWT/through timing and high-load intersection handling, not left-turn starvation.
5. The 15-30m and 30-45m windows are where online occupancy exceeds the sampled train log most clearly; the last 15 minutes no longer show the runaway late drift from the bad Stage2 diagnosis.
6. The model still under-matches the instantaneous max-queue phase often enough for queues to accumulate. ETWT is the current max-queue phase 56.2% of the time but is selected 36.9% of the time, although many misses are small.
7. The train set is compact and biased toward stable/east-west/through cases; rare high-load closed-loop states are still weakly represented. Stage3's reward and guardrails reduce this, but do not remove the extrapolation problem.

## Suggested Next Experiments

1. Keep this Stage3 checkpoint as the current Hangzhou-real candidate unless a newer run beats -625.17 reward under the same 3600s test.
2. Compare against a true expert rollout metric for `anon_4_4_hangzhou_real.json`, not only the compact sampled train log, to quantify the remaining reward gap.
3. Add expert data or online relabeling from the worst high-queue intersections and late windows found above, then re-run Stage2/Stage3 selection with max-phase match and missed-gap guardrails.
4. Preserve the Stage3 wait/starvation guard direction; do not fall back to the earlier Stage2 offline-F1 checkpoint selection that caused left-turn starvation.
