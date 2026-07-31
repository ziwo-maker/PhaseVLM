# Hangzhou Real Stage 2 Diagnosis

Date: 2026-05-24

Scope: read-only analysis of existing remote logs on `direct.virtaicloud.com:30022`.
No new training or online experiment was run.

## Conclusion

The bad Hangzhou real Stage 2 result is mainly a policy/data/selection issue, not a train-test runtime mismatch.

The main run used consistent train/test runtime settings (`seq_len=16`, `image_size=224`, `prefix_tokens=8`, `prompt_mode=image_only_short`, `max_length=64`, `bf16`). However, the offline BC objective selected a policy that strongly over-serves through/EW phases and almost stops choosing NLSL online. Once NLSL is underserved, left-turn queues accumulate in closed loop, especially in the second half of the 3600s episode.

The best existing Hangzhou-real-trained Stage 2 result among inspected logs is W2 epoch10, not the main run:

| Run | Key params | Reward | Avg queue | Avg wait | Avg travel |
|---|---|---:|---:|---:|---:|
| Main best epoch9 | alpha=0.20, temp=0.80, no global weights | -833.031 | 131.033 | 325.256 | 388.489 |
| Main epoch11 | same params | -799.369 | 126.917 | 334.729 | 380.118 |
| C epoch9 | alpha=0.35, temp=0.70 | -781.351 | 110.608 | 309.959 | 368.023 |
| W1 epoch12 | alpha=0.45, temp=0.60, weights=0.95,0.95,1.10,1.30 | -762.239 | 99.392 | 229.256 | 353.806 |
| W2 epoch10 | alpha=0.55, temp=0.55, weights=0.90,0.95,1.15,1.45 | -687.958 | 87.350 | 209.328 | 342.320 |
| W2 epoch11 | same as W2 | -719.848 | 96.442 | 239.789 | 354.814 |

For context, the existing 5816-trained checkpoint tested on Hangzhou real is still much better:

| Run | Reward | Avg queue | Avg wait | Avg travel |
|---|---:|---:|---:|---:|
| 5816 Stage2 checkpoint -> Hangzhou real | -632.374 | 69.058 | 55.523 | 322.516 |

That comparison strongly suggests the architecture and online evaluator can work on Hangzhou real; the weakness is the Hangzhou-real Stage 2 training recipe and checkpoint selection.

## Training Data Signals

Training data: `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl`

- Only 1255 grouped samples.
- Action labels: ETWT=505 (40.2%), NTST=447 (35.6%), ELWL=177 (14.1%), NLSL=126 (10.0%).
- Scenario structure: east-west dominant=834 (66.5%), north-south dominant=257 (20.5%), balanced=164 (13.1%).
- Turn structure: through_dominant=812 (64.7%), left_dominant=84 (6.7%), balanced=359 (28.6%).
- Trend structure: stable=1075 (85.7%), growing=87 (6.9%), easing=93 (7.4%).
- Expert-action margin is weak for many samples: 247/1255 (19.7%) have margin < 0.5, 441/1255 (35.1%) have margin < 1.0.
- NLSL has the weakest margins: mean 1.206, with 42/126 (33.3%) margin < 0.5.

Interpretation: the local training/validation split is small, same-distribution, and biased toward through/EW/stable states. It does not represent late closed-loop left-turn starvation well.

## Offline vs Online Mismatch

Main run training curve:

| Epoch | Train F1 | Val F1 | Val loss |
|---:|---:|---:|---:|
| 8 | 0.5246 | 0.4704 | 1.0297 |
| 9 | 0.5348 | 0.5338 | 0.9872 |
| 10 | 0.5647 | 0.4385 | 1.0090 |
| 11 | 0.5663 | 0.5084 | 1.0092 |
| 12 | 0.5816 | 0.4859 | 0.9932 |
| 13 | 0.5734 | 0.4978 | 0.9960 |
| 14 | 0.5812 | 0.4990 | 1.0088 |

Main `best.pt` was selected by `val_macro_f1` at epoch9. But online epoch11 improved reward/queue/travel over epoch9 while losing on wait. W2 is clearer: epoch11 had higher offline val F1 than epoch10 (0.5808 vs 0.5465), but epoch10 had better online reward, queue, wait, and travel.

Therefore `val_macro_f1` is not a reliable checkpoint selector for the final traffic objective.

## Online Failure Mode

Main run action distribution:

| Run | ETWT | NTST | ELWL | NLSL |
|---|---:|---:|---:|---:|
| Train labels | 40.2% | 35.6% | 14.1% | 10.0% |
| Main online epoch9 | 48.2% | 35.7% | 12.2% | 3.9% |
| W2 online epoch10 | 45.8% | 34.3% | 13.4% | 6.5% |
| 5816 checkpoint on real | 37.8% | 32.7% | 16.0% | 13.5% |

Main run phase pressure:

- Average phase queue: ETWT=1.223, NTST=0.979, ELWL=2.147, NLSL=3.716.
- NLSL was the max-queue phase in 860/1920 decisions (44.8%).
- But the policy selected NLSL only 75/1920 decisions (3.9%).
- When NLSL was the max-queue phase, main selected NLSL only 38/860 times (4.4%).
- Main max-queue match rate was only 31.9%; average missed queue gap was 3.034; strong misses with gap >= 10 occurred 180 times.

W2 epoch10 is better but still not perfect:

- NLSL action rate rises to 6.5%.
- NLSL average phase queue drops to 1.821.
- Max-queue match improves to 46.9%.
- Average missed gap falls to 1.409.
- Strong misses with gap >= 10 fall to 48.

Late-episode drift is the main symptom:

| Run | Queue window 0-15m | 15-30m | 30-45m | 45-60m |
|---|---:|---:|---:|---:|
| Main epoch9 | 3.237 | 7.623 | 9.933 | 11.469 |
| W2 epoch10 | 3.102 | 6.854 | 5.550 | 6.092 |

| Run | Wait window 0-15m | 15-30m | 30-45m | 45-60m |
|---|---:|---:|---:|---:|
| Main epoch9 | 114.873 | 442.620 | 796.364 | 925.567 |
| W2 epoch10 | 109.933 | 367.161 | 346.212 | 460.713 |

Worst local intersections in main run:

| Intersection index | Avg queue | Avg wait | Last-window queue | NLSL action | NLSL max |
|---:|---:|---:|---:|---:|---:|
| 4 | 25.01 | 1367.3 | 43.63 | 0.8% | 55.0% |
| 1 | 16.61 | 909.1 | 34.87 | 1.7% | 60.0% |
| 6 | 14.74 | 1319.5 | 33.93 | 1.7% | 40.0% |
| 16 | 10.66 | 633.5 | 11.57 | 0.8% | 48.3% |

This is a left-turn starvation pattern, not random noise.

## Why The Parameters Matter

Main run used:

- `target_mode=mixed`
- `soft_target_alpha=0.2`
- `soft_label_temperature=0.8`
- `disable_class_balance=true`
- `global_action_weights=null`
- `best_metric=val_macro_f1`

Because `mixed` loss is `(1-alpha) * hard CE + alpha * soft candidate-wait loss`, alpha=0.2 means the model mostly learns the single expert label. Since many expert labels have small margins, this amplifies noisy local labels. Temperature=0.8 also makes the soft target smoother, further weakening the candidate-wait ranking signal.

W2 changes exactly the weak points:

- alpha rises from 0.20 to 0.55, so candidate-wait ranking matters more.
- temperature drops from 0.80 to 0.55, so soft targets are sharper.
- global weights downweight ETWT/NTST and upweight ELWL/NLSL, especially NLSL.

This does not fully solve the issue, but it reduces NLSL starvation and late queue drift.

## Optimization Recommendations

1. Treat W2 epoch10 as the current best Hangzhou-real-trained Stage 2 checkpoint among inspected logs.
2. Do not select Stage 2 `best.pt` using only offline `val_macro_f1`; use online composite selection with full or at least guarded 3600s evaluation.
3. Add Stage 2 selection guardrails from existing diagnostics: NLSL action rate >= 6%, max-queue match >= 45%, gap>=10 strong misses <= 60, and no monotonic late-window queue growth.
4. Keep the W2 direction for future runs: higher soft alpha, lower soft temperature, and moderate ELWL/NLSL global weights. Avoid reverting to alpha=0.2/temp=0.8/no weights for Hangzhou real.
5. Improve the dataset, not just the hyperparameters: add or prioritize late-episode, high-volume, growing-trend, left-dominant, and worst-intersection states, especially intersections 1/4/6/16.
6. If using `min_action_margin`, do it with per-action quotas; otherwise NLSL may shrink further because NLSL has the weakest margins.
7. Consider scoring/selection by traffic-objective proxies rather than action imitation: queue, wait, travel, max-queue match, missed gap, phase starvation, and late-window drift.

