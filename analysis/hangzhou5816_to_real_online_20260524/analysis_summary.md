# Hangzhou 5816 -> Hangzhou Real Online Test Analysis

## Run

- Remote server: `direct.virtaicloud.com:30022`
- Checkpoint: `records/pipeline_runs/exp_hangzhou_5816/stage2_from_stage1temp007_lr1e4_a025_bal_bs9ga7_ep14_trainonly_20260521_165725/checkpoints/best.pt`
- Test traffic: `anon_4_4_hangzhou_real.json`
- Seed: `2024`
- Run counts: default full 3600s (`120` signal decisions per intersection)
- Result JSON: `analysis/hangzhou5816_to_real_online_20260524/online_test_hangzhou_real_from_5816_20260524_110957.json`
- State/action log: `analysis/hangzhou5816_to_real_online_20260524/state_action.json`

## Raw Metrics

| Test file | Reward | Avg queue | Queuing vehicles | Avg wait | Avg travel |
|---|---:|---:|---:|---:|---:|
| `hangzhou_real` tested by 5816 checkpoint | -632.3737 | 69.0583 | 8287 | 55.5225 | 322.5163 |

## Comparisons

| Baseline | Reward | Avg queue | Queuing vehicles | Avg wait | Avg travel |
|---|---:|---:|---:|---:|---:|
| This run: 5816 checkpoint -> `hangzhou_real` | -632.3737 | 69.0583 | 8287 | 55.5225 | 322.5163 |
| Same checkpoint -> `hangzhou_real_5816` | -666.6376 | 188.3000 | 22596 | 56.7310 | 333.4875 |
| Old README Qwen projection -> `hangzhou_real` | -1440.5432 | 670.4250 | 88451.8 | 623.2206 | 975.0054 |

Relative to the same checkpoint on `hangzhou_real_5816`, this run improves avg queue by 63.3%, queuing vehicles by 63.3%, avg wait by 2.1%, avg travel by 3.3%, and reward by 34.26 points.

Relative to the old README Qwen projection result on `hangzhou_real`, this run is much stronger: avg queue is lower by 89.7%, queuing vehicles by 90.6%, avg wait by 91.1%, avg travel by 66.9%, and reward improves by 808.17 points. Treat this as a historical comparison because the exact model/protocol may differ.

## Closed-Loop Behavior

| Statistic | This run | Same checkpoint on 5816 |
|---|---:|---:|
| Decisions | 1920 | 1920 |
| Max-queue phase match | 55.31% | 56.25% |
| Avg missed queue gap | 0.7104 | 1.8635 |
| Strong misses (`gap >= 10`) | 3 | 79 |
| Avg chosen-action probability | 0.5776 | n/a |

Action distribution:

| Run | ETWT | NTST | ELWL | NLSL |
|---|---:|---:|---:|---:|
| This run | 37.81% | 32.66% | 16.04% | 13.49% |
| Same checkpoint on 5816 | 39.11% | 32.60% | 16.35% | 11.93% |

Average phase queue from `state_action.json`:

| Run | ETWT | NTST | ELWL | NLSL |
|---|---:|---:|---:|---:|
| This run | 1.750 | 0.998 | 0.896 | 0.642 |
| Same checkpoint on 5816 | 5.330 | 2.823 | 1.717 | 1.578 |

15-minute state-queue windows:

| Run | 0-15m | 15-30m | 30-45m | 45-60m |
|---|---:|---:|---:|---:|
| This run | 2.881 | 5.983 | 4.223 | 4.054 |
| Same checkpoint on 5816 | 2.467 | 6.523 | 12.350 | 24.452 |

## Findings

1. The 5816-trained checkpoint generalizes well to `hangzhou_real`. Its action distribution stays close to the 5816 run, but the resulting closed-loop queue state is far lighter.
2. The main improvement is not a higher max-queue phase match rate. Match rate is similar, but the mistakes are much less severe: average missed gap falls from 1.86 to 0.71, and strong misses almost disappear.
3. The 5816 test failed by late-episode queue accumulation; this run does not show the same state drift. State queues peak in the 15-30m window and then remain controlled instead of growing through the last 30 minutes.
4. The test file `hangzhou_real` appears easier or better aligned with this policy than `hangzhou_real_5816`, despite the checkpoint being trained from 5816 data.

## Next Checks

1. Re-run once with a different seed if the simulator/model path has any stochasticity; otherwise this single run is a strong positive signal.
2. Test the same checkpoint on `hangzhou_synthetic_24000_60min` to see whether robustness is broad or only transfers between the two real Hangzhou files.
3. Compare against a checkpoint trained on `hangzhou_real` under the same exact online script and model path, because the historical README baseline may not be protocol-identical.
