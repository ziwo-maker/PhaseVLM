# Hangzhou Real Stage 3 A/B/C Results

Remote run tag: `abc_20260525_194332` on `/gemini/code/LLMSTSC`.

Baseline for deltas: latest Stage3 before A/B/C, reward `-625.1703`, avg queue `70.7167`, avg wait `55.1263`, avg travel `325.8897`.

## Raw Results

| Run | Reward | Delta | Avg queue | Queue delta | Avg wait | Wait delta | Avg travel | Travel delta | Queuing vehicles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A override gap=2 | -561.4438 | +63.7265 | 58.1250 | -17.8% | 44.2864 | -19.7% | 308.8163 | -5.2% | 6975 |
| A override gap=3 | -577.2071 | +47.9632 | 60.4417 | -14.5% | 43.2719 | -21.5% | 310.7714 | -4.6% | 7253 |
| A override gap=5 | -618.5408 | +6.6295 | 66.8000 | -5.5% | 53.9695 | -2.1% | 319.7295 | -1.9% | 8016 |
| A override gap=8 | -628.0177 | -2.8474 | 68.9917 | -2.4% | 57.9465 | +5.1% | 321.9236 | -1.2% | 8279 |
| B long rollout 60 | -636.6782 | -11.5079 | 71.2667 | +0.8% | 59.1567 | +7.3% | 325.4086 | -0.1% | 8552 |
| C snapshot reset | -636.2466 | -11.0763 | 72.6250 | +2.7% | 61.4875 | +11.5% | 327.1572 | +0.4% | 8715 |

## State-Action Diagnostics

| Run | Avg queue proxy | Max-phase match | Avg missed gap | Strong misses | Override applied |
|---|---:|---:|---:|---:|---:|
| A override gap=2 | 3.6026 | 69.06% | 0.1943 | 0 | 266/1920 |
| A override gap=3 | 3.7490 | 63.44% | 0.3255 | 0 | 139/1920 |
| A override gap=5 | 4.1380 | 57.24% | 0.5563 | 0 | 36/1920 |
| A override gap=8 | 4.2839 | 56.88% | 0.6729 | 0 | 7/1920 |
| B long rollout 60 | 4.4271 | 54.38% | 0.7797 | 6 | 0 |
| C snapshot reset | 4.5036 | 54.32% | 0.8089 | 13 | 0 |

## PPO Selection

`B_longroll60_lr5e7_clip001_kl12_ent0` selected update `0`; updates 2 and 4 were rejected by guardrails.

- Eval 0: avg queue `68.8667`, avg wait `54.9510`, avg travel `322.9165`
- Eval 2: avg queue `70.5750`, avg wait `57.8804`, avg travel `324.8642`; rejected for queue and wait ratio
- Eval 4: avg queue `69.1667`, avg wait `62.6847`, avg travel `323.6463`; rejected for wait ratio

`C_snapshot_worst15_45_roll32_lr5e7_clip001_kl12_ent0` selected update `4`; it loaded 48 snapshots / 768 rows from the 15-45 minute high-load window.

- Eval 0: avg queue `68.8667`, avg wait `54.9510`, avg travel `322.9165`
- Eval 2: avg queue `69.9667`, avg wait `55.1150`, avg travel `326.0218`
- Eval 4: avg queue `70.0417`, avg wait `54.1234`, avg travel `325.6326`; accepted, but final online test worsened

## Findings

1. A is the clear winner. `gap=2` gives the best reward, queue, travel, and queuing vehicles; `gap=3` gives the best waiting time. Both are much better than the pre-A/B/C baseline.
2. The override effect is monotonic with intervention strength. Stronger intervention (`gap=2`) raises max-phase match to 69.06% and removes strong misses; weak intervention (`gap=8`) barely changes behavior and loses on reward/wait.
3. B failed. Longer rollout alone did not solve the PPO issue; the run still selected update 0, and final test is worse than baseline.
4. C partly improved training eval but failed final online test. Snapshot reset reduced some eval wait metrics, but the accepted update had worse closed-loop queue/wait/travel in the final online script, indicating eval/selection mismatch or overfitting to the selected snapshots.
5. The main actionable conclusion is that the control heuristic is correcting a real state-conditional timing error. PPO should be redesigned to learn this max-phase-gap correction, not simply trained longer.

## Recommended Next Step

Treat `A override gap=2` as the current best controller and immediately run a 3-seed confirmation for gap `2` and `3`. If the improvement holds, either keep the override as a production guardrail or distill its decisions into targeted BC/PPO data.
