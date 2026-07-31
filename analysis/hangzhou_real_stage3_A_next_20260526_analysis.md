# Hangzhou Real Stage3 A-Next 结果分析

日期：2026-05-26

远端目录：`/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/A_next_20260525_151223`

说明：远端原 `summary.md/json` 一开始由于汇总脚本读取层级错误，把主指标写成了 0。已修正 `scripts/summarize_hangzhou_real_A_next_20260525.py`，重新生成正确 summary。

## Raw Results

基线：Stage3 pre-A，reward `-625.1703`，avg queue `70.7167`，avg wait `55.1263`，avg travel `325.8897`。

| Variant | Reward | Delta | Avg queue | Queue delta | Avg wait | Wait delta | Avg travel | Travel delta | Queuing vehicles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gap2_p60_s2024 | -556.1445 | +69.0258 | 57.1750 | -19.1% | 42.4464 | -23.0% | 306.8421 | -5.8% | 6861 |
| gap2_p45_s2025 | -561.4438 | +63.7265 | 58.1250 | -17.8% | 44.2864 | -19.7% | 308.8163 | -5.2% | 6975 |
| gap2_p45_s2026 | -561.4438 | +63.7265 | 58.1250 | -17.8% | 44.2864 | -19.7% | 308.8163 | -5.2% | 6975 |
| gap2_p30_s2024 | -572.1657 | +53.0046 | 58.3583 | -17.5% | 43.1394 | -21.7% | 308.0587 | -5.5% | 7003 |
| gap2_p45_s2024 | -574.8580 | +50.3123 | 58.9833 | -16.6% | 44.0430 | -20.1% | 309.5253 | -5.0% | 7078 |
| gap3_p45_s2024 | -577.2071 | +47.9632 | 60.4417 | -14.5% | 43.2719 | -21.5% | 310.7714 | -4.6% | 7253 |
| gap3_p45_s2025 | -577.2071 | +47.9632 | 60.4417 | -14.5% | 43.2719 | -21.5% | 310.7714 | -4.6% | 7253 |
| gap3_p45_s2026 | -582.2198 | +42.9505 | 61.0333 | -13.7% | 45.7240 | -17.1% | 311.9973 | -4.3% | 7324 |
| gap2_p45_tq3_s2024 | -589.7279 | +35.4424 | 61.5083 | -13.0% | 45.6142 | -17.3% | 313.6990 | -3.7% | 7381 |
| gap2_p45_total5_s2024 | -604.0063 | +21.1640 | 62.8167 | -11.2% | 49.9141 | -9.5% | 314.8471 | -3.4% | 7538 |
| gap2_p45_tq3_total5_s2024 | -604.5406 | +20.6297 | 62.8833 | -11.1% | 49.9514 | -9.4% | 314.9293 | -3.4% | 7546 |

## Diagnostics

| Variant | Max-phase match | Avg missed gap | Strong misses | Override applied | Protect reasons |
|---|---:|---:|---:|---:|---|
| gap2_p60_s2024 | 69.90% | 0.6176 | 0 | 272/1920 | left_wait_protect:9 |
| gap2_p45_s2025 | 69.06% | 0.6279 | 0 | 266/1920 | left_wait_protect:15 |
| gap2_p45_s2026 | 69.06% | 0.6279 | 0 | 266/1920 | left_wait_protect:15 |
| gap2_p30_s2024 | 68.75% | 0.6517 | 0 | 266/1920 | left_wait_protect:14 |
| gap2_p45_s2024 | 68.33% | 0.6217 | 0 | 265/1920 | left_wait_protect:16 |
| gap3_p45_s2024 | 63.44% | 0.8903 | 0 | 139/1920 | left_wait_protect:8 |
| gap3_p45_s2025 | 63.44% | 0.8903 | 0 | 139/1920 | left_wait_protect:8 |
| gap3_p45_s2026 | 62.97% | 0.8945 | 0 | 137/1920 | left_wait_protect:7 |
| gap2_p45_tq3_s2024 | 66.04% | 0.9310 | 0 | 188/1920 | left_wait_protect:20, target_queue_floor:115 |
| gap2_p45_total5_s2024 | 60.83% | 1.0731 | 0 | 133/1920 | left_wait_protect:13, total_queue_floor:177 |
| gap2_p45_tq3_total5_s2024 | 60.83% | 1.0771 | 0 | 133/1920 | left_wait_protect:13, target_queue_floor:96, total_queue_floor:82 |

## Multi-Seed Stability

`gap2_p45` 三个 seed：

- reward mean `-565.9152`, std `7.7447`
- avg queue mean `58.4111`, std `0.4956`
- avg wait mean `44.2053`, std `0.1405`
- avg travel mean `309.0526`, std `0.4094`
- overrides mean `265.7/1920`
- strong misses `0`

`gap3_p45` 三个 seed：

- reward mean `-578.8780`, std `2.8941`
- avg queue mean `60.6389`, std `0.3416`
- avg wait mean `44.0893`, std `1.4157`
- avg travel mean `311.1800`, std `0.7078`
- overrides mean `138.3/1920`
- strong misses `0`

`gap2_p45` 的平均 reward、queue、travel 明显好于 `gap3_p45`。`gap3_p45` 不再稳定占优 wait，因为 seed 2026 wait 明显变差。

## Key Findings

1. `gap2_p60_s2024` 是本轮最优。相对 Stage3 pre-A，它 reward 提升 `+69.03`，avg queue 降 `19.1%`，avg wait 降 `23.0%`，avg travel 降 `5.8%`。

2. 相对之前最好的 `gap2_p45`，`gap2_p60_s2024` 仍有增益：reward `+5.30`，avg queue `-0.95`，avg wait `-1.84`，avg travel `-1.97`，queuing vehicles `-114`。

3. left-wait protect 越保守不一定越好。同 seed 2024 下，`protect=60` 比 `protect=45/30` 都好，说明目前不是左转饥饿主导，而是 through/max-phase 纠错仍然偏保守。`protect=60` 只触发 9 次 left protect，但 NLSL 动作仍有 `215/1920=11.2%`，没有跌破 10% 安全线。

4. queue floor 方向失败。`target_queue>=3` 把 overrides 从 265 降到 188，reward 降到 `-589.73`；`total_queue>=5` 把 overrides 降到 133，reward 降到 `-604.01`。这些 floor 去掉了太多有效纠错。

5. max-phase match 与结果一致。`gap2_p60` match `69.90%`、missed gap `0.6176`，强 miss 为 0；queue-floor 版本 match 掉到 `60.83%-66.04%`，missed gap 上升到 `0.93-1.08`，性能同步变差。

## Decision

短期推荐把 A 的默认版本从 `gap=2, protect=45` 改为候选 `gap=2, protect=60`，但需要补 3-seed 确认。当前 `protect=60` 只跑了 seed 2024，不能直接宣称最终稳定最优。

下一轮最小实验：

| Priority | Variant | Purpose |
|---|---|---|
| 1 | gap2 protect=60 seed 2025/2026 | 验证 p60 是否稳定超过 p45 |
| 2 | gap2 protect=75/90 seed 2024 | 看是否还能继续放松 left protect |
| 3 | gap2 protect=0 seed 2024 | 测上界和左转风险，不作为默认 |
| 4 | gap2 protect=60 + NLSL action-rate guard | 如果 protect 更高继续变好，用 action-rate guard 替代 wait protect |

不要继续推 `target_queue_floor` 或 `total_queue_floor`，除非改成更温和的 per-phase/per-intersection 条件。
