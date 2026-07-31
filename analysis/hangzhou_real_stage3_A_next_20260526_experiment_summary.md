# Hangzhou Real Stage3 A-Next 实验总结

日期：2026-05-26

远端结果目录：

`/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real/A_next_20260525_151223`

本地相关文件：

- 详细分析：`analysis/hangzhou_real_stage3_A_next_20260526_analysis.md`
- 上一轮 A/B/C 总结：`analysis/hangzhou_real_stage3_abc_20260525_summary.md`
- A-next 脚本：`scripts/run_hangzhou_real_stage3_A_next_20260525.sh`
- A-next 汇总脚本：`scripts/summarize_hangzhou_real_A_next_20260525.py`

## 实验目的

上一轮 A/B/C 已经证明，Hangzhou real Stage3 的主要问题不是 PPO 训练不够，而是闭环高负载状态下模型经常错过当前 max-queue phase。`A override gap=2` 可以直接修正这个问题，并显著降低 queue、wait 和 travel。

这次 A-next 的目标是继续围绕 A 做小范围优化，重点验证两个假设：

1. `left_wait_protect=45` 是否过于保守；
2. 给 override 加 `target_queue_floor` 或 `total_queue_floor` 是否能减少低负载误触发。

## 实验配置

基础 checkpoint：

`stage3_hz5816e4_to_hzreal_full3600_lr1e6_clip002_kl08_ent5e4_u4_eval2_bs8ga2_waitmax_guard_20260524_161825/checkpoints/best.pt`

在线测试设置：

- dataset: `hangzhou`
- traffic file: `anon_4_4_hangzhou_real.json`
- run counts: `3600`
- min action time: `30`
- checkpoint: same Stage3 best checkpoint
- override base: `max_phase_override_gap=2` or `3`

共跑了 11 个在线测试：

- `gap2_p45` 三个 seed：2024/2025/2026
- `gap3_p45` 三个 seed：2024/2025/2026
- `gap2_p30_s2024`
- `gap2_p60_s2024`
- `gap2_p45_tq3_s2024`
- `gap2_p45_total5_s2024`
- `gap2_p45_tq3_total5_s2024`

## 注意事项

远端第一次生成的 `summary.md/json` 有一个汇总脚本 bug：主指标被读成了 0。原因是 `test_result.json` 的真实指标在 `results` 子字段里，而脚本直接从顶层读取。已修正：

```python
payload = _load_json(result_path)
result = payload.get("results", payload) if isinstance(payload, dict) else {}
```

修正后已在远端重新生成正确的 `summary.md/json`。

## Raw Results

基线是 Stage3 pre-A：

- reward: `-625.1703`
- avg queue: `70.7167`
- avg wait: `55.1263`
- avg travel: `325.8897`

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

## 多 Seed 结论

`gap2_p45` 三个 seed：

- reward mean `-565.9152`, std `7.7447`
- avg queue mean `58.4111`, std `0.4956`
- avg wait mean `44.2053`, std `0.1405`
- avg travel mean `309.0526`, std `0.4094`
- override mean `265.7/1920`
- strong misses `0`

`gap3_p45` 三个 seed：

- reward mean `-578.8780`, std `2.8941`
- avg queue mean `60.6389`, std `0.3416`
- avg wait mean `44.0893`, std `1.4157`
- avg travel mean `311.1800`, std `0.7078`
- override mean `138.3/1920`
- strong misses `0`

结论：`gap2_p45` 的平均 reward、queue 和 travel 明显好于 `gap3_p45`。上一轮看到的 `gap3` wait 优势不稳定，seed 2026 上 wait 明显变差。

## 关键发现

1. `gap2_p60_s2024` 是本轮单次最优。相对 Stage3 pre-A，reward 提升 `+69.03`，avg queue 降 `19.1%`，avg wait 降 `23.0%`，avg travel 降 `5.8%`。

2. 相对旧 A 最优 `gap2_p45_s2025/s2026`，`gap2_p60_s2024` 仍然更好：reward `+5.30`，avg queue `-0.95`，avg wait `-1.84`，avg travel `-1.97`，queuing vehicles `-114`。

3. `left_wait_protect=45` 大概率偏保守。`protect=60` 只保护 9 次左转，但 NLSL action rate 仍约 `11.2%`，没有跌破 10% 安全线。当前主矛盾仍然是 through/max-phase 清队列，而不是左转饥饿。

4. queue floor 方向失败。`target_queue_floor=3` 和 `total_queue_floor=5` 都显著减少有效 override，使 max-phase match 降低、missed gap 上升，最终 reward/queue/wait 全部变差。

5. 不应继续简单减少 override。A 的收益来自及时纠正 max-phase miss，floor 把很多有效纠错挡掉了。更合理的方向是放松 left protect 或换成动作占比 guard，而不是加全局 queue floor。

## 当前决策

短期推荐把 A 的候选默认从：

```text
gap=2, left_wait_protect=45
```

升级为待确认候选：

```text
gap=2, left_wait_protect=60
```

但 `protect=60` 目前只跑了 seed 2024，所以还不能直接作为最终稳定结论。需要补 seed 2025/2026。

## 下一步实验

| Priority | Variant | 目的 |
|---|---|---|
| 1 | gap2 protect=60 seed 2025/2026 | 验证 p60 是否稳定超过 p45 |
| 2 | gap2 protect=75/90 seed 2024 | 看是否还能继续放松 left protect |
| 3 | gap2 protect=0 seed 2024 | 测上界和左转风险，不作为默认 |
| 4 | gap2 protect=60 + NLSL action-rate guard | 如果更高 protect 继续变好，用 action-rate guard 替代 wait protect |

不建议继续 `target_queue_floor` 或 `total_queue_floor`，除非改成更温和的 per-phase/per-intersection 条件。
