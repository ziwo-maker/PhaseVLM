# Hangzhou Real Stage3 A/B/C 复盘与 A 优化空间

日期：2026-05-25

相关记录：
- A/B/C 结果汇总：`analysis/hangzhou_real_stage3_abc_20260525_summary.md`
- Stage3 诊断：`analysis/hangzhou_real_stage3_diagnosis_20260525/analysis_summary.md`
- A 的实现入口：`projection_train/run_projection_online.py`
- 远端运行 tag：`abc_20260525_194332`

## 结论先行

这轮最重要的经验是：问题不是“PPO 训练不够久”，而是当前策略在闭环高负载状态下有一个明确的相位时机错误。A 的 max-phase queue override 直接修正这个错误，所以立刻有效；B 的长 rollout 和 C 的 snapshot reset 没有改变这个决策机制，所以没有带来收益。

当前应该把 `A override gap=2` 当作 Hangzhou-real 的临时最优控制器，但不要急着把它视为最终答案。A 还有提升空间，不过提升方向不是继续盲目降低 gap，而是把固定阈值改成“按相位、路口、时间窗、等待风险自适应”的 override。

## 原始结果

基线是 A/B/C 前最新 Stage3：reward `-625.1703`，avg queue `70.7167`，avg wait `55.1263`，avg travel `325.8897`。

| Run | Reward | Delta | Avg queue | Queue delta | Avg wait | Wait delta | Avg travel | Travel delta | Queuing vehicles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A override gap=2 | -561.4438 | +63.7265 | 58.1250 | -17.8% | 44.2864 | -19.7% | 308.8163 | -5.2% | 6975 |
| A override gap=3 | -577.2071 | +47.9632 | 60.4417 | -14.5% | 43.2719 | -21.5% | 310.7714 | -4.6% | 7253 |
| A override gap=5 | -618.5408 | +6.6295 | 66.8000 | -5.5% | 53.9695 | -2.1% | 319.7295 | -1.9% | 8016 |
| A override gap=8 | -628.0177 | -2.8474 | 68.9917 | -2.4% | 57.9465 | +5.1% | 321.9236 | -1.2% | 8279 |
| B long rollout 60 | -636.6782 | -11.5079 | 71.2667 | +0.8% | 59.1567 | +7.3% | 325.4086 | -0.1% | 8552 |
| C snapshot reset | -636.2466 | -11.0763 | 72.6250 | +2.7% | 61.4875 | +11.5% | 327.1572 | +0.4% | 8715 |

| Run | Avg queue proxy | Max-phase match | Avg missed gap | Strong misses | Override applied |
|---|---:|---:|---:|---:|---:|
| A override gap=2 | 3.6026 | 69.06% | 0.1943 | 0 | 266/1920 |
| A override gap=3 | 3.7490 | 63.44% | 0.3255 | 0 | 139/1920 |
| A override gap=5 | 4.1380 | 57.24% | 0.5563 | 0 | 36/1920 |
| A override gap=8 | 4.2839 | 56.88% | 0.6729 | 0 | 7/1920 |
| B long rollout 60 | 4.4271 | 54.38% | 0.7797 | 6 | 0 |
| C snapshot reset | 4.5036 | 54.32% | 0.8089 | 13 | 0 |

## 这次学到的经验

1. 不要把“guardrail 保住了 update 0”误读成“Stage3 PPO 成功学习”。最新 Stage3 的 `best.pt` 选的是 update `0`，后续 PPO update 被拒绝。成功的是选择机制避免了退化，不是 PPO 找到了更好策略。

2. 坏结果要先拆成闭环状态分布问题。和本地 train traffic log 相比，online 平均 occupancy 只高 `3.9%`，但集中在 ET/WT through movement、`intersection_1_4`、`intersection_4_4`、`intersection_1_3`，以及 15-45 分钟窗口。这个定位比只看 reward 有用。

3. 当前残余问题不是 NLSL 左转崩溃。Stage3 online 的 NLSL action rate 是 `13.7%`，高于本地 train label 的 `10.0%`。现在更像是 ETWT/through 高负载状态下没有及时服务。

4. A 有效说明“max queue phase miss”是真问题。A 前 max-phase match 约 `55.2%`，strong miss 有 `7` 次；A gap=2 后 match 到 `69.06%`，strong miss 清零，reward、queue、wait、travel 全部明显改善。

5. B 失败说明单纯拉长 rollout 不够。B 仍然选 update `0`，final test 比基线更差，说明 PPO 的数据或目标没有命中关键错误。

6. C 失败说明 snapshot eval 和最终闭环在线测试有错位。C 在 eval 中接受了 update `4`，但 final online queue/wait/travel 都变差。后续 selection 不能只看平均 eval，还要看 15-45 分钟、最差路口、max-phase miss。

7. 强 override 不是越多越好。gap=2 比 gap=3 多了 `127` 次 override，queue、travel、reward 更好，但 wait 比 gap=3 高约 `1.01`。真正值得优化的是这 127 次额外 override 里，哪些降低队列，哪些只是增加等待。

## A 是否还有提升空间

有，而且方向比较清楚。但空间大概率不是再做一个固定阈值扫出来，而是把 gap=2 和 gap=3 的优点合并：尽量保住 gap=2 的 queue/reward/travel，同时接近 gap=3 的 wait。

当前 A 的规则很简单：

- 找当前 queue 最大的 phase；
- 如果该 phase 和模型动作不同，且 queue gap 大于阈值，就 override；
- 如果模型原本选的是左转 phase，而目标是 through phase，并且左转 wait 超过 `45`，就保护左转，不 override。

这个规则已经抓住了主矛盾，但还有几个盲点：

- 它只看 raw queue gap，不看 total queue，也不区分轻载路口和重载路口；
- 它对 ETWT、NTST、ELWL、NLSL 使用同一个 gap 阈值；
- 它只保护“模型已经选择左转”的情况，不能主动用 wait bonus 识别高等待左转；
- 它不区分 15-45 分钟高负载窗口和其他窗口；
- 它不区分最差路口和普通路口；
- 它不看模型置信度，也不看最近是否刚服务过目标 phase。

所以 A 的下一步不应该是“gap=1 更强试试看”。gap=1 很可能进一步降低 queue，但也更容易打乱循环、拉高等待，尤其是在低负载路口或左转保护边界附近。更好的方向是保留 gap=2 的强纠错能力，但加条件。

## 推荐的 A 系列下一轮实验

### P0：先确认 gap=2 和 gap=3 的稳定性

先做 3-seed 或 5-seed：

| Variant | 目的 |
|---|---|
| A gap=2 protect=45 | 当前 best，确认均值和方差 |
| A gap=3 protect=45 | 当前 best-wait，确认是不是稳定低 wait |

必须同时记录 reward、avg queue、avg wait、avg travel、queuing vehicles、max-phase match、avg missed gap、strong misses、override rate、NLSL action rate。

如果 gap=2 的多 seed 方差很大，后面先做稳定性；如果 gap=2 稳定优于 gap=3，后面直接以 gap=2 为主线。

### P1：left wait protect sweep

当前 A 使用 `max_phase_override_left_wait_protect=45`。这个值很可能影响 gap=2 和 gap=3 的 wait/queue tradeoff。

| Variant | 预期作用 |
|---|---|
| gap=2 protect=30 | 更早保护左转，可能降低 wait，但 queue/travel 可能回升 |
| gap=2 protect=45 | 当前 best |
| gap=2 protect=60 | 更偏 through 清队列，可能继续降 queue，但 wait 风险更高 |
| gap=3 protect=30/45/60 | 看 gap=3 的低 wait 是否来自更少 override，还是左转保护边界 |

这组实验能回答一个关键问题：gap=2 比 gap=3 多出来的 127 次 override，到底是不是主要伤在左转等待。

### P2：加入 absolute queue floor

固定 gap=2 在轻载状态下可能太敏感。例如原动作 queue=0、目标 queue=2 时也会 override，但这种切换未必值得。可以试：

```text
override if gap >= 2 and target_queue >= 3
override if gap >= 2 and total_queue >= 5
override if gap >= 2 and (target_queue >= 3 or total_queue >= 5)
```

这类规则的目标是保留高负载纠错，减少轻载噪声 override。它最有可能把 gap=2 的 wait 往 gap=3 拉近，同时不明显牺牲 queue。

### P3：phase-specific threshold

不同相位不应该共用一个阈值。ETWT 是当前主问题，但左转有等待风险。

候选规则：

| Target / original | 建议 |
|---|---|
| target 是 ETWT/NTST，original 也是 through | gap=2 |
| target 是 ETWT/NTST，original 是 left | gap=2 + 更强 left wait protect，或 gap=3 |
| target 是 ELWL/NLSL | gap=2，但要求 left wait 或 left queue 确实高 |

这样做的目的不是提高 max-phase match 到 100%。交通信号需要公平轮转，max-phase match 只是 guardrail。目标是消除大的 miss，同时避免为了小 queue gap 牺牲等待。

### P4：time-window adaptive

诊断显示 online 比 train 更差主要发生在 15-30 分钟和 30-45 分钟：

| Window | Train avg occupancy | Online avg occupancy |
|---|---:|---:|
| 00-15m | 10.987 | 10.329 |
| 15-30m | 14.136 | 16.077 |
| 30-45m | 11.043 | 11.997 |
| 45-60m | 10.830 | 10.838 |

可以尝试：

```text
15-45m: gap=2
其他时间: gap=3 或 gap=5
```

如果收益主要来自 15-45 分钟，这个版本可能比全程 gap=2 更稳。

### P5：intersection-specific threshold

最差路口不是均匀分布，优先关注：

- `intersection_1_4`
- `intersection_4_4`
- `intersection_1_3`
- `intersection_4_2`

候选规则：

```text
worst intersections: gap=2
other intersections: gap=3
low-load intersections: gap=5 or disabled
```

这个方向适合在 P0/P1 后做，因为它能降低全局副作用，但需要确认不同 seed 下最差路口是否稳定。

### P6：queue + wait hybrid score

当前 target phase 是 raw queue 最大的 phase。可以改成：

```text
score(phase) = queue(phase) + alpha * wait_bonus(phase)
target_phase = argmax score
```

其中 `wait_bonus` 可以用 clipped/normalized wait，尤其给 ELWL/NLSL 一个温和补偿。这个方向要小心：现在主问题不是左转崩溃，所以 alpha 不能大。建议只在 P1 证明 wait 确实被 gap=2 伤到后再做。

### P7：confidence-aware override

如果能拿到模型 action probability 或 logit margin，可以让 override 只在模型不确定时更激进：

```text
low confidence: gap=2
high confidence: gap=3 or keep model action
```

这比纯规则更接近“只纠正模型容易犯错的状态”。如果当前 policy 接口拿不到 logits/probs，就先不做这个，避免改动过大。

### P8：把 A 变成训练信号，而不是永久外挂

A gap=2 已经提供了高价值反例：模型原动作、override 后动作、queue gap、phase queue、phase wait 都能记录下来。后续可以：

- 只抽取 overridden states；
- 用 A 的 final action 做 pseudo-label；
- 对 gap 大、strong miss、最差路口、15-45 分钟样本加权；
- 做一轮 targeted BC 或 BC+PPO；
- online test 时先不启用 override，看策略是否真的学会。

但这个要排在多 seed 确认之后。否则可能把单次 seed 的启发式噪声蒸馏进模型。

## 最小可执行实验矩阵

如果只想跑最少实验，建议这样排：

| Priority | Variant | 判断标准 |
|---|---|---|
| 1 | gap=2 protect=45, 3 seeds | 确认当前 best 是否稳定 |
| 2 | gap=3 protect=45, 3 seeds | 确认 best-wait 是否稳定 |
| 3 | gap=2 protect=30 | 看能否降低 wait 且少损 queue |
| 4 | gap=2 + target_queue>=3 | 看能否去掉轻载噪声 override |
| 5 | 15-45m gap=2, others gap=3 | 看收益是否来自高负载窗口 |
| 6 | worst intersections gap=2, others gap=3 | 看能否减少全局副作用 |

如果第 3 或第 4 个实验能把 avg wait 拉到 `43.x`，同时 avg queue 仍接近 `58-59`，就比当前 gap=2 更值得作为下一版 A。

## 下一版验收标准

不要只用 reward 选。建议同时满足：

- reward 优于 `-561.44`，或 reward 接近但 avg wait 明显低于 `44.29`；
- avg queue 不高于 `59.0`；
- avg wait 尽量接近或低于 `43.27`；
- avg travel 不高于 `309.0`；
- strong misses 保持 `0`；
- max-phase match 不必越高越好，但 avg missed gap 应保持在 `0.2-0.3` 附近；
- NLSL action rate 不低于 `10%`；
- 最差路口 `intersection_1_4`、`intersection_4_4`、`intersection_1_3` 不能恶化。

## 我的判断

A 肯定还有提升空间，但它已经把最容易拿的收益吃掉了。下一步的收益应该来自“少数场景少犯错”，不是全局更强硬。

最值得先试的是两个方向：

1. `gap=2 + target_queue floor`：保留强纠错，减少轻载误触发。
2. `gap=2 protect=30/45/60`：弄清楚 wait 损失是不是左转保护阈值导致的。

如果这两个都没有超过当前 gap=2，就把 gap=2 作为 runtime guardrail 固化，同时把 overridden states 蒸馏回模型。B/C 的结果已经说明，继续盲跑 PPO 不如让训练直接学习 A 暴露出来的错误边界。
