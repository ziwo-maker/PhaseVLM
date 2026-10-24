# JSCD 混入 Hangzhou_5816 扩容实验结果分析

**日期：** 2026-08-20  
**运行目录：** `records/pipeline_runs/jscd_mix5816_scale_20260820_100052/`  
**最终结论：** 9/9 任务正常完成，但混入现有 5816 rule-teacher 数据后，三种训练臂的主指标 AWT 均相对原 JSCD P2 退化。JSCD 仍在 3/3 种子上优于 Static-PRCT，却无法优于 Frozen；不能扩展 JSCD 的性能主张或启动 P3。

## 1. 完整性与比较口径

- 完成状态：`frozen`、`static_prct`、`jscd` 各 3 个训练 seed (`2024/42/123`)，共 9/9 为 `OK`。训练与线上评估日志未出现 traceback、NaN/Inf、OOM、失败重试或缺失 checkpoint。
- 训练预算：每臂使用相同 mixed loss、5 epoch、相同架构、相同 warm start、3 卡 DDP 和相同 `val_top1_cost_regret` checkpoint-selection 规则。
- primary P2 数据：86 个 `Hangzhou_real` JSCD P2 快照，每种子 77/9 snapshot train/validation，1,232/144 条记录。
- supplement：从 120 个完整的 5816 snapshot 中固定选取 86 个，每种子同样按 77/9 snapshot 切分，得到 1,232/144 条记录。因此混合后每个 arm/seed 是 **172 snapshots、2,464/288 train/validation records**。
- 隔离审计：对于同一个 training seed，F/P/J 的 5816 train hash 和 validation hash 完全一致；三臂只保留原 P2 的 teacher/continuation 差异。所有线上评估均为 `anon_4_4_hangzhou_real.json`、3600 秒、`eval_seed=2024`、`queue_wait_short`。
- 统计单位：训练 seed。线上评估在固定 flow/controller/seed 配置下是确定性的，不能把它视为额外独立重复。

## 2. 逐种子线上结果

所有指标均为越低越好，格式为 `ATT / AQL / AWT`。

| 训练 seed | Frozen | Static-PRCT | JSCD |
|---:|---:|---:|---:|
| 2024 | 313.41 / 61.73 / 54.52 | 314.30 / 62.41 / 56.50 | 314.91 / 62.54 / 55.57 |
| 42 | 311.54 / 61.01 / 51.31 | 313.05 / 62.55 / 53.83 | 314.91 / 62.59 / 52.68 |
| 123 | 314.80 / 62.54 / 57.32 | 316.24 / 64.14 / 59.24 | 316.27 / 64.69 / 57.09 |
| 均值 +/- sample std | **313.25 +/- 1.63 / 61.76 +/- 0.77 / 54.38 +/- 3.00** | 314.53 +/- 1.61 / 63.03 +/- 0.96 / 56.52 +/- 2.70 | 315.36 +/- 0.79 / 63.28 +/- 1.23 / 55.11 +/- 2.24 |

混合训练后，Frozen 是唯一同时拥有最低均值 ATT、AQL 和 AWT 的臂。

## 3. 相对原 P2 的数据扩容效应

下表 `Delta = mixed - original P2`，负值有利。原 P2 与本实验按同一三个训练 seed、同一 `Hangzhou_real` 测试配置对齐。

| 臂 | 原 P2 ATT / AQL / AWT | 混合后 ATT / AQL / AWT | Delta ATT / AQL / AWT | AWT 相对变化 |
|---|---:|---:|---:|---:|
| Frozen | 314.36 / 62.69 / 50.27 | 313.25 / 61.76 / 54.38 | -1.11 / -0.93 / **+4.11** | **+8.18%** |
| Static-PRCT | 313.90 / 62.73 / 50.83 | 314.53 / 63.03 / 56.52 | +0.63 / +0.30 / **+5.70** | **+11.21%** |
| JSCD | 312.72 / 61.77 / 48.03 | 315.36 / 63.28 / 55.11 | +2.64 / +1.50 / **+7.09** | **+14.75%** |

逐种子方向也不支持扩容收益：Frozen 的 AWT 仅 1/3 改善；Static-PRCT 与 JSCD 的 AWT 均为 0/3 改善。故不能把增加现有 5816 数据作为 JSCD 的有效训练扩容策略。

## 4. JSCD 的配对机制结论

`Delta = JSCD - reference`，负值有利于 JSCD。

| 对比 | ATT Delta | AQL Delta | AWT Delta | AWT 方向 | exact Wilcoxon |
|---|---:|---:|---:|---:|---:|
| JSCD - Frozen | +2.11 | +1.51 | **+0.73** | 1/3 favorable | 0.50 |
| JSCD - Static-PRCT | +0.83 | +0.24 | **-1.41** | 3/3 favorable | 0.25 |
| Static-PRCT - Frozen | +1.28 | +1.27 | **+2.14** | 0/3 favorable | 0.25 |

与原 P2 相比，JSCD-vs-Frozen 的 AWT 效应从 `-2.25`（2/3 favorable）反转为 `+0.73`（1/3 favorable）。JSCD-vs-Static-PRCT 的 AWT 优势则从 `-2.80`（2/3 favorable）缩小为 `-1.41`（3/3 favorable）。

因此，混合训练后唯一仍成立的方向性观察是：JSCD 比 Static-PRCT 少退化一些。它不是性能优势，因为 Frozen 仍是所有主指标最优的匹配控制。三个训练 seed 的精确检验也均不能形成显著性主张。

## 5. 解释与边界

### 5.1 这不是纯粹的“更多数据”消融

5816 supplement 的完整标签带有 `label_source=rule`、`rule_gap=2.0` 和 `left_wait_protect=60.0`。P2 primary 则来自 `policy_responsive_neighbors`、H=120、global-cost 的反事实 teacher。故本试验同时改变了：

1. 样本量，从 86 到 172 snapshots；
2. 流量来源，从 Hangzhou_real 加入 5816；
3. 教师语义，从 counterfactual cost ranking 加入 rule teacher。

所有臂接收同一份 supplement，所以 F/P/J 的横向相对比较是公平的；但与原 P2 的纵向变化不能归因于样本量本身。较合理的解释是 rule-teacher 与反事实 teacher 的决策边界不兼容，尤其会破坏 JSCD 学到的联合策略一致性。

### 5.2 离线选择指标没有证明线上可靠性

例如 Static-PRCT/seed-123 的最佳 validation regret 是本组中较低的 `0.293673`，但其线上 AWT 是所有九个 mixed 模型中最差的 `59.24`。这不是充分的相关性分析，但说明该混合语义下的 `val_top1_cost_regret` 不能单独用作部署收益证据。

### 5.3 仍缺少安全性结论

本轮仅记录 ATT/AQL/AWT。p95/max movement wait、spillback、action entropy 与 phase frequency 仍未抽取，不能声称 JSCD 或混合训练在安全性上有效。

## 6. 决策与下一步

1. **停止把现有 5816 rule 数据直接混入 JSCD P2 的训练。** 该配方在三臂上均抬高 AWT，JSCD 的恶化最严重。
2. **不启动 P3 或更多 seed。** 原 P2 已失败 Frozen 的 3/3 gate；此处也只在 1/3 对上赢 Frozen，且平均方向为不利。
3. 若需回答“训练集更大是否有帮助”，唯一有解释力的下一实验是**同 teacher 语义的两倍数据控制**：在额外、与 primary 不重叠的 snapshots 上，分别生成 Frozen、Static-PRCT 和 JSCD H=120 标签，而不是共享 rule labels。三臂仍使用 matched data amount、source checkpoint 和 seed。
4. 因 5816 已被用于训练，后续泛化评估必须换到未训练的第三个流量条件或严格隔离的时间段；不能再把 5816 作为本模型的 held-out generalization result。
5. 在任何新的 JSCD pilot 前，仍应先修复原 P2 记录的 `pi_0` replay/score drift，并补全预注册的 safety extraction。

## 可引用的谨慎表述

> 在一个三训练种子、训练集从 86 扩展到 172 snapshots 的受控混合实验中，将现有 Hangzhou_5816 rule-teacher 监督与 Hangzhou_real counterfactual 数据混合，使 Frozen、Static-PRCT 和 JSCD 的平均 AWT 均上升。JSCD 在每个种子上均优于混合后的 Static-PRCT，但仍劣于 Frozen，因而该结果不支持通过该数据混合策略改善 JSCD 的部署性能。
