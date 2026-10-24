# JSCD 下一阶段研究与补实验计划

**日期：** 2026-08-19  
**状态：** 本地审稿人式规划，待外部复核；**仅计划，本文档没有启动任何实验。**  
**唯一主创新：** Joint-Policy Self-Consistent Counterfactual Distillation（JSCD，联合策略自洽反事实蒸馏）。

## 1. 决策摘要

经同 horizon、同训练标签、同 source checkpoint、同学生训练配置和同训练 seed 的匹配对照，原始 PRCT 不能支持“部署性能提升”的主张。下一阶段不应继续扩展 PRCT seed 或调 loss，而应检验一个更根本的假设：**教师的反事实 continuation policy 必须与最终联合部署策略自洽。**

JSCD 只改离线教师生成时的后续控制者；它不引入新学生结构、新损失、在线搜索、强化学习或额外部署开销。若 JSCD 不通过下文 P0/P1 的证据门，本方向停止作为性能创新推进。

## 2. 已有证据账本

### 2.1 校正后的匹配 PRCT 对照

运行根目录：`records/pipeline_runs/prct_hangzhou_matched_h120_20260818_212203/`。基线是专门训练的 `Frozen-neighbor H=120` 学生，不是 zero-shot 或不匹配的历史 checkpoint。两组固定 source Stage-2 checkpoint、1,888 标签/118 snapshots、K、学生损失、优化器、epoch、checkpoint 选择准则和九个训练 seed；唯一变化是首个目标干预后非目标路口的控制规则。

其中 `Delta = PRCT - Frozen`，ATT/AQL/AWT 均越低越好：

| K | 独立训练 seed 数 | ATT Delta | AQL Delta | AWT Delta | 结论 |
|---:|---:|---:|---:|---:|---|
| 32 | 9 | +1.09 | +2.21 | +1.52 | 无改善，所有比较均不显著 |
| 118 | 9 | -0.78 | -0.40 | +3.54 | AWT 恶化 7.19%；Wilcoxon `p=0.0547` |

`K=118` 的绝对聚合为：Frozen `ATT 327.997 +/- 1.860`、`AQL 184.531 +/- 2.056`、`AWT 49.175 +/- 1.822`；PRCT `ATT 327.219 +/- 2.938`、`AQL 184.132 +/- 5.636`、`AWT 52.714 +/- 3.904`。AWT 的配对 Delta 95% CI 为 `[-0.10, +7.18]`，九对中七对恶化。

### 2.2 教师机制确实改变了标签

- PRCT 与 Frozen 的标签在 `513 / 1,888 = 27.17%` 样本上不同。
- expert action agreement 为 `72.83%`，raw-best agreement 为 `72.30%`，平均 Kendall tau 为 `0.6898`。
- 重复 smoke 中，256 个实际反事实 state 的 expert agreement 为 `97.66%`，仅 6 个 expert-label 分歧。
- 改变标签样本的 PRCT candidate winner gap 平均为 `0.80`，未改变样本为 `1.21`；但 `84.6%` 的改变标签仍满足 gap `>= 0.10`，因此简单的 margin filter 不能解释或修复失败。

结论是：PRCT 不是 no-op；负结果不能归因于“教师没产生差异”。

### 2.3 用户指定的 `Hangzhou_real` 再评估

测试文件是 `anon_4_4_hangzhou_real.json`，不是 `anon_4_4_hangzhou_real_5816.json`。在三个**按既有 5816 模式选择**的 K=118 checkpoint pair 上，`Delta = PRCT - Frozen`：

| 训练 seed | ATT Delta | AQL Delta | AWT Delta |
|---:|---:|---:|---:|
| 2024 | +2.61 | +3.08 | +6.24 |
| 11 | +1.99 | +1.10 | -0.21 |
| 123 | +0.52 | -0.32 | +7.29 |
| 选中三对均值 | +1.71 | +1.29 | +4.44 |

该诊断子集的 ATT 为 3/3 恶化，AQL 和 AWT 均为 2/3 恶化，方向上强化了 PRCT 的负面信号；但它是选择性 `n=3`，不是无偏统计样本。`Hangzhou_real` 与采集/教师 snapshots 同流，不能充当 flow-generalization；`5816` 保留为 flow-disjoint 的次要测试。

固定 checkpoint 的 online seed `2024/42/123` 得到相同指标，符合固定交通和确定性控制器的预期。因此，**训练 seed 是唯一独立统计单位**；不得将 online seed 当作独立重复。

### 2.4 应撤销的旧结论

以下历史说法不可再用于论文或计划依据：PRCT 的 `27.9%` AWT 改善，以及 responsive-H120 相对 frozen-c30 的优越性。旧比较同时改变了 target intervention 时长、首个 interval 的邻居控制器及后续邻居控制器，不是匹配因果对照。历史文档 `PRCT_EVIDENCE_CONCLUSION*.md` 仅保留审计价值。

## 3. 失败假设：PRCT 的 policy mismatch

令 `pi_0` 为 source Stage-2 policy，`pi_theta` 为训练后实际部署在每个路口的学生。校正后的 PRCT 对候选动作的估值是：

```text
force target i to candidate a once
then every intersection follows the old source policy pi_0

a_PRCT*(s, i) = argmin_a C(s, i, a; continuation = pi_0)
```

但部署时所有路口共同执行的是 `pi_theta`。所以 PRCT 近似优化的是候选动作在 `Q^(pi_0)` 下的排序，却让新策略在一个不同的联合 policy 下运行。教师排序改变、但线上 AWT 未改善的现象与这个 mismatch 一致。

这一点不是“再加一个风险权重”能解决的问题：loss、ranking、KL 或 temperature 只能改变学生如何拟合一个固定教师，不能把反事实 rollout 的 continuation policy 改成部署 policy。

## 4. JSCD 是什么

**JSCD = Joint-Policy Self-Consistent Counterfactual Distillation，联合策略自洽反事实蒸馏。** 它把 teacher construction 从“用旧 source policy 评价候选动作”改为“用当前将被联合部署的学生 policy 评价候选动作”。

第 `t` 轮的当前策略为 `pi_t`。对 canonical snapshot `s`、目标路口 `i` 与候选相位 `a`：

```text
initial 30-second interval:
  target i executes candidate a
  non-target intersections retain their restored snapshot phases

all later control intervals:
  every intersection, including i, acts with the current policy pi_t

a_JSCD*(s, i; pi_t) = argmin_a C(s, i, a; continuation = pi_t)
```

初始 interval 的恢复/transition 协议、候选集合、cost、feasibility filter 和随机性必须在 PRCT 与 JSCD 中完全锁定；它不是 JSCD 的额外自由度。关键变化仅是后续决策点上所有节点的 controller。

最小外循环只做一次自洽更新：

1. 使用 source policy `pi_0` 按校正 PRCT 语义生成第一轮教师并蒸馏出 `pi_1`。
2. 固定模型结构、数据、student loss、优化器和训练预算，以 `pi_1` 作为所有节点的 continuation policy 生成第二轮 JSCD 教师。
3. 蒸馏得到 `pi_2`，然后停止。不会进行无界的 policy iteration 或在线搜索。

因此 JSCD 是**教师构造创新**，而不是“PRCT 加新 loss”。部署仍是现有 PhaseVLM/Stage-2 学生的一次前向决策，不产生新的 rollout 或时延。

## 5. 主张、反主张与最小证据

| 编号 | 需要验证的主张 | 必须排除的解释 | 最低可信证据 |
|---|---|---|---|
| C1 | source continuation 与 current-policy continuation 给出实质不同的候选排序 | JSCD 只是对 PRCT 的重命名 | 固定 holdout snapshots 上预注册的 rank/action instability |
| C2 | 仅替换第二轮教师 continuation 为当前学生能改善联合部署 | gain 来自第二次训练、更多 rollout 或随机 checkpoint | 计算匹配的三臂、配对训练 seed 对照 |
| C3 | JSCD 不以尾部等待、spillback 或动作塌缩换平均指标 | 仅 AWT 平均值好看 | 双 flow 上的 safety hard gate |

JSCD 当前没有“已验证提升”的性能主张。P0--P2 只是在决定它是否值得成为下一阶段主创新。

## 6. P0--P3 预注册补实验计划

### P0：消除 `Hangzhou_real` 的选择偏差

- **动作：** 对九个既有 K=118 Frozen/PRCT checkpoint pair 全部在 `anon_4_4_hangzhou_real.json` 重评估。
- **统计单位：** 九个训练 seed；每个 checkpoint 仅运行一次确定性的 online evaluation。
- **输出：** 每对 Delta、配对均值、bootstrap CI、Wilcoxon 检验，以及与 `5816` 的方向一致性。
- **目的：** 只完成当前 PRCT 负结果的完整审计，不训练模型，不为 JSCD 制造正例。
- **门：** 无论正负都不改写 PRCT 的主结论；P1 才是 JSCD 的技术可行性门。
- **预算：** 小于 1 GPU-hour。

### P1：先验证自洽性是否会改变反事实决策

- **动作：** 在 32 个固定、未参与第二轮教师生成的 holdout snapshots 上，对四个候选动作分别计算 `pi_0` continuation 与 `pi_1` continuation 的成本、top-1 action、Kendall tau 与 cost-gap sign。
- **一致性检查：** 先以 `pi_0` 重放一次现有 corrected PRCT，逐项比对候选排名和 action；不一致先修实现，不进入 JSCD 训练。
- **P1 通过门：** top-1 action 在至少 20% snapshots 改变，且平均 Kendall tau 不高于 0.80；并人工审查所有 rank reversal 的 transition/feasibility log。阈值是用于避免无意义工程投入的预注册机制门，不是统计显著性声明。
- **P1 失败解释：** `pi_1` 没有足以改变反事实的联合行为，JSCD 不具备可辨识的教师机制；停止性能方向，最多保留 PRCT mismatch 的诊断结论。
- **预算：** 在 32-snapshot timing smoke 通过后约 4 GPU-hours。

### P2：一次自洽更新的三 seed 配对 pilot

仅 P1 通过才执行。对每个训练 seed 使用相同的 source checkpoint、snapshot pool、训练预算、初始化、候选与 horizon `H=120`，并且训练/评测脚本记录 checkpoint hash。

| 臂 | 第一轮 | 第二轮 | 为何存在 |
|---|---|---|---|
| F：Frozen-H120 | frozen-neighbor teacher | 相同 frozen-neighbor teacher | 控制两轮蒸馏本身 |
| P：Static-PRCT | corrected PRCT，以 `pi_0` continuation | 仍以 `pi_0` continuation | 控制额外 rollout/第二轮训练，但保留 mismatch |
| J：JSCD | 与 P 完全相同，得到共享的 `pi_1` | 以共享 `pi_1` 作为全节点 continuation | 唯一改变是 self-consistent continuation |

J 与 P 必须从同一个 `pi_1`、相同样本、相同 update 数、相同 model selection 规则开始第二轮，确保 J-P 只测 JSCD 机制；F 用两轮匹配预算排除“多蒸馏一次”解释。

- **评测：** 主测试 `Hangzhou_real`，次测试 `5816`；每个 checkpoint 单次 deterministic online evaluation；以三个训练 seed 配对汇总。
- **预先定义的 pilot 晋级门：** J 在 `Hangzhou_real` 上相对 F 的 AWT 必须 3/3 配对下降、均值相对降幅至少 3%，并相对 P 在至少 2/3 对中更低；两条 flow 上 ATT、AQL、p95/max per-movement wait、spillback 均不得有超过 5% 的恶化；action entropy 与 phase frequency 不得出现单相位塌缩。
- **P2 失败解释：** 即使 continuation rank 会变化，也不能转化为可靠的部署收益；停止 P3，不对 JSCD 作性能创新声明。
- **预算：** 先做一臂 timing smoke，再将 pilot 上限控制在约 12 GPU-hours。

### P3：仅对通过 P2 的方向扩展到九个训练 seed

- **动作：** 将 F/P/J 扩展到九个预先列出的训练 seed，保持 P2 的脚本与哈希锁定；不得因中间结果添加/替换 seed。
- **统计：** 对 J-F 和 J-P 分别报告 paired Delta、95% bootstrap CI、Wilcoxon、效应方向计数；训练 seed 是唯一 `n`。
- **论文级门：** AWT 的 paired CI 完全小于零，且两个 safety flow 均通过 hard gate；否则写为无效或条件性结果，而非泛化优势。
- **预算：** 需独立批准，不能以 P2 的预算自动延展。

## 7. 必须先补齐的评测与审计

P2 前为所有臂加入并冻结：

- p95 与 max per-movement wait；
- spillback 率/次数及其计算口径；
- action entropy、各 phase frequency 与最小相位覆盖；
- 完整 run metadata：source/first/second-pass checkpoint hash、label hash、traffic file、snapshot IDs、训练 seed；
- teacher audit：每个候选的 controller identity、action sequence、transition、candidate cost 和 feasibility decision。

这些是安全约束和可复核性要求，不是第二个方法模块或新的优化目标。

## 8. 明确不做的实验

在 P0/P1 之前或 P2 失败后，以下实验不应启动：

- 继续扩大原 PRCT 的 seed、K、H=30/H=300 扫描；
- KL、ranking、temperature、margin、loss-weight 的网格搜索；
- cherry-pick checkpoint 或只报告 `Hangzhou_real` 的正例；
- 把 deterministic online seed 当作统计重复；
- CRCD、PRCAT、OOD/coreset、PPO、额外 VLM 或在线 planner 的并行叠加；
- 因为 P2 样本少而提前声称跨流量泛化。

这些做法都会掩盖“continuation policy 是否必须自洽”这一唯一主创新的因果归因。

## 9. 执行顺序与停止规则

| 顺序 | 工作 | 是否训练 | 继续条件 | 预计成本 |
|---:|---|---|---|---:|
| 1 | P0：九对 `Hangzhou_real` 重评估 | 否 | 结果归档即可 | < 1 GPUh |
| 2 | P1：32 snapshot continuation-rank 审计 | 否 | 通过 top-1/tau 门 | 约 4 GPUh |
| 3 | P2：F/P/J 三 seed pilot | 是 | 全部 pilot 与 safety 门通过 | <= 12 GPUh |
| 4 | P3：九 seed 确认 | 是 | 获得额外预算 | 另行审批 |

**最重要的停止规则：** P1 不通过，或 P2 未同时优于静态 PRCT 且满足 safety hard gate，就不再把 JSCD 作为性能方法推进；研究结论收缩为“反事实教师的 deployment-policy mismatch 是可观测诊断，但该修复未产生可靠收益”。

## 10. 当前论文定位

JSCD 的新颖性在于把多路口反事实蒸馏的 continuation policy 与联合部署 policy 对齐，而不是新增一个复杂的学生或训练目标。这个主张只有在 P1 显示排序确实改变、且 P2/P3 显示变化带来可重复且安全的部署收益后才成立。此前，应以负结果和严格的对照设计为准，不写性能优越结论。
