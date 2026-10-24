# 主论文需要补充的消融章节

生成时间：2026-08-28（修订）

本文件只整理主论文需要补写或补强的消融章节，不包含 `Backbone Scaling and Checkpoint Selection Analysis`。
当前重点是隔离视觉表示、监督预算、数据质量和训练目标的作用，并用严格的统计与数据契约避免把未受控的结果写成因果结论。



## 2. Representation and Stage-1 Alignment Ablation

**建议作为方法可解释性章节；若论文不主张视觉表示贡献，可放附录。**

当前的 `text/numeric only`、`TS2I only`、`compact cue` 和 `full` 若同时改变输入、投影器和参数量，不是干净消融。建议拆成两个实验块。

### 2.1 模态 2x2 factorial

| 视觉输入 | compact queue/wait cue |
|---|---|
| 无 | 无 |
| 无 | 有 |
| 有 | 无 |
| 有 | 有 |

要求：

- 纯 text/numeric controller 必须重新训练，不能用 `zero_prefix` 代替；
- 使用参数量匹配的 numeric-to-prefix encoder；
- 固定数据、loss、训练轮数、checkpoint 选择和 training seeds；
- 报告参数量、可训练参数量和训练计算量。

### 2.2 Stage-1 alignment deletion

固定输入模态和 Stage-2 配方，只比较：

- `aligned Stage 1 -> Stage 2`；
- `random projector -> Stage 2`；
- 可选：Stage-1 early checkpoint 或 frozen random projector。

每个配置至少使用 3 个 training seeds。该实验回答的是“Stage-1 初始化是否有用”，不能同时改变 prompt、数据集和 Stage-2 loss。

## 3. Visual State-Dependence Ablation

**建议保留，主文可放一个紧凑表，完整结果放附录。**

现有视觉干预实验支持“单个 Qwen3-VL-8B checkpoint 使用与路口状态对应的视觉信息”，但不支持“预训练 VLM 优于参数匹配的 numeric encoder”。

### 推荐对照

- `full`：正确的 TS2I image；
- `shuffled_image`：打破 image-intersection correspondence，同时尽量保持图像分布；
- `mean/constant image`：使用通道均值或真实图像均值，而不是只使用全零图像；
- `zero_prefix`：仅作为 OOD failure analysis，不能写成 retrained text-only controller。

### 必须补强的控制

- 将 batch 内 cyclic shift 改为无固定点随机 derangement，并记录 permutation seed；
- 可增加跨时间 shuffle，区分空间错配与时间相关性；
- 报告第一次决策的 open-loop action agreement，再单独报告闭环 trajectory agreement；
- 现有 4 个 online seeds 是同一 checkpoint 的 simulator seeds，不能声称跨训练 checkpoint robustness；
- 不把 `constant_image` 或 `zero_prefix` 的巨大退化解释为“视觉贡献百分比”；它们是 OOD 压力测试。

### 论文表述边界

`shuffled_image` 是最干净的 state-dependence 干预；`constant_image` 和 `zero_prefix` 只能作为更强的分布外失败模式。

## 4. Data Quality and Training Objective Ablation

**建议作为补充材料或附录。**

当前把 SimExpert、DAgger mixed、RuleOnly、hard BC、ranking、KL 和 full loss 放在同一张对照表中，会同时改变数据来源、状态分布和优化目标，无法解释差异来自哪里。应拆成两个 deletion matrix。

### 4.1 固定数据，删除 loss 组件

在同一标签池、同一 split 和同一 checkpoint 规则下比较：

1. `hard BC`；
2. `hard BC + soft target`；
3. `hard BC + soft target + ranking`；
4. `hard BC + soft target + ranking + base-policy KL`。

每一步只增加一个 loss component，并固定其权重、温度、margin 和有效样本规则。

### 4.2 固定 loss，比较数据来源

在相同样本数、相同训练 recipe 下比较：

- `SimExpert`；
- `DAgger mixed`；
- `RuleOnly`；
- source data + target data 的混合方案。

RuleOnly 通常没有完整 candidate-cost 信息，因此不能假设它能够使用 ranking 或 KL；若实际退化为 hard BC，必须在表格中明确标注，而不是写成 full-loss 对照。

### 4.3 推荐负控制

- **Same-state control：** 固定与主实验完全相同的 snapshot/state 池、样本数、train/val split、初始化和训练 recipe，只改变标签构造；至少比较真实 corrected targets 与随机/打乱后的 targets。
- **Shuffled-target control：** 在同一批状态内打乱 corrected targets，并记录 permutation seed；打乱后应尽量匹配原始 action histogram，避免把 action rebalance 与标签质量混为一谈。
- **State-selection control：** 在相同标签预算下比较主选择策略与随机选择状态；随机选择应使用相同候选池、相同 K 和独立但可复现的 selection seed。
- 报告主指标 `AWT/ATT/AQL`，以及 `trigger-state agreement`、`non-trigger introduced-error rate`、`residual override rate`、action histogram 和 label/cost agreement。
- 负控制若与真实标签臂表现接近，不能继续声称收益来自 corrected-target 质量；若显著退化，才可排除单纯 action rebalancing 或 generic label noise 的解释。

这些负控制可以排除“只是 action rebalancing 或 generic label noise 带来收益”的解释。

## 5. 统一实验与统计契约

以下要求适用于所有主文消融：

1. **统计单位：** 训练结果的主要独立单位是 training seed；同一 training seed 下的多个 simulator seeds 是对固定 checkpoint 的重复测量，先聚合到 training-seed 均值，再做跨训练 seed 推断。不能把 intersection decision、history row 或 deterministic simulator seed 当作独立训练样本。
2. **层级报告：** 对每个 `(training seed, simulator seed)` 保存完整 episode 指标；报告 simulator-seed 均值/标准差，以及跨 training seeds 的 mean +/- std、paired delta、95% CI 和方向计数。若需要利用 episode 级数据，使用 hierarchical bootstrap（先按 training seed 重采样，再在 seed 内按 simulator seed 重采样），不能把所有 episode 直接合并。
3. **配对设计：** 同一 source checkpoint、数据 split、初始化 seed、optimizer、epoch、checkpoint selection 和 evaluation protocol；同一 training seed 使用相同 simulator-seed 集合，并报告 per-training-seed paired delta。
4. **统计检验：** 预注册 AWT 为 primary metric；以 training seed 为单位使用 paired bootstrap/permutation 或 Wilcoxon，并对多个 ablation family 做 Holm 校正。
5. **Checkpoint 选择：** 不得依据最终 online test 选择 checkpoint；使用固定 final epoch 或独立 validation flow 的预注册规则。
6. **数据账本：** 保存 checkpoint hash、traffic/flow hash、snapshot IDs、label hash、loss 配置和 evaluator hash。
7. **Headroom gate：** 每个目标需求场景先报告 zero-shot 与可行 target-trained/reference baseline；若 zero-shot 已接近上限，应解释为无 headroom，而不是简单归因于训练失败。
