# Stage 3：跨路网适配的文献结论与方法重构

> 日期：2026-08-10
> 状态：文献结论与方法重构建议
> 范围：PhaseVLM 从 Jinan（3x4）到 Hangzhou（4x4）的跨路网少样本适配
> 对应论文：PhaseVLM（AAAI 2027 投稿）

---

## 核心结论

现有 Stage 3 结果表明：以 Waiting Time 为重点的 RuleOnly 启发式标签，配合目标路网少样本微调，能够改善在线交通指标。但它尚不能验证完整 SimExpert 方法，也不能验证反事实 cost ranking 的有效性。

更强的下一代方法应为：**拓扑条件化、Pareto 约束的反事实优势蒸馏**。

1. 通过唯一的 CityFlow 评估器收集目标路网决策状态。
2. 对每个状态快照的四个候选相位进行多 horizon 反事实仿真，保存各动作的 ATT、AQL、AWT 和 spillback。
3. 剔除违反 AWT/spillback 约束或被 Pareto 支配的动作。
4. 将其余动作的反事实偏好以优势加权软标签蒸馏到冻结主干的 PhaseVLM。
5. 使用路网图表示条件化可训练的 prefix/TS2I adapter，并在多种拓扑与需求任务上进行元训练。

该方法以闭环交通结果监督策略，替代固定的 `queue + 2 * wait` 启发式；Queue-Wait 权衡成为显式约束，而不是手调的全局标量权重。

---

## 当前结果实际支持什么

文档中的 Jinan-to-Hangzhou 实验显示，`awt=2.0` 的 RuleOnly 重标注相较早期 `queue + 0.5 * wait` 启发式，可改善记录的在线指标。这足以说明标签质量与 Waiting Time 优先级对目标路网适配重要。

当前可辩护的结论是：

> Wait-balanced RuleOnly 行为克隆是 PhaseVLM 的一个有前景的跨路网少样本适配基线。

以下结论尚不被当前实现支持：

- 完整的 snapshot-based SimExpert 重标注能够提升迁移；
- pairwise cost ranking 或软反事实 target 导致了当前提升；
- 固定的 2:1 Wait:Queue 比例是跨路网通用最优值。

---

## 代码证据

### 当前 Stage 3 并未使用完整 SimExpert

复现实验路径启用了 `--allow_ruleonly_fallback`。`relabel_with_simexpert()` 明确指出完整 SimExpert relabeling 尚未实现，随后回退到 RuleOnly。

- [`scripts/run_stage3_cross_network_adaptation.py`](../scripts/run_stage3_cross_network_adaptation.py#L244)
- [`scripts/run_stage3_cross_network_adaptation.py`](../scripts/run_stage3_cross_network_adaptation.py#L253)

该 fallback 仅从每个相位的 queue/wait 选择单个硬标签 `expert_action`，并未为所有候选动作构造 `candidate_action_metrics` 或 `candidate_action_waits`。

- [`scripts/run_stage3_cross_network_adaptation.py`](../scripts/run_stage3_cross_network_adaptation.py#L259)

### 对 Soft Target 与 Cost Ranking 的影响

训练循环仅在至少两个候选动作具有有效 cost 时启用 soft target；同样，pairwise ranking loss 在没有有效候选对时会被 mask 掉。

- [`projection_train/train_projection_bc.py`](../projection_train/train_projection_bc.py#L561)
- [`projection_train/train_projection_bc.py`](../projection_train/train_projection_bc.py#L588)

因此，Stage 3 使用文档所述 RuleOnly 记录时，`soft_target_alpha` 和 `ranking_loss_weight` 没有反事实候选 cost 可用。这些记录上的训练实际退化为 hard behavior cloning。

现有 `SimExpertCollector` 已经具备所需核心能力：恢复快照、逐动作仿真，并输出每个动作的 ATT/AQL/AWT/spillback 指标。Stage 3 缺少的是调用并序列化这些数据的 adapter，而不是底层候选仿真器。

- [`utils/sim_expert_collector.py`](../utils/sim_expert_collector.py#L250)
- [`utils/sim_expert_collector.py`](../utils/sim_expert_collector.py#L432)

---

## 已核验的相关论文

下列论文由 OpenAlex 发现，并通过 Crossref DOI 元数据或正式会议论文集核验。优先引用正式发表版本而非预印本。

| 论文 | 顶会/顶刊 | 对 Stage 3 的直接价值 |
|---|---|---|
| [MetaLight: Value-Based Meta-Reinforcement Learning for Traffic Signal Control](https://doi.org/10.1609/aaai.v34i01.5467) | AAAI 2020 | 交通信号跨任务快速适配的直接先例，说明普通微调可升级为元适配。 |
| [IG-RL: Inductive Graph Reinforcement Learning for Massive-Scale Traffic Signal Control](https://doi.org/10.1109/tits.2021.3070835) | IEEE T-ITS, 2022 | 使用归纳图策略处理未见路网规模和结构，是拓扑条件化的直接依据。 |
| [PressLight](https://doi.org/10.1145/3292500.3330949) | KDD 2019 | Max-pressure 是强且可解释的控制基线与教师信号。 |
| [CoLight](https://doi.org/10.1145/3357384.3357902) | CIKM 2019 | 以图注意力刻画路口协作，可作为 topology adapter 的参考。 |
| [Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control](https://doi.org/10.1609/aaai.v34i04.5744) | AAAI 2020 | 给出大规模路网下共享参数与去中心化控制的经验。 |
| [CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario](https://doi.org/10.1145/3308558.3314139) | WWW 2019 | 支撑唯一 canonical observation、transition 和 evaluation contract。 |
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) | ICML 2017 | 少样本参数快速适配的标准 episodic 训练范式。 |
| [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | AISTATS 2011 | 支持在部署策略实际访问的状态上查询专家，以降低 covariate shift。 |
| [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html) | ICML 2017 | 将优化目标与安全/质量约束分开，而不是混成单一加权 reward。 |
| [IPO: Interior-Point Policy Optimization under Constraints](https://doi.org/10.1609/aaai.v34i04.5932) | AAAI 2020 | 多约束累计 cost 优化的实用参考。 |
| [Adaptive traffic signal control with actor-critic methods in a real-world traffic network with different traffic disruption events](https://doi.org/10.1016/j.trc.2017.09.020) | Transportation Research Part C, 2017 | 顶级交通期刊中关于分布变化下自适应控制的参考。 |

### 文献综合

交通信号控制文献针对 Stage 3 问题给出两条互补路线。MetaLight 解决快速适配，但需要训练时构造任务分布；IG-RL 通过归纳图表示处理拓扑变化。更强的 PhaseVLM 应结合两者：共享的拓扑条件化表征加上小规模目标域更新。

模仿学习文献支持在已部署策略访问到的状态上查询专家，当前 canonical evaluator hook 已满足这一点。缺失的是教师必须提供反事实动作质量，而不只是一个启发式硬动作标签。

约束控制文献不建议将冲突目标全部塞入固定标量 reward。这与当前观测到的 Queue-Wait 权衡直接对应：AWT 与 spillback 应作为明确约束，ATT/AQL 在可行集合内优化。

---

## 推荐方法

### 1. 多 Horizon 反事实 SimExpert 标签

对每个选择的网络级快照 `s_t` 和每个四相位动作 `a`，恢复精确 CityFlow 状态，令其他路口遵循当前部署策略，并以多个 horizon 仿真，例如 30、90、180 秒：

```text
c(s_t, a) = [ATT_30, AQL_30, AWT_30, spillback_30,
             ATT_90, AQL_90, AWT_90, spillback_90,
             ATT_180, AQL_180, AWT_180, spillback_180]
```

保存原始 metric 向量及其相对当前部署动作的增量 cost：

```text
Delta_c(s_t, a) = c(s_t, a) - c(s_t, a_base)
```

不要在存储前按单个状态进行 min-max 归一化。逐状态归一化会抹去低拥堵与高风险拥堵状态之间的严重程度差异。可以在训练 loss 内归一化，但应保留原始值以便审计。

### 2. Pareto 可行性与约束处理

对每个状态，若候选动作 `a` 满足任一条件，则将其剔除：

```text
AWT(s_t, a) > AWT(s_t, a_base) + epsilon_wait
spillback(s_t, a) > spillback(s_t, a_base) + epsilon_spillback
```

在余下动作中剔除 Pareto-dominated action：若另一个动作在所有跟踪 metric 上均不差、且至少一个 metric 严格更好，则该动作被支配。

这会改变当前 2:1 启发式的解释：

- `AWT` 变成受控约束，而非全局固定系数；
- ATT/AQL 仅在满足 Wait 与 spillback 预算的动作中优化；
- 有效权衡可以随状态与 horizon 自适应变化。

### 3. 优势加权偏好蒸馏

在可行动作上构造偏好分布：

```text
A(s_t, a) = -w^T Delta_c(s_t, a)
q(a | s_t) proportional to exp(A(s_t, a) / tau)
```

用以下目标训练 PhaseVLM 的 action logits：

```text
L = CE(q(. | s_t), pi_theta(. | s_t))
  + lambda_wait * max(0, Delta_AWT - epsilon_wait)
  + lambda_spill * max(0, Delta_spillback - epsilon_spillback)
```

在 validation flow 上以 primal-dual 或 barrier-style 调度更新 `lambda_wait` 与 `lambda_spill`。这比对所有动作对施加相同 pairwise rank loss 更合理，尤其当部分相位几乎等价、另一些相位差异显著时。

### 4. 拓扑条件化 Adapter

保持 Qwen3-VL 冻结。构造路网图，节点特征包括：

- lane 与 movement capacity；
- 当前相位和相位切换状态；
- detector history 与 TS2I embedding；
- 上下游拥堵摘要；
- movement-to-phase compatibility。

以小型 GNN 或 Graph Transformer 得到图 context，只通过可训练 prefix/LoRA adapter 注入模型。这保留当前低成本适配设置，同时使 12 路口与 16 路口网络在模型内部可区分。

### 5. 多任务元训练

只有构造源任务分布后，MAML-style adaptation 才有意义。应从多种 CityFlow 路网与系统化扰动构造 episode：

- 网格规模与路口数量；
- 车道数与转向比例；
- 需求倍率和方向性需求偏移；
- 相位时长与事故扰动。

每个 episode 以 K 个完整网络快照作为 support set，并使用 flow-disjoint query flow。部署时只更新 topology adapter 与 prefix/TS2I 参数。

### 6. 主动 SimExpert 查询

K 不应仅随机选择快照。应优先选择以下分数高的完整网络快照：

- policy entropy 或 ensemble disagreement；
- 相对源策略 embedding 的 out-of-distribution 距离；
- AWT 尾部风险和 spillback 风险；
- 时间多样性。

这使昂贵的 SimExpert 仿真集中于源策略不确定、分布偏移明显或可能造成累计延误的状态。

---

## 必需实施顺序

### P0：先让当前方法主张成立

1. 实现调用已有 snapshot-based SimExpert candidate simulator 的 Stage 3 adapter。
2. 在每条重标注记录写入 `candidate_action_metrics`、`candidate_action_waits`、`base_action` 和 `expert_action_margin`。
3. 对于计划进行 soft-target 或 ranking 训练的样本，若有效候选 cost 少于两个则直接终止。
4. 记录 candidate-cost coverage、soft-target coverage、ranking-pair coverage 及 action margin 分布。
5. 科学实验中移除 `--allow_ruleonly_fallback`；它仅可用于 smoke test。

### P1：建立因果消融

在完全相同的采集状态、源 checkpoint、目标 flow 和独立训练 seed 下运行嵌套对照：

| ID | 目标标签 | 训练目标 | 用途 |
|---|---|---|---|
| A | RuleOnly | Hard BC | 复现当前基线。 |
| B | Full SimExpert | Hard BC | 单独衡量标签质量收益。 |
| C | Full SimExpert | Soft counterfactual preference distillation | 验证全动作信息的价值。 |
| D | Full SimExpert | Preference distillation + Pareto constraints | 验证显式 AWT/spillback 可行性是否消除权衡。 |
| E | D | Topology adapter + meta-training | 验证结构条件化能否改善跨路网迁移。 |

### P2：评测协议

- collection、hyperparameter selection 和 final test 使用彼此独立的 flow；
- validation 按完整决策时间快照或按 episode 切分，绝不按单个路口记录随机切分；
- 至少报告 3 个独立 adaptation seed 与配对 simulator seed；
- 报告均值、标准差、配对差异、effect size 和正确的 95% t interval；
- 评估 ATT、AQL、AWT、travel time、spillback、adaptation time 与 SimExpert 查询成本；
- 对比 zero-shot PhaseVLM、RuleOnly adaptation、full SimExpert adaptation、target-trained upper bound、MaxPressure/PressLight，以及可行时的归纳图控制基线；
- 同时报告 Jinan-to-Hangzhou、Hangzhou-to-Jinan，以及 same-network adaptation attribution control。

不要只按 `val_acc` 选择最终 checkpoint。应使用 held-out state set 上的 counterfactual cost regret 与小型 flow-disjoint simulator validation suite 选择模型，然后锁定 checkpoint 再做 final test。

---

## 论文定位建议

当前证据支持一个范围明确的贡献：

> 使用 canonical evaluator 的冻结 VLM 交通控制器跨路网少样本适配协议，其中 RuleOnly 是强启发式基线。

只有在 P0/P1 通过后，才应主张更强的贡献：

> 拓扑条件化、Pareto 约束的反事实优势蒸馏，可以在控制 Waiting Time 与 spillback 退化的同时实现高标签效率的跨路网适配。

该叙事比将固定 Wait:Queue 权重描述为通用发现更强，也直接回应了 offline-online gap、拓扑偏移与 Queue-Wait trade-off。
