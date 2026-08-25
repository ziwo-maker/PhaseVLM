# Research Brief

## 项目与投稿目标

- **项目**：PhaseVLM，面向 CityFlow 交通信号控制（TSC）的拓扑感知视觉语言策略学习系统。
- **目标会议**：KDD，优先按 KDD 2027 的 research track 准备；在确定实验排期和投稿节点前，须以官方 CFP 核验具体截止日期、匿名要求与篇幅限制。
- **论文定位**：不是泛化的视觉语言模型展示，而是研究如何用可信、闭环的反事实监督，在路网与车流变化下进行安全、数据高效的 TSC 策略适应。

## Problem Statement

现有 PhaseVLM 可将单路口多车道检测器历史编码为时间序列交通图像，并以冻结的视觉语言主干输出信号相位动作；但源路网训练的策略在新路网或未见车流上适应时，离线反事实标签容易破坏多路口之间的闭环协调。尤其是把非目标路口冻结在旧轨迹上的长时域反事实 rollout，会诱导动作分布塌缩，并在较大标注预算下恶化在线等待时间。

本研究要解决的问题是：**如何构造能保留多路口闭环协调、同时对低收益标签保持保守的反事实教师，使预训练 TSC 策略能在需求/路网转移下稳定地少样本适应。**

## Context

### Current System

PhaseVLM 的现有实现和可复现实验链路包括：

- 以 30 秒窗口、每 2 秒采样的多车道状态序列（常用 15 个时间步、8 个输入通道）生成可学习的三通道交通图像；
- 使用冻结的 Qwen-VL/Qwen3-VL 视觉语言主干提取表征，并由 `VisionPrefixProjector` 将视觉特征映射为相位决策前缀；
- Stage 1 做交通图像与文字描述的对齐，Stage 2 以行为克隆和成本排序等目标学习动作策略；
- Stage 3 从部署策略收集状态，再用 CityFlow/SimExpert 式反事实评估生成适应监督，并在 canonical online evaluator 上评测。

当前最有根据的方法主线是 **Policy-Responsive Counterfactual Teacher Construction (PRCT)**：对目标路口枚举候选相位时，其他路口不冻结在采样轨迹，而是在 120 秒 CityFlow rollout 中每 30 秒依据源策略重新规划。这样得到的反事实成本比较应更接近部署时的联合闭环行为。PRCAT（基于反事实优势的保守权重和源动作正则）仅保留为可证伪的鲁棒性模块，当前不是主贡献。

### Why KDD

KDD 的数据挖掘与应用研究定位适合将贡献表述为：从结构化、时变城市交通数据中学习可迁移决策策略，并通过严谨的反事实教师构造、分布转移协议与安全指标，说明何种监督数据真正有价值。KDD 版本必须突出数据/监督构造机制、可重复的实证证据及失败模式，而不能把 VLM 主干本身作为唯一新颖性来源。

## Core Research Questions and Claims

1. **协调偏差**：冻结邻居的反事实教师是否会系统性偏离部署时的多路口闭环协调，并导致动作塌缩或高预算退化？
2. **PRCT 有效性**：让非目标路口随源策略重新规划的 PRCT，能否在相同标注预算下比 frozen-neighbor 教师更稳定地改善 AWT/ATT/AQL，并避免饥饿风险？
3. **泛化边界**：PRCT 的收益是否在需求变化和第二个路网上仍存在，还是只在源策略仍有可改善空间时出现？
4. **PRCAT（条件性）**：优势加权/源动作正则能否减少 held-out demand 的退化；若不能胜过未加权 PRCT，则作为负结果保留而不进入核心方法声明。

论文级主张必须限定为：**policy-responsive multi-intersection counterfactual evaluation 是更忠实的 TSC 适应教师构造；其收益和边界通过闭环协调偏差、动作分布和跨分布在线结果共同验证。**

## Data and Evaluation Protocol

- **主源网络**：Jinan，已使用 `anon_3_4_jinan_real.json`。
- **需求 hold-out**：Jinan-2500，使用与收集流量分离的 `anon_3_4_jinan_real_2500.json`。
- **跨网络验证**：Jinan -> Hangzhou，并在条件允许时加入反向迁移或 New York 作为第二个独立路网证据。
- **教师数据**：在部署的源策略下收集状态；收集 flow、训练/验证划分和最终 online test flow 必须隔离。所有标签元数据须记录候选动作成本、目标路口、rollout 时域、邻居策略和随机种子。
- **主对比**：zero-shot、frozen-neighbor C30、unweighted responsive-H120（PRCT）、PRCAT weight-only、PRCAT KL-only、PRCAT full；保留 ORA/RuleOnly、MaxPressure 等任务基线作为外部比较。
- **预算曲线**：至少比较 `K={0, 32, 118}`；主要结论以 9 个独立训练 seed 汇总，不以单次训练或仅 simulator seed 的差异下结论。
- **指标**：平均旅行时间（ATT）、平均队列长度（AQL）、平均等待时间（AWT）、p95/max movement wait、排队车辆数、动作熵与每类相位频率。安全结论不能只用均值 AWT，必须排除 starvation/极端等待恶化。
- **评测契约**：训练、checkpoint selection 和最终结论均须落在 canonical `CityFlowEnv` 评测链路；禁止以与最终 benchmark 行为不一致的 `TrajectoryCityFlowEnv` 指标替代最终结果。

## Existing Evidence

- responsive-H120 在原 Jinan-to-Hangzhou Phase-7 设置中，`K=118` 的 9-seed AWT 为 48.66，对比 zero-shot 的 67.75；在健康的 Jinan pt8 源策略上，in-network 9-seed AWT 改善 3.00 秒（6.8%）。
- Jinan-2500 held-out demand 上，responsive-H120 的 AWT 差异为 `-0.24 +/- 0.82`，尚不显著。因此不能声称已解决需求转移；它是后续保守机制和跨分布验证的动机。
- PRCAT full 的有效 pilot（训练 seed 2024，3 个评测 seed）为 `AWT=42.48 +/- 0.22`，优于 zero-shot 的 43.98，但劣于匹配的 unweighted responsive-H120 seed（39.44）和其 9-seed 均值（40.97 +/- 0.84）。该模块当前为负结果。
- 高预算退化并非仅由 OOD 选样引起：既有实验中 random 与 OOD 选样在 `K=118` 都明显劣化，说明必须研究教师闭环质量和训练目标，而非继续优化选样启发式。

所有数值在进入论文前必须从保存的 run metadata、checkpoint、独立 online evaluation 和训练 seed 汇总中复核。

## What We Already Tried

- **PPO Stage 3**：曾出现大幅训练内收益，但追溯后发现 `TrajectoryCityFlowEnv` 与标准 `CityFlowEnv` 行为不一致；在 canonical benchmark 上 BC 与 PPO 可产生相同行为。因此 PPO 结果不能进入论文主张。
- **冻结邻居的长时域反事实与 CRCD reliability gating**：出现动作类型塌缩和高 K 在线退化，已排除为主线。
- **OOD-guided / coreset state selection**：没有稳定击败 random selection；大 K 退化仍存在，已排除为主贡献。
- **PRCAT full loss**：当前 pilot 相比未加权 responsive teacher 为负，禁止在没有新的、预注册的可检验假设前扩展其 seed 或把它表述为收益。
- **仅以 validation accuracy 选模型**：已知与 online 性能不一致；后续选点必须使用预先定义的、隔离的 online validation 规则。

## Constraints

- **Compute**：当前可用 3 x RTX 5090（每张约 32 GB）。idea-discovery 的 pilot 总预算不超过 3 GPU-hours；确认实验与所有 principal arms 的总预算不超过 24 GPU-hours，超出时优先完成预注册主对比而非扩展探索。
- **Data**：仅使用本仓库已有的 CityFlow roadnet/traffic 数据、已收集 state snapshot 和可复现 SimExpert/RuleOnly 标签链路；不得将 final test flow 用于标签选择、损失调参或 checkpoint 选择。
- **Reproducibility**：每个 principal arm 保存命令、配置、收集 flow、标签版本、训练 seed、评测 seed、checkpoint hash 与完整 online metrics；无效 run 必须明确标记且不混入汇总。
- **Timeline**：先完成最小可信实验包和研究报告，再决定是否启动完整 KDD 写作；具体投稿日程以官方 CFP 为准。
- **Target quality bar**：要求跨需求与跨网络的独立证据、适当的不确定性报告、对教师偏差的机制性消融，以及与传统 TSC 方法的可比在线评测。

## Non-Goals

- 不主张“通用 advantage weighting”或“把 VLM 用于交通控制”本身具有新颖性。
- 不把尚未通过 canonical evaluator 的 PPO 训练收益当作证据。
- 不为得到正结果而在 final test flow 上选择标签、超参数、模型或 seed。
- 不将 PRCAT 的现有负 pilot 隐去，也不在没有增量证据时将其列作主要贡献。
- 不以单一城市、单次训练或只报告平均指标支持泛化/安全结论。

## Immediate Next Step

以本 brief 为 Phase 0 输入，先进行 KDD 定向文献调研：检索多智能体/多路口反事实评估、offline-to-online TSC adaptation、闭环 dataset/teacher construction、保守离线策略更新及分布转移评测。随后只生成能够直接检验 PRCT 协调偏差与跨分布边界的 idea candidates。

## Evidence Pointers

- `idea-stage/docs/research_contract.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_RESULTS.md`
- `review-stage/AUTO_REVIEW.md`
- `README/stage2_triple_newyork_20260808.md`
- `projection_train/run_projection_online.py`
