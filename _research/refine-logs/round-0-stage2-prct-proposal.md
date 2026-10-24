# 初始方案：将 Stage 2 重构为部署一致的反事实教师蒸馏

**日期：** 2026-08-18  
**当前结论：** `REVISE`。代码路径可运行，但尚未完成决定性配对证据。

## Problem Anchor

- **底线问题：** PhaseVLM 在少样本适应时把每个多路口状态压成一个反事实“最优动作”标签。传统 SimExpert 在评估目标路口候选相位时冻结其他路口在快照相位，而部署中邻居会在后续控制点随策略响应。该教师-部署语义错配已与高标注预算的动作塌缩和在线退化同时出现。
- **必须解决的瓶颈：** 不是增加学生容量，而是在不增加部署时计算的前提下，令 Stage 2 的离线监督代表多路口闭环中的真实响应。
- **非目标：** 不提出新 VLM、PPO/离线 RL、主动选样、CRCD reliability gate、PRCAT advantage/KL、通用偏好损失或无条件域迁移。
- **约束：** 保持冻结 Qwen-VL 和现有四相位动作空间；仅用已有 CityFlow snapshots、flows 和 canonical `CityFlowEnv`；principal arms 首轮不超过 24 GPUh；训练/收集/test flow 隔离。
- **成功条件：** 在同 snapshot、同 horizon、同学生训练协议下，policy-responsive 教师相对 frozen-neighbor 教师出现可解释的标签差异，并在有 source-policy headroom 的目标上同时改善 AWT 与尾部安全指标。

## 技术缺口

现有 Stage 2 的 hard BC、soft candidate cost 和 pairwise ranking 都只决定如何消费一个教师；它们不能修复教师本身对邻居响应的错误干预。项目内 New York 消融还显示，单独加 cost ranking/KL 可改善 queue 而恶化 wait，因而不应把训练损失包装为新的主贡献。最小干预是只替换反事实 rollout 中非目标路口的控制来源，并复用既有数据 schema 和训练器。

## 方法论断

**Stage 2 蒸馏在每个 canonical snapshot 中只强制目标路口的首个候选动作；随后让所有路口按冻结 source policy 重新决策，用 H=120 的四动作全局反事实成本构成训练监督，而不再克隆 frozen-neighbor rollout 的标签。**

给定状态 `s`、目标路口 `i`、候选动作 `a` 与 source policy `pi_b`：

`u_i(0)=a; u_j(0)=pi_b(h_j(0)) (j != i); u_j(t)=pi_b(h_j(t)) (t>0)`。

候选成本 `C(s,i,a;pi_b)` 是同一 snapshot、同一 yellow transition、同一候选动作集合下 H=120 CityFlow rollout 的全局 ATT/AQL/AWT/spillback 代理成本。四个 `C` 值经过已有 Pareto/constraint filter 与归一化后，沿用 `expert_action`、`candidate_action_waits` 和 `candidate_action_metrics` 输入现有 Stage-2 trainer。

## 唯一贡献与复杂度预算

- **唯一主创新：** deployment-consistent multi-intersection counterfactual teacher construction。
- **复用而非贡献：** 冻结 Qwen-VL、TS-to-image projector、VisionPrefixProjector、soft target、pairwise ranking、CityFlow 和 source policy。
- **新可训练模块：** 无。
- **明确删除：** PRCAT、CRCD、OOD/coreset selection、PPO、test-time planning、memory 或 LLM critic。
- **推理时成本：** 无变化；反事实 rollout 仅发生于离线标注。

## 实现落点

`projection_train/policy_responsive_simexpert.py` 已包含该机制：`_simulate_candidate` 在首个 interval 用 `force_target_action` 干预目标路口，在其后每个 30 秒控制点通过同一个 source checkpoint 为网络重新出动作；`RollingSequenceBank` 从 canonical history 推进 15x8 输入序列。输出保持 SimExpert schema，因此 `GroupedTrafficActionDataset` 与 `train_projection_bc.py` 不需要 PRCT 专用分支。临界比较必须固定 mixed-loss 配置；hard-BC only 仅作删除消融，不形成第二创新。

## 可证伪主张

1. **C1，机制：** 在 H=120 与所有其他变量相同的条件下，frozen-neighbor 与 policy-responsive teacher 的四动作排名、winner margin 和 action frequency 有系统差异；frozen 分支的偏置与后续 tail wait/phase collapse 相关。
2. **C2，结果：** 在有 headroom 的 source-to-target 任务，PRCT teacher 训练出的 Stage-2 student 相对 frozen-H120 降低 AWT，并且 ATT/AQL、p95/max movement wait、spillback 和 action entropy 不越过预注册安全界限。
3. **C3，边界：** 当 source policy 在目标流没有可利用余量时，PRCT 不应凭空改善。Jinan-2500 的零增益必须保留而不能重调至正结果。

## 初始验证草图

1. 对同一批 snapshot 做 frozen-H120 / PRCT-H120 label audit，先验证干预语义是否真实改变教师。
2. 在 Jinan -> Hangzhou、K=118 上固定学生训练配方，仅替换教师；用 3 个独立训练 seed 做 24 GPUh 内 gate。
3. 仅在 gate 通过后扩展到 K=32、9 training seeds 与第二个有 headroom 的目标网络。

