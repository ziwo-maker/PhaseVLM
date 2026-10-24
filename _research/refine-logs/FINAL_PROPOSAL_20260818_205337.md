# 研究方案：Stage-2 策略响应式反事实教师蒸馏（PRCT）

**状态：** `REVISE`。这是一个可运行、单创新点的重构方案；尚不能以现有结果声称跨网络普适性。  
**主创新：** 让 Stage 2 的反事实教师遵守部署时的多路口闭环响应语义。

## Problem Anchor

PhaseVLM 的 Stage 2 以 SimExpert 反事实标签微调冻结视觉语言策略。传统标签在一个 CityFlow snapshot 中强制目标路口的候选相位，却把其余路口冻结在快照相位。该干预并不等价于部署：在真实闭环控制中，邻居会在后续 30 秒决策点观察新的交通状态并随 source policy 重规划。已观察到的后果是高标注预算的动作分布塌缩和在线等待时间恶化。

要解决的问题是：**如何在不新增学生模型、在线搜索或部署成本的前提下，构造与多路口闭环相一致的反事实 Stage-2 教师。**

不研究新 VLM、PPO/离线 RL、OOD/coreset selection、CRCD reliability gate、PRCAT weighting/KL 或通用偏好学习。它们不是本论文贡献，且项目中已有负向或不稳定证据。

## 技术缺口

Stage 2 的 hard BC、cost soft target 和 pairwise ranking 只规定学生如何拟合标签；它们无法补偿标签 rollout 中把邻居冻结的因果语义错误。把 ranking、KL、权重或新 backbone 叠加进去会模糊归因，且已有 New York 结果显示 cost ranking 可能以 queue 改善换取 wait 恶化。正确的最小修复是改变**谁在反事实分支的后续控制点行动**，其余学生、损失和测试协议一律不动。

## 方法论断

**PRCT 在同一 canonical CityFlow snapshot 中只强制目标路口的第一个候选动作；从随后的控制点起，所有路口均由冻结的 source policy 基于分支自身的历史重新决策。H=120 的四动作全局成本构成 Stage-2 的训练教师。**

对 snapshot `s`、目标路口 `i`、候选相位 `a` 与 source policy `pi_b`，定义：

```text
u_i(0) = a
u_j(0) = pi_b(h_j(0)), j != i
u_j(t) = pi_b(h_j(t)), t in {30, 60, 90, ...}
C_PRCT(s, i, a; pi_b) = aggregate_{tau=1..120} global_cost(x_tau)
```

`global_cost` 使用实现中已有的 ATT、AQL、AWT 与 spillback 代理。每个候选分支从相同 snapshot 恢复，保留相同的 yellow transition、候选集合、Pareto/constraint filter、cost weights 和 random seed。target 只被干预首个控制区间，之后也恢复为 `pi_b`，避免把“持续锁死 target”混入方法。

```text
canonical source-policy snapshot and 15x8 histories
        |
        +-- force one of four phases at target i for one interval
        |       +-- let source policy re-plan the rest of the network
        |       +-- H=120 global rollout cost
        |
        +-- four costs + existing feasibility filters
                    |
                    +-- unchanged Stage-2 supervised trainer
                    |
                    +-- standalone PhaseVLM at inference
```

## 代码与训练接口

- `PolicyResponsiveSimExpertRelabeler._simulate_candidate` 恢复 snapshot，利用 `RollingSequenceBank` 保持与 online policy 相同的序列输入；首个 interval 用 `force_target_action`，随后通过 `ProjectionPolicyRunner.predict_from_sequences` 对网络重新出动作。
- 教师仍写入 `expert_action`、`candidate_action_waits`、`candidate_action_metrics`、base action 与 candidate metrics。`GroupedTrafficActionDataset` 和 `train_projection_bc.py` 已可直接消费这些字段。
- 学生只更新已有的 TS-to-image/projector 参数；Qwen-VL backbone 保持冻结。frozen-H120 与 PRCT-H120 必须固定 data split、snapshot pool、source checkpoint、loss、optimizer、epochs、checkpoint selection 和 training seeds。
- **不是新 loss：** 论文主臂沿用项目已验证的 student loss，所有 teacher 对照使用完全相同的配置。hard-BC only 是必要的删除检查，不能写成第二机制。
- 离线标注产生额外 rollout 成本；部署与推理时没有新 rollout、模型或延迟。

## 贡献边界

| 内容 | 论文定位 |
|---|---|
| policy-responsive multi-intersection counterfactual teacher | 唯一主创新 |
| 冻结 VLM 和 prefix policy | 复用 backbone |
| cost soft target / pairwise ranking | 固定 student recipe，不主张创新 |
| PRCAT weights/KL | 已有负向模块，作为删除证据 |
| CRCD、OOD selection、PPO | 排除，不扩展 |

这不是“首个反事实 TSC”或“任何域迁移均有效”的声明。它只声明：在部署策略具有可改善空间时，frozen-neighbor counterfactual 不是一个保留联合响应的教师，而 PRCT 是更忠实、无部署成本的替代。

## 现有证据与诚实边界

- Jinan -> Hangzhou 的已有 responsive-H120 `K=118` 九训练 seed 聚合 AWT 为 `48.88 +- 2.76`，zero-shot 为 `67.75`；历史 frozen-c30 为 `83.24`，frozen-H300 为 `236`。这支持继续研究，但 frozen-H120 matched control 尚缺。
- 健康 Jinan-real source 的 in-network responsive-H120 `K=118` 为 `40.97 +- 0.84`，zero-shot 为 `43.98`，说明收益不完全依赖跨网络命名。
- Jinan-2500 same-flow 的 responsive-H120 `K=118` 为 `41.47 +- 0.98`，与 zero-shot `41.47` 持平。这是 source-policy headroom 的边界，不是应被调参掩盖的失败。
- PRCAT full 的有效 pilot `42.48 +- 0.22` 劣于匹配 unweighted PRCT seed 的 `39.44`，因此不扩展其 seeds，也不进入标题、摘要或贡献。

上述数字均需在进入论文前由原始 run metadata、checkpoint hash、独立 online evaluation 和正确的 training-seed 聚合复核。

## 可证伪预测

1. **教师机制：** 同 snapshot、H=120、transition、cost 与 source policy 下，frozen-H120 和 PRCT-H120 应产生显著的 action/rank disagreement；若差异接近零，则“协调语义修复”不成立。
2. **因果结果：** 在 Jinan -> Hangzhou 等有 headroom 的目标，PRCT-H120 相对 frozen-H120 应改善预注册 AWT，且 p95/max movement wait、spillback 不恶化超过 5%，action entropy 不向单相位塌缩。
3. **边界：** 在 Jinan-2500 等无 headroom 流量，PRCT 预期为中性；这限制而非削弱有条件主张。

## 风险与缓解

- **最大风险：** 正例目前没有同 horizon frozen 对照，且第二个非饱和目标网络未完成。先跑 label audit 和 paired frozen-H120，不通过则降级为教师偏差诊断。
- **文献风险：** Semantic Scholar 本轮 API 限流，OpenAlex 的两条窄查询未返回直接的多路口反事实教师先例。该事实不证明新颖性；投稿前必须人工完成 10 篇直接相关 work 的 citation/方法核对，并把 claim 改成与最近工作可验证的差异。
- **统计风险：** 24 GPUh 内只能得出三 training-seed pilot；任何论文级结论需在预算获批后补足九个独立 training seeds。
- **安全风险：** 平均 AWT 的正向结果不足够；tail wait、spillback、action entropy 是硬门，而非附录装饰。

