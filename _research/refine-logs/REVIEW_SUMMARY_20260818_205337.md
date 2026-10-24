# 审稿与收敛总结

**问题：** Stage 2 frozen-neighbor counterfactual teacher 与部署时的闭环协调不一致。  
**初始路线：** 将 hard BC、cost ranking、PRCAT、CRCD、OOD selection 与后续自适应并列推进。  
**日期：** 2026-08-18  
**有效外部审稿回合：** 0  
**最终状态：** `REVISE`。

## Problem Anchor

以不增加部署成本的方式，消除 frozen-neighbor 反事实标签的多路口协调偏差，并以 matched teacher control、独立 training seeds 与闭环安全指标检验它。

## 本轮事实核验

| 维度 | 结论 | 证据 | 仍需完成 |
|---|---|---|---|
| Problem Fidelity | 强 | 改动只位于反事实 rollout 的后续控制者，直接对应 observed high-K degradation | frozen-H120 配对因果证据 |
| Method Specificity | 强 | canonical snapshots、15x8 histories、target override、replanning、same-schema handoff 均已有实现 | 真实标签 audit ledger |
| Contribution Quality | 中等偏强 | 一条教师干预语义主线；不新增可训练模块 | 直接先例文献核验 |
| Frontier Leverage | 有意克制 | 冻结 VLM 是应用 backbone，非 novelty | 不得强加 LLM/RL 模块 |
| Feasibility | 强 | 五个核心 Python 文件静态编译；相关 14 个测试通过 | label rollout GPU-hours 实测校准 |
| Validation Focus | 强 | 先 label mechanism，再 matched online causal gate | 预算内先只做 3 training seeds |
| Venue Readiness | 尚不足 | 已有正例与 zero-gain boundary | 9 seeds、第二 non-saturated target、literature novelty |

## 回合记录

| 回合 | 主要问题 | 收敛操作 | 状态 | 剩余风险 |
|---|---|---|---|---|
| 0 | Stage 2 含多个可竞争叙事 | 将唯一贡献固定为 policy-responsive teacher；ranking/KL/PRCAT 降为固定配方或删除 | 完成 | 方法需 matched control |
| 1 | 独立方法审查 | 调用 `gpt-5.6-sol` reviewer | 未完成，服务端 503 | 不可把本地判断当外审评分 |

## 锚定与简化检查

- **保留的锚点：** 多路口闭环协调语义，而非“更大的 VLM”或“更聪明的 loss”。
- **删除的复杂度：** PRCAT、CRCD、OOD selection、PPO、online planning、memory/critic。
- **不采纳的漂移建议：** 用新 RL reward、LLM planner 或更多 data selection 替换教师质量问题。
- **对 Stage 2 的重构：** 它成为现有学生对 deployment-consistent teacher 的蒸馏阶段；不会同时声称新的架构、loss 和适应算法。

## 外部审稿与文献状态

本轮外审请求失败：`503 Service Unavailable`，请求 ID `7d568ac6-691a-4a58-8759-297897aebd02`。Semantic Scholar 查询返回 `429`；OpenAlex 返回的是泛化 survey，未提供可用于确认或否定 PRCT 的直接先例。因此最终没有编造外部评分或文献新颖性结论。

## 结论

方法已足够具体，可进入**证据 gate**；但不能进入论文写作 readiness。只要 frozen-H120 与 PRCT-H120 的 handler-only 比较不能同时显示教师差异和安全的 online 改善，就应把故事降级为数据/诊断发现。

