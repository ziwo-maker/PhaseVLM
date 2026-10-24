# Pipeline Summary

**问题：** 冻结邻居的局部反事实教师无法表示部署时的多路口协调响应。  
**最终方法论断：** PRCT 固定目标动作、让其余路口按 source policy 重规划，以 H120 全局 rollout 构造四动作教师。  
**最终结论：** `REVISE`（方法已聚焦；第二个非饱和目标网络复现尚未完成）。  
**日期：** 2026-08-18

## 最终交付物

- 方案：`refine-logs/FINAL_PROPOSAL.md`
- 审稿总结：`refine-logs/REVIEW_SUMMARY.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 实验跟踪：`refine-logs/EXPERIMENT_TRACKER.md`

## 贡献快照

- **主贡献：** policy-responsive multi-intersection counterfactual teacher。
- **可选支撑：** action collapse / tail-wait 的机制诊断。
- **明确拒绝的复杂度：** PRCAT、CRCD、OOD selection、DAgger、额外 RL、额外学生模块。

## 必须证明

1. 在 matched frozen-H120 对照下，收益来自非目标路口的策略响应而非视界或损失。
2. PRCT 在第二个有 headroom 的网络上复现，同时不牺牲 tail wait、spillback 和动作多样性。

## 首先执行

1. R001-R003：复算结果并对账标签、快照和聚合。
2. R004：生成 frozen-H120 / responsive-H120 handler-only 配对标签审计。
3. R006：先跑 Hangzhou K=118 的三 training-seed 因果 pilot。

## 主要风险

- **风险：** 现有正例可能是 Hangzhou 单例。  
  **缓解：** B3 headroom gate 后开展第二网络的 frozen-H120 对照。
- **风险：** H120 是后验选择。  
  **缓解：** frozen-H120 与 responsive-H120 同视界直接比较，H30 仅做解释性对照。
- **风险：** Jinan-2500 零增益削弱泛化。  
  **缓解：** 定义为 source-policy saturation boundary 并在论文主文报告。

## 下一动作

- 执行 R001-R003；完成后按 M0/M1 门决定是否启动 `run-experiment`。

