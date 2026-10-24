# 方案重构报告

**问题：** 以当前项目代码和实验重构一个主创新，并给出最有说服力的补实验。  
**日期：** 2026-08-18  
**最终状态：** `REVISE`。

## 重构结果

- **最终论断：** PRCT 通过使非目标路口随 source policy 重规划，修复 frozen-neighbor counterfactual 的协调不一致。
- **主贡献数量：** 1。
- **新增可训练模块：** 0。
- **部署额外成本：** 0；仅离线标签构造增加 rollout。

## 被拒绝的路线

| 路线 | 证据 | 处理 |
|---|---|---|
| CRCD reliability gate | 高 K 退化未被阻止，且 30s teacher 最好 | 删除 |
| OOD / diversity selection | 未稳定优于 random | 删除 |
| PRCAT advantage weight + KL | `42.48+-0.22` 劣于 matched unweighted `39.44` | 作为负向删除检查 |
| 更大学生/更多 loss | 不对应教师语义问题 | 不做 |

## 分数状态

未给出伪造的外部数值评分：必需的 GPT-5.6-Sol 审稿两次均因服务 `503` 失败。基于代码和结果的本地门控判断为：问题锚点、具体性、可行性和验证焦点足够；新颖性边界和跨网络外部有效性仍需补强，因此不能标为 READY。

## 最高价值补实验

1. 同一 H120、同一快照、只切换 frozen vs policy-responsive non-target controller，先在标签层再在闭环层配对比较。
2. 将 Hangzhou 主结果以九个 training seeds 和 tail safety 统计确认，而不是复用零散均值。
3. 在 headroom gate 后，复现到第二个网络；Jinan-2500 饱和流保留为零增益边界。

## 交付

- 初稿：`refine-logs/round-0-initial-proposal.md`
- 最终方案：`refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 执行追踪：`refine-logs/EXPERIMENT_TRACKER.md`

