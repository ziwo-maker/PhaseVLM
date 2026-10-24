# Stage-2 重构报告

**日期：** 2026-08-18  
**最终判断：** `REVISE`（机制可实施，论文学术证据未闭合）。

## 最终方案快照

1. **方法论断：** 用 policy-responsive rollout 生成 Stage-2 四动作反事实教师；只干预目标路口的首个动作，之后由冻结 source policy 重规划整个网络。
2. **唯一贡献：** 修复反事实教师的 deployment consistency。
3. **学生保持不变：** 复用 `candidate_action_metrics` / `candidate_action_waits`、现有 dataset 与 Stage-2 trainer；不提出新 loss。
4. **明确边界：** 只有 source policy 有 target-domain headroom 时才预期改善；Jinan-2500 null result 必须报告。
5. **安全原则：** AWT 改善不可以 tail wait、spillback 或 action collapse 为代价。

## 本轮输出

- 初始完整方案：`refine-logs/round-0-stage2-prct-proposal.md`
- 外审原始记录：`refine-logs/round-1-stage2-prct-review.md`
- 最终方案：`refine-logs/FINAL_PROPOSAL.md`
- 补实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 运行跟踪：`refine-logs/EXPERIMENT_TRACKER.md`

## 代码可实施性

- `projection_train/policy_responsive_simexpert.py`、`cost_ranking.py`、`grouped_dataset.py`、`train_projection_bc.py` 与 `scripts/relabel_policy_responsive.py` 的 `py_compile` 已通过。
- `PYTHONPATH=. pytest -q tests/test_policy_responsive_simexpert.py tests/test_cost_ranking.py tests/test_dagger_stage2_dataset.py tests/test_prcat_relabel.py`：`14 passed`。
- 因机制、schema 与训练器已经闭环，本轮**没有新增代码**。新增学生模块会违背唯一主创新和现有负向 evidence；最有价值的工程工作是做 label/result ledger 与 frozen-H120 runner。

## 评分演化

外部 reviewer 因服务端 503 没有提供有效七维评分，故不填写伪造的数值分数。证据 gate 判断为 `REVISE`：可运行性强，贡献焦点清楚，但 novelty/因果/外部有效性尚不足。

## 主动拒绝的路线

| 路线 | 拒绝原因 |
|---|---|
| PRCAT advantage weighting + source KL | 有效 pilot 劣于未加权 PRCT，扩展会稀释主张 |
| CRCD multi-horizon reliability | 已出现 action collapse / high-K degradation |
| OOD/coreset state selection | 未稳定击败 random，不能解释高 K 退化 |
| PPO Stage 3 | canonical evaluator 不支持旧收益声明 |
| 新 VLM / topology module | 与教师语义 bottleneck 无直接因果关系 |

## 剩余弱点与下一门

1. `frozen-H120` 未与 PRCT-H120 在同一 snapshot/loss/seed 协议下完成，当前不能把结果归因给“响应语义”。
2. 只有一个主要 positive cross-network setting；Jinan-2500 是真实 boundary，不能替代第二个 positive target。
3. 24 GPUh 首轮只够三 training-seed pilot；顶会证据需要条件性扩展到九个 training seeds。
4. 直接文献差异尚未完成，当前不能使用“first”措辞。
5. `RESEARCH_BRIEF.md` 标记 KDD 2027，而 `README.md` 标记 AAAI 2027；投稿 venue 和格式需在写作前锁定。

