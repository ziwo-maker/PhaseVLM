# PRCT 实验跟踪表

| Run ID | 里程碑 | 目的 | 系统 / 变量 | 数据 / split | 指标 | 优先级 | 状态 | 决策备注 |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 复算主结果账本 | 既有 Hangzhou / Jinan-real / Jinan-2500 聚合 | 原始 eval summaries | mean, std, CI | MUST | TODO | 解决 `48.66` 单 seed 与 `48.88` n=9 混用 |
| R002 | M0 | 数据完整性 | 所有主臂 label JSONL + snapshot manifest | clean snapshots | SHA256, schema, actual K | MUST | TODO | Jinan-2500 必须记录 K=116 |
| R003 | M0 | 确定性 smoke | PRCT H120 重放两次 | 16 snapshots | labels/costs/ranks | MUST | TODO | 不一致则修复 before training |
| R004 | M1 | handler-only 标签差异 | frozen-H120 vs PRCT-H120 | Hangzhou same snapshots | disagreement, tau, entropy | MUST | TODO | 其余配置完全锁定 |
| R005 | M1 | 视界对照 | responsive-H30 vs H120 | 同上 | label/rank shift | SHOULD | TODO | 仅解释 H120 作用 |
| R006 | M2 | 主实验 pilot | zero, frozen-H120, PRCT-H120 | Jinan->Hangzhou K=118, 3 train seeds | AWT, ATT, AQL, safety | MUST | TODO | handler-only paired design |
| R007 | M2 | 预算交互 | frozen-H120, PRCT-H120 | Jinan->Hangzhou K=32, 3 train seeds | 同 R006 | MUST | TODO | 排除只在 K=118 有效 |
| R008 | M3 | 主实验确认 | R006/R007 补齐 | 9 train seeds | paired CI/permutation | MUST | TODO | 过门才跨网络 |
| R009 | M3 | 安全分析 | R008 checkpoints | per movement | p95/max wait, spillback, entropy | MUST | TODO | 平均 AWT 无法单独过门 |
| R010 | M4 | Headroom gate | 候选第二网络 zero-shot vs refs | 3 fixed eval seeds | AWT, ATT, AQL | MUST | TODO | 饱和则换候选，不训 PRCT |
| R011 | M4 | 外部 pilot | frozen-H120 vs PRCT-H120 | second network, K=118, 5 train seeds | 同 R006 | MUST | TODO | 五 seed 一致才补到九 |
| R012 | M5 | 边界复核 | existing Jinan-2500 full matrix | collect=test, K=32/116 | all online metrics | MUST | TODO | 明确零增益而非失败调参 |
| R013 | M5 | 删除检查 | existing PRCAT full pilot | Jinan-real K=118 | matched AWT | MUST | DONE | `42.48` 劣于 unweighted `39.44`；不扩种子 |

