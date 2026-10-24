# Stage-2 PRCT 实验跟踪表

| Run ID | 里程碑 | 目的 | 系统 / 变量 | 数据 / split | 指标 | 优先级 | 状态 | 决策备注 |
|---|---|---|---|---|---|---|---|---|
| P001 | M0 | 结果账本 | 既有 Hangzhou/Jinan-real/Jinan-2500 results | 原始 metadata + eval summaries | hashes, actual K, source ckpt | MUST | TODO | 主文数值必须可重建 |
| P002 | M0 | snapshot 确定性 | PRCT-H120 replay x2 | 固定 16 snapshots | candidate costs/ranks/schema | MUST | TODO | 不通过则重建 teacher data |
| P003 | M1 | handler-only 教师审计 | frozen-H120 vs PRCT-H120 | 同一 clean snapshot pool | disagreement, tau, margin, entropy | MUST | TODO | 所有其他变量锁定 |
| P004 | M1 | phase/safety 诊断 | P003 teacher outputs | 路口、traffic-load 分层 | action frequency, tail-risk proxy | MUST | TODO | 不能只看 expert action |
| P005 | M2 | 训练 smoke | frozen-H120 / PRCT-H120, seed 2024 | Jinan->Hangzhou K=118 | metadata, online AWT/safety | MUST | TODO | 先校正单位 GPUh 与路径 |
| P006 | M2 | 因果 pilot | frozen-H120 / PRCT-H120, seeds 42/123 | 同 P005，flow-disjoint test | paired AWT, ATT, AQL, safety | MUST | TODO | 与 P005 合为 3 training seeds |
| P007 | M2 | zero-shot ledger | fixed source checkpoint only | 同一 held-out test flow | AWT, ATT, AQL, safety | MUST | TODO | 不是独立训练 arm |
| P008 | M3 | headroom boundary | existing Jinan-2500 matrix | collect=test, actual K=116 | all online metrics | MUST | TODO | 不重调 final flow |
| P009 | M3 | 删除证据 | existing PRCAT full pilot | matched Jinan-real seed | AWT and configuration | MUST | DONE | 已知劣于 unweighted PRCT |
| P010 | M4 | 第二目标 headroom gate | zero-shot vs feasible references | second target, fixed eval seeds | AWT, ATT, AQL | CONDITIONAL | TODO | 无 headroom 不训练 |
| P011 | M4 | 外部有效性 | frozen-H120 vs PRCT-H120 | second target K=118 | paired all metrics | CONDITIONAL | TODO | 3 seeds gate 后才扩到 9 |
| P012 | M4 | 预算 / 简化 | frozen/PRCT K=32; PRCT hard-BC deletion | only after M2 positive | paired online results | CONDITIONAL | TODO | 不产生第二贡献 |

