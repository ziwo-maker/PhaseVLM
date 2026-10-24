# Stage-2 PRCT 补实验计划

**问题：** frozen-neighbor counterfactual 是否因破坏多路口闭环响应而产生有害 Stage-2 教师？  
**方法论断：** PRCT 只改变离线 rollout 的后续控制者，令四动作教师与部署 source-policy 响应一致。  
**日期：** 2026-08-18  
**预算策略：** 先完成不超过 24 GPUh 的因果 gate；任何九 seed 或第二网络结论均需 gate 通过与额外预算批准。

## Claim Map

| Claim | 为什么重要 | 最低可信证据 | 实验块 |
|---|---|---|---|
| C1：教师修复了协调语义 | 唯一创新必须先证明机制不同 | 同 H=120 同 snapshot 的 label/rank difference，且可重复 | B0, B1 |
| C2：收益来自响应语义，不是更长 horizon 或 loss | 排除最直接替代解释 | frozen-H120 与 PRCT-H120 的 paired online comparison | B2 |
| C3：收益有诚实边界 | 不能 cherry-pick positive city | Jinan-2500 null audit + headroom preregistration | B3 |

**必须排除的反主张：** 改善只来自 horizon、K、loss、checkpoint/seed 偶然性，或以平均 AWT 交换 starvation、spillback、action collapse。

## 主文故事与删减原则

- **主文必须有：** B1 说明 handler-only 的教师差异；B2 给出 frozen-H120 与 PRCT-H120 的匹配闭环结果；B3 保留 null boundary。
- **附录：** PRCAT negative pilot、historical frozen-c30/H300、H30 sensitivity、完整 phase frequency、snapshot determinism、所有配置 hashes。
- **当前不跑：** PRCAT 扩 seed、CRCD 阈值搜索、OOD selector、loss grid、更多 VLM backbone。
- **frontier primitive：** 无。冻结 Qwen-VL 是 deployed policy backbone，PRCT 的贡献不依赖 LLM/VLM/RL 的新能力；不设置伪“frontier necessity”实验。

## 实验块

### B0：数据与评测契约审计

- **主张：** 主比较仅有 teacher handler 一项变化。
- **设置：** 用 `scripts/audit_prct_hangzhou.py` 建立 source checkpoint hash、collection/test flow、snapshot manifest、有效 K、H、cost weights、transition、student loss、training/eval seeds 的账本；每条 label 必须有四个候选成本和可解析 snapshot。
- **校验：** 对 16 个 snapshots，PRCT 重放两次；运行 `scripts/verify_simexpert_determinism.py`。训练、checkpoint selection、final test 只使用 canonical `CityFlowEnv`。
- **成功门：** 输入 hash、schema、actual K 完全可追溯；replay 在预注册 tolerance 内一致。否则停止所有训练。
- **成本：** 0 GPUh，约 0.5 engineer-day。
- **论文位置：** appendix reproducibility table。

### B1：同 horizon、同 snapshot 的教师语义审计

- **主张：** PRCT 的关键变量确实是 non-target policy response，而不是 H=120 本身。
- **设置：** 在 Jinan -> Hangzhou 的同一 16 个 smoke snapshots 后扩展到固定 clean pool。生成 `frozen-H120` 与 `PRCT-H120`；固定 source checkpoint、candidate actions、yellow transition、global cost、Pareto/constraint filter、random seed 与 H=120。仅切换后续非目标路口的 controller。
- **指标：** expert-action disagreement、四动作 Spearman/Kendall correlation、winner margin、Pareto feasibility、action frequency/entropy；按路口与 traffic-load 分层。检查 frozen 分支的 through-phase 占比是否与 subsequent tail wait 相关。
- **成功门：** handler-only 变化必须产生非平凡且可解释的 rank/action shift；若无差异，停止 PRCT 方法主张。该块只说明教师不同，不能替代在线结果。
- **成本：** 至多 6 GPUh（先 16 snapshots 估算单位成本，超过预算即缩小 pool 并记录）。
- **论文位置：** 图 2：rank disagreement 与 phase-frequency/安全诊断。

### B2：24 GPUh 内的预注册因果 gate

- **主张：** 在高标注预算下，PRCT 避免 frozen-H120 的闭环退化。
- **设置：** Jinan source -> Hangzhou target，flow-disjoint collection/test，固定 `K=118` clean snapshots。比较 `zero-shot`（只评测，复用确定义 checkpoint）、`frozen-H120`、`PRCT-H120`。后两者用相同的 snapshots、canonical sequences、student loss、optimizer、epochs、checkpoint criterion 和独立 training seeds `{2024, 42, 123}`。每个 checkpoint 以同一组固定 CityFlow evaluation seeds 测量；统计单位只计 training seed。
- **指标：** 预注册 primary 为 AWT；secondary ATT/AQL；安全为 p95/max per-movement wait、spillback、phase entropy/frequency。报告每 seed 的 paired `Delta = PRCT - frozen`，不把 evaluator seeds 当作独立训练重复。
- **通过门：** 三个 paired training seeds 的 AWT delta 均小于零，且聚合改善大于历史 run-to-run 噪声；所有 safety 指标相对 frozen 不恶化超过 5%，entropy 不向单一相位集中。任一 safety 失败或方向不一致则停止扩种子，并只报告诊断结果。
- **成本：** B1 后最多 18 GPUh。先跑每 teacher 一个 seed 计时并验证 metadata，再以 3 张 GPU 各承担独立 seed；每 worker 独立 output/log/result 路径。总 B0-B2 不超过 24 GPUh。
- **论文位置：** 主表 1 与 paired-slope 图。该结果是**pilot gate**，不是九 seed 论文结论。

### B3：边界与简化检查（不新训练）

- **主张：** 方法不被 Jinan-2500 的 null result 反证为“调参失败”，而是受 source-policy headroom 限制。
- **设置：** 复核现有 same-flow Jinan-2500 9-seed matrix：clean snapshot IDs、actual `K=116`、zero-shot、frozen-c30、responsive-H30、PRCT-H120。附录呈现 PRCAT full 的负向 matched pilot。
- **成功门：** 保持现有零增益结论；不得依据 final test flow 调参或挑 seed。学生 loss deletion（hard BC only）只在 B2 gate 通过后用同样标签追加，不提前引入第二变量。
- **成本：** 0 GPUh，约 0.5 engineer-day。
- **论文位置：** 主文 boundary panel；loss/PRCAT 为 appendix deletion table。

### B4：条件性确认，不能预支主张

- **触发条件：** B0-B2 全过门且额外算力获批。
- **第一步：** 用 3 个固定 eval seeds 只筛选已有 source checkpoint 中有 target headroom 的第二个方向；若 zero-shot 已接近可行参考则不训练 PRCT。
- **比较：** frozen-H120 vs PRCT-H120，K=118、相同 snapshot pool 和 student recipe。先 3 training seeds，再补至 9 个独立 training seeds；必要时在 B2 positive 后加入 K=32 预算对照。
- **最终论文门：** 两个有 headroom 的目标均完成 nine-training-seed 配对汇总，配对 CI/permutation 与所有 safety guardrail 通过。否则论文只主张“coordinated teacher bias diagnosis”，不主张通用 adaptation method。

## 运行顺序与决策门

| 里程碑 | 目标 | 必跑运行 | Stop / Go | 预算 |
|---|---|---|---|---|
| M0 | 锁定证据对象 | B0 ledger + 16 snapshot replay | hash/schema/determinism 失败即停 | 0 GPUh |
| M1 | 验证教师差异 | B1 frozen-H120 vs PRCT-H120 | rank/action 无差异即停 | <=6 GPUh |
| M2 | 测试因果收益 | B2 single-seed smoke 后 3 paired seeds | AWT 方向或安全失败即停 | <=18 GPUh |
| M3 | 保留边界 | B3 Jinan-2500 + PRCAT deletion audit | 只报告可复核的 null/negative | 0 GPUh |
| M4 | 论文确认 | B4 second target + 9 seed expansion | 需 M0-M3 全过门和新增预算 | 额外申请 |

## 风险与缓解

- **教师不是因果变量：** B1/B2 同 horizon、同 snapshots、同 student recipe，避免把 horizon/loss 误当主效应。
- **三个 seed 过弱：** M2 只作 allocation gate；不在摘要或结论写显著性/泛化，直到 M4。
- **坏 snapshots/非确定性：** M0 检验有任何失败就重建数据，不静默过滤。
- **安全被均值掩盖：** 5% tail/spillback 硬限制；违反即使 AWT 改善也不通过。
- **目标已饱和：** 先做 headroom gate；Jinan-2500 只作为边界，不能反复换权重。

## 最终清单

- [ ] B0 的 manifest、hash 与 canonical evaluator 对齐
- [ ] B1 证明 frozen-H120 与 PRCT-H120 的 handler-only 差异
- [ ] B2 的三个 paired training seeds 和所有 safety 指标完成
- [ ] B3 的 zero-gain / PRCAT deletion 复核完成
- [ ] M4 在额外预算下完成前，不作跨网络普适性或论文 ready 声明

