# PRCT 实验结果与主创新结论

**日期：** 2026-08-18  
**研究对象：** Policy-Responsive Counterfactual Teacher Construction (PRCT)  
**结论状态：** 条件性推荐为唯一主创新；尚不满足“无条件泛化方法”的论文主张门槛。

---

## 1. 一页结论

PRCT 的核心思想是：枚举目标路口候选相位时，不再将所有其他路口冻结在 snapshot 的相位，而是在后续控制点让它们按同一冻结 source policy 重新决策。该改动只发生在**离线教师构造**，不增加学生可训练模块，也不增加部署时开销。

现有结果支持以下条件性结论：

> 当 source policy 在目标设置仍有改善余量时，policy-responsive H120 counterfactual teacher 能避免 frozen-neighbor teacher 的高标签预算退化，并稳定降低 closed-loop AWT；当 source policy 在目标流上已接近饱和时，PRCT 不产生额外收益，但高预算下仍是最不易退化的臂之一。

因此，建议保留 **PRCT 作为论文唯一主创新**，但论文只能主张“修复多路口教师的协调语义”，不能主张“任何网络、任意流量下均可提升 few-shot adaptation”。

---

## 2. 统计与可比性规则

### 2.1 统计单位

- 所有 `mean +- std` 中的主要统计单位是**独立训练种子**，不是 CityFlow evaluation seed。
- CityFlow/SimExpert 重放在已修复确定性问题后可重复；同一 checkpoint 的多个 evaluation seed 不能视为独立训练重复。
- 若某个臂只有 1 或 3 个训练种子，只能作为 pilot / 方向性证据，不能作为最终的泛化结论。

### 2.2 不可直接混合的实验

以下结果来自不同 source checkpoint、网络、`max_length` 或流量，不能将其 AWT 均值直接合并：

1. Jinan -> Hangzhou：pt32/hd256，16 路口跨网络迁移。
2. Jinan real in-network：pt8/hd64/ml416，12 路口。
3. Jinan-2500 same-flow：pt8/hd64/ml416，12 路口，collection flow 与 test flow 相同。
4. PRCAT pilot：Jinan pt8/hd64、`max_length=64` 的单训练种子诊断，不能与 ml416 的完整矩阵合并。

### 2.3 指标

- **主指标：** AWT（average waiting time）。
- **次指标：** ATT（average travel time）、AQL（average queue length）。
- **安全指标：** p95/max per-movement wait、spillback、phase/action entropy 与各相位频率。已有多数历史结果主要报告 ATT/AQL/AWT；在下一轮因果验证中，安全指标必须补齐。

---

## 3. PRCT 主结果

### 3.1 Jinan -> Hangzhou：高预算退化被 PRCT H120 逆转

**设置：** Jinan Stage-2 pt32/hd256 source policy，Hangzhou target；collection flow 与 held-out test flow 分离；120 个快照中 118 个可用；H120 使用同一 source checkpoint、同一 mixed loss（hard CE + soft target alpha=0.3 + ranking weight=0.05）。

| 方法 | K | 训练种子 n | ATT | AQL | AWT | 相对 zero-shot 的 AWT 变化 |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot | - | - | 335.63 | 199.32 | 67.75 | - |
| frozen c30 | 8 | 3 | 327.66 +- 3.19 | 188.58 +- 5.09 | 61.81 +- 4.49 | -5.94 |
| frozen c30 | 32 | 3 | 332.90 +- 1.75 | 198.50 +- 4.69 | 80.39 +- 8.11 | +12.64 |
| frozen c30 | 118 | 3 | 335.37 +- 1.72 | 201.95 +- 3.64 | 83.24 +- 14.13 | +15.49 |
| responsive H30 | 32 | 9 | 327.45 +- 2.02 | 188.40 +- 3.67 | 66.23 +- 6.70 | -1.52 |
| responsive H30 | 118 | 3 | 331.51 +- 0.75 | 194.37 +- 1.10 | 76.10 +- 3.29 | +8.35 |
| **PRCT responsive H120** | **32** | **9** | **325.89 +- 3.91** | **184.31 +- 6.93** | **57.93 +- 4.57** | **-9.82** |
| **PRCT responsive H120** | **118** | **9** | **326.64 +- 3.44** | **182.24 +- 4.77** | **48.88 +- 2.76** | **-18.87** |

**观察：**

- Frozen c30 从 K=8 到 K=118 出现高预算退化：AWT `61.81 -> 83.24`，并伴随明显方差扩大。
- PRCT H120 K=118 为当前该设置最佳臂：相对 zero-shot AWT 降低 `18.87`（约 `27.9%`），相对 frozen c30 K=118 降低 `34.36`（约 `41.3%`）。
- PRCT H120 K=118 相对 H120 K=32 继续降低 AWT `9.05`，同时 AQL 从 `184.31` 降至 `182.24`；ATT 与 K=32 基本持平。
- H30 不足以解决高 K 问题：responsive H30 K=118 仍差于 zero-shot。这表明“允许邻居响应”与“响应作用的时间长度”都重要，H120 是当前实证上有效的工作点。

**解释：**

冻结邻居教师会把局部放行行动放在不可能持续的周边相位环境中。长视界时该偏差积累为 through-phase 偏置和高预算退化。PRCT 让非目标路口根据 rollout 内不断更新的状态重新规划，避免教师把局部短期收益错误排序为网络级长期收益。

**该结果能支持：** PRCT 在该跨网络迁移设置中显著优于 zero-shot，并强烈表明高 K 退化不是“标签数量必然有害”。

**该结果不能单独支持：** 收益完全由 policy response 而非 H120 视界造成，因为当前没有完全匹配的 frozen-H120 vs responsive-H120 九种子对照。

### 3.2 Jinan real in-network：小而稳定的正收益

**设置：** Jinan pt8/hd64/ml416 source policy，Jinan real flow；9 个独立训练种子；H120，global cost，mixed loss。

| 方法 | K | 训练种子 n | ATT | AQL | AWT | 相对 zero-shot 的 AWT 变化 |
|---|---:|---:|---:|---:|---:|---:|
| zero-shot | - | - | 274.07 | 165.98 | 43.42 | - |
| PRCT responsive H120 | 32 | 9 | 275.13 +- 1.91 | 166.82 +- 3.06 | 43.18 +- 0.90 | -0.24 |
| **PRCT responsive H120** | **118** | **9** | 274.91 +- 2.29 | 166.68 +- 3.72 | **41.59 +- 0.76** | **-1.84 (4.2%)** |

**观察：**

- K=118 的九个训练种子均优于 zero-shot；记录中的 sign test 为 `p about 0.004`。
- K=32 仅 5/9 种子优于 zero-shot，均值 `-0.24` 处于噪声内。
- ATT/AQL 的变化在约 `+-1` 范围内，可靠信号集中在 AWT。

**解释与含义：**

in-network source policy 已较强，剩余改善空间比跨网络 Hangzhou 小很多，因此 `4.2%` 的 AWT 降幅是合理但有限的正结果。它证明 PRCT 不仅在跨网络目标有效，但不支持对 ATT/AQL 的全面改进主张。

### 3.3 Jinan-2500 same-flow：明确的零收益边界

**设置：** Jinan pt8/hd64/ml416，collection flow = test flow = `anon_3_4_jinan_real_2500.json`；120 个快照中 4 个损坏，清洗后 116 个，因此名义 K=118 实际使用 K=116；9 个独立训练种子。

| 方法 | 名义 K | 实际 K | 训练种子 n | ATT | AQL | AWT | 相对 zero-shot 的 AWT 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | - | - | - | 256.66 | 127.57 | 41.47 | - |
| frozen c30 | 32 | 32 | 9 | 257.23 | 128.67 | 42.95 +- 0.93 | +1.47 |
| frozen c30 | 118 | 116 | 9 | 258.90 | 131.06 | 46.78 +- 0.92 | +5.30 |
| responsive H30 | 32 | 32 | 9 | 257.46 | 129.06 | 43.57 +- 0.78 | +2.10 |
| responsive H30 | 118 | 116 | 9 | 259.68 | 131.99 | 48.46 +- 2.41 | +6.99 |
| responsive H120 | 32 | 32 | 9 | 257.04 | 128.21 | 42.75 +- 0.89 | +1.28 |
| **PRCT responsive H120** | **118** | **116** | **9** | 256.87 | 128.01 | **41.47 +- 0.98** | **0.00** |

**观察：**

- 六个蒸馏臂没有一个优于 zero-shot；PRCT H120 K=116 是唯一不退化的臂，但仅与 zero-shot 持平。
- collection/test flow 相同并未改变结论，因此零收益不能归因于 collection-test mismatch。
- 其他五臂在 ATT/AQL/AWT 上同步变差，表明这不是只影响 AWT 的统计波动。

**解释与含义：**

该流量下 source policy 已接近可达到的性能，反事实教师没有提供足够的额外决策信息。它是 PRCT 的**适用边界**，不是应通过调权重、挑种子或复用 final test flow 继续优化掉的失败。

---

## 4. 被否定或降级的替代路线

### 4.1 Frozen long-horizon、CRCD reliability gate：负结果

| 教师 / 方法 | K=8 AWT | K=32 AWT | K=118 AWT | 结论 |
|---|---:|---:|---:|---|
| frozen c30 | 61.81 +- 4.49 | 80.39 +- 8.11 | 83.24 +- 14.13 | 高 K 明显退化 |
| frozen c300 | 78.68 +- 12.33 | 94.38 +- 7.16 | 236.00 +- 18.52 | 长视界冻结邻居产生灾难性退化 |
| CRCD reliability gate | 68.05 | 97.01 | 155.85 | gate 未防止退化，反而恶化 |

冻结 H300 与 CRCD 的 action distribution 已出现全直行（ETWT）塌缩。CRCD K=118 的 `155.85` 同时差于 zero-shot `67.75` 和 frozen c30 `83.24`。因此，可靠性筛选、多视界投票或 longer frozen rollout 不应进入当前论文主线。

### 4.2 Loss 不是高 K 退化的根因

在相同 frozen c30 标签上，hard-BC only 的 AWT 为：K=8 `67.24 +- 9.28`、K=32 `70.30 +- 5.11`、K=118 `94.74 +- 12.98`。它没有消除高 K 退化，且 K=118 比 mixed + ranking 的 `83.24 +- 14.13` 更差。

因此，问题在 frozen-neighbor counterfactual 的 hard label 语义，而非 soft target 或 ranking loss。本结论支撑优先修改教师，不再搜索新的 student loss。

### 4.3 PRCAT：应作为负向删除消融

PRCAT 在 PRCT H120 标签上加入 source-relative advantage weight 与低优势 source-action KL prior。唯一有效的 full-PRCAT pilot 使用一个 training seed（2024）和三个 CityFlow evaluation seeds：

| 方法 | 配置 | 训练种子 n | AWT |
|---|---|---:|---:|
| zero-shot | Jinan real，max_length=64 | - | 43.98 |
| PRCT unweighted | 相同 source / seed 2024 | 1 | **39.44** |
| PRCT unweighted | 相同协议 | 9 | 40.97 +- 0.84 |
| PRCAT full | 相同 source / seed 2024 | 1 | 42.48 +- 0.22 |

PRCAT 虽优于 zero-shot，却比严格匹配的 unweighted PRCT seed 差 `3.04`。另外两个排队 seed 受诊断进程冲突影响而无效，不能作为证据。因此：

- 不扩展 PRCAT 训练种子；
- 不把 advantage weight/KL prior 写入标题、摘要或贡献；
- 仅在附录以“删除该额外复杂度”的负向消融报告。

### 4.4 OOD / diversity selection：不是主线

在 Jinan -> Hangzhou 的早期 frozen c30 设定中：

| Selection method | K | 训练种子 n | AWT |
|---|---:|---:|---:|
| random | 8 | 9 | 63.59 +- 4.61 |
| raw OOD | 8 | 3 | 62.18 +- 1.56 |
| OOD + diversity | 8 | 3 | 64.11 +- 6.85 |
| OOD + window quota | 8 | 3 | 63.36 +- 6.57 |
| random | 32 | 9 | 70.08 +- 6.51 |

raw OOD K=8 的三 seed 均值略优于 random，但训练样本数不平衡、置信区间宽，不能声称统计优势；diversity 变体没有增益。更重要的是，它们都没有解决 frozen teacher 的高 K 问题。因此 OOD/selection 不是 PRCT 论文的第二贡献。

---

## 5. 对 PRCT 主创新的证据评估

| 评价维度 | 当前判断 | 支撑证据 | 仍缺的证据 |
|---|---|---|---|
| 问题贴合度 | 强 | frozen teacher 高 K 退化、through collapse；PRCT 直接改变非目标路口响应 | 无 |
| 机制具体性 | 强 | 同 snapshot restore、target override、source-policy rollout、四动作 schema 已实现 | handler-only 审计需固化为论文图表 |
| 简洁性 | 强 | 不增加学生参数、VLM 主干、loss 或在线开销 | 无 |
| 实证效果 | 中等偏强 | Hangzhou 27.9% AWT 改善；Jinan-real 4.2% 且 9/9 正向 | 第二个非饱和目标网络 |
| 因果识别 | 中等 | c30/H300/CRCD/hard-BC 的负结果与 PRCT 正结果一致 | **缺 frozen-H120 vs PRCT-H120 的完全匹配对照** |
| 外部有效性 | 中等偏弱 | 跨网络 Hangzhou 正例 + Jinan real 正例 | Jinan-2500 为零收益；还需第二个有 headroom 的目标 |
| 新颖性边界 | 中等 | 新意位于 policy-responsive multi-intersection teacher，而非 generic counterfactual/RL | 需最终检索确认是否存在直接同类教师 |
| 论文就绪度 | REVISE | 单一故事清晰、负结果可解释 | 需要完成两项关键验证 |

### 5.1 可以写入论文的主张

1. Frozen-neighbor counterfactual supervision 在多路口 TSC 中可能产生协调不一致标签，并在高标签预算下发生 action-collapse / AWT 退化。
2. PRCT 用 source-policy-responsive rollout 构造教师，在已有的 Jinan -> Hangzhou 和 Jinan real 设置中提高 closed-loop AWT 表现。
3. PRCT 的收益受 source-policy headroom 限制；Jinan-2500 same-flow 是零收益边界。
4. CRCD、hard-BC、PRCAT、OOD/diversity selection 不能替代教师语义修复。

### 5.2 目前不能写入论文的主张

1. PRCT 在任意网络、任意需求分布上都提升适配性能。
2. PRCT 的收益已经被完全归因于 policy response，而不是 H120 horizon、实现细节或未控制变量。
3. PRCT 改善 ATT、AQL、AWT 和所有安全指标；现有稳定主信号是 AWT，tail safety 需要补测。
4. PRCAT / advantage-weighted conservative distillation 是一个有效的第二创新。
5. raw OOD selection 显著优于 random。

---

## 6. 最关键的补实验与决策门

### 必须完成 1：matched frozen-H120 vs PRCT-H120

在 Jinan -> Hangzhou 使用同一 clean snapshot pool、同一 source checkpoint、同一 H=120、同一 cost、同一 yellow transition、同一 loss、同一训练预算和相同训练 seeds。唯一变量是 non-target intersections 的控制者：frozen snapshot phase 或 source policy re-planning。

- 先运行 3 个 paired training seeds 作为预算 gate；
- 只有 AWT 方向三 seed 一致且 p95/max movement wait、spillback、entropy 不恶化时，才扩展至 9 个 seeds；
- 若 PRCT 仅优于 c30、但不优于 frozen-H120，则不得把收益归因于 policy response。

### 必须完成 2：第二个有 headroom 的目标网络

先以 zero-shot、可行 reference 和固定 evaluation seeds 执行 headroom gate。只有 zero-shot 显示真实可改善余量时，才比较 frozen-H120 与 PRCT-H120；先 3 seeds，再 9 seeds。

- 若复现成功：PRCT 可作为聚焦方法论文主创新；
- 若失败：降级为“协调不一致教师偏差的诊断发现”，而不是通用 adaptation method。

### 必须完成 3：安全指标补齐

每个 main-table arm 报告 p95/max per-movement wait、spillback、action entropy 和 phase frequency。若平均 AWT 改善以 starvation、spillback 或单相位塌缩为代价，方法应判为失败。

---

## 7. 最终建议

**建议：继续以 PRCT 作为唯一主创新。**

推荐的论文定位是：

> Policy-responsive counterfactual teachers correct coordination-inconsistent supervision for multi-intersection traffic signal control.

推荐的贡献表述是：

1. 定义并实现一种 fixed-local-action、policy-responsive-network-reaction 的多路口 counterfactual teacher；
2. 证明 frozen-neighbor teacher 的高预算退化与 PRCT 的闭环改善之间的机制联系；
3. 报告 source-policy headroom 限制下的零收益边界，而非隐去负结果。

**不建议：** 再把 PRCAT、CRCD、OOD selection、DAgger 或新 loss 作为并行创新加入。这会削弱“教师干预语义修复”这一最有力的单一故事。

**投稿决策：** 当前应为 `REVISE`。完成 matched frozen-H120 对照和第二个有 headroom 的网络复现后，若二者均通过，可将结论升级为 `READY`；任一关键对照失败，则应收窄为诊断型论文主张。

---

## 8. 结果来源

- `refine-logs/EXPERIMENT_FINAL_REPORT.md`：Hangzhou Phase 4-7、frozen / CRCD / loss / responsive H30/H120 结果。
- `README/jinan_real_responsive_h120_20260815.md`：Jinan real H120 的九训练种子聚合。
- `README/jinan2500_pt8_full_matrix_sameflow_20260817.md`：Jinan-2500 collect=test 的完整九种子矩阵。
- `refine-logs/PRCAT_EXPERIMENT_PLAN.md`：PRCAT matched pilot、无效 queued seeds 与停止决策。
- `refine-logs/EXPERIMENT_PLAN.md`：下一轮 matched H120 与第二网络验证的执行门。

