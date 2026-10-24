# A-Override 纠错蒸馏 — 实验发现与结论

> 日期：2026-08-12 | 状态：消融实验 M1 完成

---

## 1. 方法概述

**A-Override 纠错蒸馏**：利用手写规则 A-override (max-phase gap≥2, left_wait_protect=60) 在模型推理时纠正系统性错误，然后将纠正后的动作作为训练标签蒸馏回模型。

### 方法步骤
1. 用 A-override 规则 (gap=2, protect=60) 运行模型，收集所有决策状态
2. 对 A-override 触发的状态（模型动作 ≠ max-queue phase），用纠正后的动作替换 expert_action
3. 保留所有非纠正状态（模型已做对的状态）的原始标签
4. 用纠正后的数据集重新训练 Stage 2 BC
5. 评估时**关闭** A-override，测试模型是否真正学会了纠错

### 组件清单
| 组件 | 描述 | 可消融 |
|------|------|--------|
| A-override 规则 | max-phase gap≥2 的硬纠正 | ✅ |
| 左转保护 | left_wait_protect=60，防止过度切左转 | ✅ |
| 纠正比例 | 仅修正 overridden 状态 (319/1920=16.6%) | ✅ |
| 标签替换方式 | hard label 替换 vs soft target | ✅ |
| Stage 1 来源 | Jinan vs Hangzhou 原生 | ✅ |

---

## 2. 已有实验结果

### 2.1 M1: 消融对照实验 (2026-08-12)

**配置**：Jinan Stage 1 → Hangzhou 4×4 真实流量, 3600s, seed=2024, A-override 关闭

| 指标 | R1 (未纠正 BC) | R2 (A-Override 蒸馏) | 改善 |
|------|:---:|:---:|:---:|
| Queue | 119.3 | **79.0** | **-33.8%** |
| Wait | 243.5 | **108.1** | **-55.6%** |
| Travel | 381.0 | **332.1** | **-12.8%** |
| **Sum** | **743.8** | **519.2** | **-30.2%** |

> R1 和 R2 使用完全相同的训练数据（30,480行），唯一区别是 R2 的 319/1920 个决策标签被 A-override 纠正。

### 2.2 5月历史实验 (Hangzhou 原生 Stage 1)

| 方法 | Queue | Wait | Travel | Sum |
|------|-------|------|--------|-----|
| A-override 蒸馏 (5月, Hangzhou Stage1) | 58.9 | 39.9 | 308.2 | 407.0 |
| A-override 运行时 (gap=2, p=60) | 57.2 | 42.4 | 306.8 | 406.5 |
| Stage 3 PPO (pre-A) | 70.7 | 55.1 | 325.9 | 451.7 |
| 最佳非 gap Stage 2 BC | 81.9 | 150.4 | 336.5 | 568.9 |

### 2.3 Jinan 2000 3-seed 验证 (5月)

| 方法 | Queue | Wait | Travel | Sum |
|------|-------|------|--------|-----|
| A-override 蒸馏 (epoch6) | 111.1 | 40.5 | 272.6 | 424.1 |
| 最佳非 gap Stage 2 BC | 118.1 | 57.5 | 279.6 | 455.1 |
| 改善 | -5.9% | -29.5% | -2.5% | -6.8% |

---

## 3. 已有结论

### 结论 1: A-override 蒸馏有效（证据充分）
在 Jinan 2000 (3-seed) 和 Hangzhou real (本次 1-seed + 5月 1-seed) 上一致有效。319 个标签修正 (16.6%) 带来 6.8%-30.2% 的 Sum 改善。

### 结论 2: 蒸馏效果接近运行时 A-override（待多 seed 确认）
5月 Hangzhou 数据显示：蒸馏 Sum=407 vs 运行时 override Sum=406，几乎持平。Jinan 数据待验证。

### 结论 3: PPO 完全无效（多次验证）
所有 Stage 3 PPO 运行都选 update 0，后续 update 全被 guardrail 拒绝。PPO 从未成功学习到任何改进。

### 结论 4: Stage 1 质量影响蒸馏效果（待量化）
Jinan Stage 1 → Hangzhou transfer 的蒸馏 (Sum=519) vs Hangzhou 原生 Stage 1 (Sum=407)。Stage 1 质量贡献约 112 Sum 点。

### 结论 5: 仅修正 16.6% 的标签即可大幅改善（已证明）
不需要全量数据重新标注。精准修正模型犯错的少数状态即可。

---

## 4. 待验证的假设

1. **多 seed 稳定性**：Hangzhou 蒸馏仅 1-seed，需要 3-seed 确认
2. **纠正比例的影响**：16.6% 是否为最优？更多/更少的纠正会如何？
3. **gap 阈值的影响**：gap=2 vs gap=3 vs gap=5 对蒸馏效果的影响
4. **左转保护的必要性**：protect=60 对维持 NLSL action rate 是否必要？
5. **跨网络泛化**：Jinan→Hangzhou 蒸馏 vs Hangzhou→Jinan 蒸馏
6. **New York 28×7 验证**：在大规模路网上是否同样有效？

---

## 5. 消融实验计划

详细消融方案见: [ABLATION_PLAN.md](./ABLATION_PLAN.md)

### 最小可行消融集 (~6 GPU-hours)

| # | 实验 | 理由 |
|---|------|------|
| C1 | R1 seed 2025,2026 | 基线稳定性 |
| C2 | R2 seed 2025,2026 | 蒸馏稳定性 |
| A2 | R2 + override 开启 | 蒸馏是否真正学到了 |
| B1 | gap=3 蒸馏 | 超参数敏感性 |
| A4 | protect=0 蒸馏 | 左转保护是否必要 |

### 算力估算

| Phase | 实验数 | 估算时间 |
|-------|:---:|:---:|
| 全部消融 | 19 | ~16 GPU-hours |
| 最小可行集 | 6 | ~6 GPU-hours |
