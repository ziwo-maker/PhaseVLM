# Stage 3: A-Override 纠错蒸馏 — 实验方案

> 日期：2026-08-12
> 状态：已有初步验证，需要补全对照实验和多 seed

---

## 核心发现

### 已有实验结果（5月28-30日）

实验记录表明，A-Override 纠错蒸馏 **已经跑过并且效果极其显著**：

**Jinan 2000（3-seed 验证）:**

| 方法 | Queue↓ | Wait↓ | Travel↓ | Sum↓ |
|------|--------|-------|---------|------|
| 最佳非 gap Stage 2 BC | 118.08 | 57.46 | 279.59 | 455.14 |
| **A-Override 蒸馏 (epoch6)** | **111.06** | **40.52** | **272.55** | **424.13** |
| 提升 | -5.9% | **-29.5%** | -2.5% | **-6.8%** |

> 测试时 A-override **关闭**，0/1440 次 override，模型真正学会了纠错。

**Hangzhou Real（单次实验）:**

| 方法 | Queue↓ | Wait↓ | Travel↓ | Sum↓ |
|------|--------|-------|---------|------|
| 最佳非 gap Stage 2 BC | 81.91 | 150.43 | 336.53 | 568.87 |
| Stage 3 PPO (pre-A) | 70.72 | 55.13 | 325.89 | 451.74 |
| A-Override 运行时 (gap=2, p=60) | 57.18 | 42.45 | 306.84 | 406.47 |
| **A-Override 蒸馏 (无 override)** | **58.87** | **39.92** | **308.23** | **407.02** |
| vs 最佳非 gap BC | -28.1% | **-73.5%** | -8.4% | **-28.4%** |
| vs Stage 3 PPO | -16.8% | -27.6% | -5.4% | -9.9% |

> 蒸馏模型（测试时**不启用** A-override）与启用 A-override 的模型**性能相当**！

## 待补全的实验

5 月的实验缺少：
1. Hangzhou real 多 seed 验证（仅单次实验）
2. 同架构、同数据、唯一变量是"是否用 override 标签"的干净对照
3. 消融实验：override 样本比例、权重、gap 阈值的影响
4. New York 28×7 上的验证
5. 与当前最新 Stage 2 baseline 的公平对比

## 本次实验计划

### M0: Smoke Test（验证实验可运行）
- 使用已有 rule_gap2 数据，重跑 Hangzhou real 蒸馏训练
- 确认结果可复现

### M1: 干净对照实验
| # | 数据 | 训练 | 测试 |
|---|------|------|------|
| R1 | 原始 SimExpert | Stage 2 BC（标准） | 无 override |
| R2 | SimExpert + override 纠正 | Stage 2 BC（蒸馏） | 无 override |
| R3 | 原始 SimExpert | Stage 2 BC（标准） | A-override gap=2 p=60 |

### M2: 多 Seed 验证
- R1/R2/R3 各跑 3 seed（2024/2025/2026）
- 计算均值和 95% CI

### M3: 跨网络验证
- Jinan → Hangzhou 迁移
- New York 28×7 上验证（可选）

## 成功标准
- R2（蒸馏）在 3-seed 均值上显著优于 R1（原始 BC）
- R2 接近或达到 R3（运行时 override）的性能
- Wait 指标不显著退化
