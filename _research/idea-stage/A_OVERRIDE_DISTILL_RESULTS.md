# A-Override Distillation Experiment Results

**Date**: 2026-08-12
**Experiment**: R1 (uncorrected BC) vs R2 (A-override distilled BC)
**Status**: ✅ COMPLETED

## Results

| 指标 | R1 (未纠正) | R2 (A-Override蒸馏) | 改善 |
|------|------------|-------------------|------|
| Queue | 119.3 | **79.0** | **-33.8%** |
| Wait | 243.5 | **108.1** | **-55.6%** |
| Travel | 381.0 | **332.1** | **-12.8%** |
| **Sum** | **743.8** | **519.2** | **-30.2%** |

> 测试条件：Hangzhou 4×4 真实流量, 3600s, **A-override 关闭**, seed=2024

## 与历史结果对比

| 方法 | Queue | Wait | Travel | Sum |
|------|-------|------|--------|-----|
| R2 (本次蒸馏, Jinan Stage1) | 79.0 | 108.1 | 332.1 | 519.2 |
| 最佳非gap Stage2 (Hangzhou原生) | 81.9 | 150.4 | 336.5 | 568.9 |
| 5月蒸馏 (Hangzhou原生Stage1) | 58.9 | 39.9 | 308.2 | 407.0 |

## 关键发现

1. **A-override 蒸馏极其有效**: 在同架构、同数据、唯一变量是319个标签修正（16.6%的决策）的情况下，Sum 改善 30.2%
2. **即使用 Jinan Stage 1 做 transfer**，蒸馏后也超越了 Hangzhou 原生训练的最佳非 gap Stage 2
3. **Wait 改善最大 (-55.6%)**: 蒸馏让模型学会了更及时地服务 max-queue phase，减少等待时间
4. **提升空间**: 如果用 Hangzhou 原生 Stage 1，预期 Sum 可接近 5 月的 407

## 下一步

- 多 seed 验证 (seeds 2025, 2026)
- 用 Hangzhou 原生 Stage 1 复现
- 消融：不同 override 比例的影响
