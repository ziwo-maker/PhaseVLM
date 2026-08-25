# jinan_real_2500（held-out）pt8/hd64/ml416 完整矩阵（6 arm × K={32,118} × 9 seed）

> 日期：2026-08-16（启动）→ 2026-08-17（完成）　|　任务：用用户指定源 `All_best/jinan_2500_best_AQL=126.6167，ATT=256.2499，AWT=42.1846.pt` 跑 jinan2500（held-out 目标 `jinan_real_2500`）全部实验
> 输出目录：`records/pipeline_runs/jinan_2500_pt8_full_matrix_20260816/`
> 聚合文件：`full_matrix_aggregate.json`

## 0. 结论先行（一句话）

**这是近零/负面结果：源策略在 held-out 目标上已接近上限，6 个 Stage-3 蒸馏变体中只有 `responsive H120 K=118` 能勉强持平（ΔAWT=−0.24，−0.6%，噪声内），其余 5 臂全部负收益（+2.5 ~ +4.7）。**

与 in-network 的 `jinan_real_pt8_full_matrix_20260815`（同一 H120 K=118 臂拿到 −6.8%）形成鲜明对比：**counterfactual 蒸馏在跨场景 / held-out 上无法复现增益**。

## 1. 实验性质

**held-out / 跨场景**。源 checkpoint 训练数据**仅含 `jinan_real`**（`exp_jinan` 2026-05-29，目录名含 `ml416`），target `jinan_real_2500` 在源分布之外（held-out）。三臂对比：

- `frozen c30`：policy-free 邻居 SimExpert，30s 局部（负对照）
- `responsive H30`：policy-responsive SimExpert，horizon 30（短视界对照）
- `responsive H120`：policy-responsive SimExpert，horizon 120（主复现臂）

## 2. 配置

| 项 | 值 |
|---|---|
| 源 checkpoint | pt8 / hd64、`queue_wait_short`、**`max_length=416`**（ml416，非默认 64） |
| 源 align ckpt | 用 `jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt` 替代（源自身 align 缺失；align 权重会被 init_policy 覆盖） |
| collection flow | `anon_3_4_jinan_real_2000.json`（与 target 不相交） |
| test flow | `anon_3_4_jinan_real_2500.json`（held-out） |
| 损失 | mixed（hard CE + soft α=0.3 + ranking w=0.05） |
| label budgets | K=32、K=118 |
| 训练种子 | 2024 42 123 7 11 99 314 2718 9001（9 个） |
| eval 种子 | 2024 42 123 |

> **ml416 是实质差异**：`queue_wait_short` prompt 实测 289 tokens，`max_length=64` 会截断到前 64 token；必须用 416 忠实还原源策略。

## 3. 结果（n=9 训练种子，mean±std）

| Arm | ATT | AQL | AWT | ΔAWT vs zero-shot |
|---|---|---|---|---|
| zero-shot | 256.66 | 127.57 | **41.47** | — |
| frozen c30 K=32 | 257.90 ± 0.59 | 129.29 ± 1.07 | 44.02 ± 1.37 | +2.55 ❌ |
| frozen c30 K=118 | 257.47 ± 0.55 | 129.23 ± 0.87 | 44.01 ± 1.07 | +2.54 ❌ |
| responsive H30 K=32 | 258.30 ± 1.81 | 129.92 ± 2.55 | 45.24 ± 2.90 | +3.77 ❌ |
| responsive H30 K=118 | 258.84 ± 1.07 | 130.74 ± 1.67 | 46.14 ± 2.10 | +4.67 ❌ |
| responsive H120 K=32 | 258.70 ± 0.64 | 130.83 ± 1.18 | 45.47 ± 1.28 | +3.99 ❌ |
| **responsive H120 K=118** | 256.81 ± 0.73 | 127.66 ± 1.24 | **41.23 ± 0.82** | **−0.24（−0.6%）** ⚠️ |

**关键结论：**

- **没有任何臂取得统计显著的改进**。最佳臂 `responsive H120 K=118` 的 −0.24 在 std=0.82 之内（不显著）。
- **除 H120 K=118 外，全部臂明显更差**（AWT +2.5 ~ +4.7），且三指标（ATT/AQL/AWT）一起涨。
- **H120 K=118 是唯一三指标同时回落到 zero-shot 水平的臂**（ATT +0.14 / AQL +0.10 / AWT −0.24），但均属噪声。
- **时程单调性在 held-out 上被打破**：in-network 里 H120 > H30 ≈ c30 的清晰排序，这里 H120 K=32（+3.99）反而比 c30（+2.55）更差；只有 K=118 才把 H120 拉回持平。

### per-seed AWT（vs zero-shot 41.47）

**responsive H120 K=118**（4 优 / 1 平 / 4 劣，净 −0.24）：
`123:39.42, 314:40.58, 42:40.91, 11:40.99, 2718:41.48, 99:41.69, 2024:41.84, 9001:41.90, 7:42.30`

**responsive H120 K=32**（0/9 优于 zero-shot）：
`99:43.64, 7:43.72, 2024:45.01, 123:45.33, 2718:45.39, 314:45.69, 9001:45.72, 11:46.68, 42:48.03`

**frozen c30 K=32**（0/9 优于 zero-shot）：
`2718:42.20, 42:42.22, 11:43.05, 2024:43.45, 9001:44.04, 7:44.13, 123:45.17, 99:45.87, 314:46.08`

**responsive H30 K=32**（方差最大，含离群点 9001=52.30）：
`7:41.69, 42:43.64, 2024:43.85, 99:43.89, 123:43.90, 2718:44.47, 11:46.51, 314:46.91, 9001:52.30`

## 4. 与 in-network 复现的对比（核心科学点）

| | in-network（jinan_real → jinan_real） | **held-out（jinan_real → jinan_real_2500）** |
|---|---|---|
| zero-shot AWT | 43.98 | 41.47 |
| H120 K=118 ΔAWT | **−3.00（−6.8%），9/9 种子优于 zero-shot** | **−0.24（−0.6%），4 优/1 平/4 劣** |
| 结论 | 蒸馏有稳定正收益 | 蒸馏无收益（持平） |

同一 `policy-responsive H120 K=118` 臂、同一 mixed loss、同一 9 seed 协议，**在分布内能挤出 −6.8%，跨到 held-out 就归零**。这说明：

1. 反事实标签的质量收益高度依赖「标签生成时源策略所见状态」与「学生部署时所见状态」的分布一致性；held-out 场景的 snapshot 与训练分布漂移，反事实标签的优越性被稀释。
2. 源策略（ml416 pt8）本身泛化能力已经足够，held-out 上 zero-shot 就是强基线，蒸馏挤不出额外空间。

## 5. 解释

1. **为什么 held-out 无增益**：Stage-3 蒸馏本质上是在「源策略生成的反事实标签」上做监督学习。in-network 时这些标签来自同分布状态，学生能学到 H120 长视界的真实收益；held-out 时状态分布漂移，反事实标签的可靠性下降，学生只能逼近一个「平均正确」的伪标签，最终收敛到 ≈ zero-shot（源策略）的水平。
2. **H120 K=118 仍是唯一可用的臂**：它在 held-out 上至少不退化（Δ≈0），而 H30 / c30 / H120 K=32 都会把策略带坏。这与 Phase-7 跨网络结论（H120 K=118 最佳臂）一致，但绝对增益随分布偏移而衰减。
3. **ATT/AQL 无信号**：与 in-network 一致，可靠信号几乎只落在 AWT；held-out 上连 AWT 信号也消失。

## 6. 运营备注

- **OOM 修复**：ml416 序列（~700 token）使训练激活内存翻倍，`batch_size=2` 在 31GB GPU 上边缘 OOM（约 67% config 命中 `Tried to allocate 38MiB` 碎片 OOM）。修复（`scripts/run_phase7_crcd.py`）：
  - `_env()` 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - 训练 `--batch_size 1 --grad_accum_steps 16`（有效 batch 仍 16；ranking loss 为 action 维 pairwise，不依赖 batch≥2）
  - 修复后 peak 28GB → 24.5GB；幂等重跑自动跳过已完成的 tr2024/tr99，补齐其余 seed。
- 新增/修改脚本：`scripts/run_jinan2500_pt8_{all,setup,train}.sh`、`scripts/aggregate_jinan_pt8_full_matrix.py`；`run_phase7_crcd.py` 已加 `--max_length` 覆盖（默认 64 向后兼容）。
- 源自身 align ckpt 缺失，用 `jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt`（pt8/hd64）替代。
