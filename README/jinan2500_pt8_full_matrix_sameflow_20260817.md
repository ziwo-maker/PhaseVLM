# jinan_real_2500 同流（sameflow）pt8/hd64/ml416 完整矩阵（3 arm × K={32,118} × 9 seed）

> 日期：2026-08-17　|　任务：用户要求「stage 2 之后训练数据集与测试数据集一致」，按 **collect flow = test flow = `anon_3_4_jinan_real_2500.json`** 重做完整矩阵
> 输出目录：`records/pipeline_runs/jinan_2500_pt8_full_matrix_sameflow_20260817/`
> 聚合文件：`full_matrix_aggregate.json`（23:44 生成，rc=0）
> 源 checkpoint：`All_best/jinan_2500_best_AQL=126.6167，ATT=256.2499，AWT=42.1846.pt`（pt8/hd64/ml416）

## 0. 结论先行（一句话）

**同流重做确认了负面结果：即使 collect flow = test flow = 2500，6 个 Stage-3 蒸馏变体无一优于 zero-shot；最佳臂 `responsive H120 K=118` AWT=41.47，与 zero-shot 完全持平（ΔAWT=−0.00）。**

与上一轮跨流 run（`jinan2500_pt8_full_matrix_20260816`，collect=2000/test=2500，H120 K=118 ΔAWT=−0.24）结论一致。**流一致性不是瓶颈**：源策略在 `jinan_real_2500` 上已饱和，反事实蒸馏无论流对齐与否都无法挤出增益。

## 1. 实验性质与同流约束

本 run 专为回答用户的一致性疑问而设：stage-2 之后**训练（collect）与测试（test）用同一份流量** `anon_3_4_jinan_real_2500.json`。三臂对比沿用：

- `frozen c30`：policy-free 邻居 SimExpert，30s 局部（负对照）
- `responsive H30`：policy-responsive SimExpert，horizon 30（短视界对照）
- `responsive H120`：policy-responsive SimExpert，horizon 120（主复现臂）

## 2. 配置

| 项 | 值 |
|---|---|
| 源 checkpoint | pt8 / hd64、`queue_wait_short`、**`max_length=416`**（ml416） |
| 源 align ckpt | `jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt`（pt8/hd64 替代） |
| collection flow | `anon_3_4_jinan_real_2500.json`（**与 test 相同**） |
| test flow | `anon_3_4_jinan_real_2500.json` |
| 损失 | mixed（hard CE + soft α=0.3 + ranking w=0.05） |
| label budgets | K=32、K=118 |
| 训练种子 | 2024 42 123 7 11 99 314 2718 9001（9 个） |
| eval 种子 | 2024 42 123 |
| 快照 | collect 120 个，其中 **4 个被 CityFlow dump 写坏**（0 字节 ×1、196608 截断 ×3），过滤后 **116 个 clean**（1392 records）；K=118 臂实际选 `min(118,116)=116` |

## 3. 结果（n=9 训练种子，mean±std）

| Arm | ATT | AQL | AWT | ΔAWT vs zero-shot |
|---|---|---|---|---|
| zero-shot | 256.66 | 127.57 | **41.47** | — |
| frozen c30 K=32 | 257.23 | 128.67 | 42.95 ± 0.93 | +1.47 ❌ |
| frozen c30 K=118 | 258.90 | 131.06 | 46.78 ± 0.92 | +5.30 ❌ |
| responsive H30 K=32 | 257.46 | 129.06 | 43.57 ± 0.78 | +2.10 ❌ |
| responsive H30 K=118 | 259.68 | 131.99 | 48.46 ± 2.41 | +6.99 ❌ |
| responsive H120 K=32 | 257.04 | 128.21 | 42.75 ± 0.89 | +1.28 ❌ |
| **responsive H120 K=118** | 256.87 | 128.01 | **41.47 ± 0.98** | **−0.00** ⚠️ |

**关键结论：**

- **没有任何臂取得改进**。最佳臂 `H120 K=118` AWT=41.47 = zero-shot 精确持平（Δ=−0.00），其余 5 臂全部负收益（+1.3 ~ +7.0）。
- **同流没有救回 in-network 式增益**。与跨流 run 相比：c30 K=118（44.01→46.78）、H30 K=118（46.14→48.46）反而更差；H120 K=32（45.47→42.75）略好但仍差于 zero-shot；H120 K=118（41.23→41.47）基本不变，仍是唯一的持平臂。
- **三指标同涨同落**（负收益时 ATT/AQL/AWT 一起升），与 in-network 中「信号只落在 AWT」的规律相反——held-out 上连 AWT 信号也没有。

## 4. 跨流 vs 同流对比（核心科学点）

| | 跨流 20260816（collect=2000 / test=2500） | **同流 20260817（collect=test=2500）** |
|---|---|---|
| zero-shot AWT | 41.47 | 41.47 |
| H120 K=118 ΔAWT | −0.24（−0.6%） | **−0.00（0.0%）** |
| 最佳臂 | H120 K=118（持平） | H120 K=118（持平） |
| 最差臂 | H30 K=118（+4.67） | H30 K=118（+6.99） |
| 结论 | 蒸馏无收益 | 蒸馏无收益（更明确） |

用户要求的「collect=test」一致性成立后，结论**不反转**。这说明之前的负面结论不是「流不匹配」造成的假象，而是源策略在 `jinan_real_2500` 上 zero-shot 已饱和、反事实标签无增益空间的真实属性。

## 5. 解释

1. **流一致性消除的是「快照分布 vs 测试分布」的偏移**，但 2500 流本身的交通状态分布与源训练分布（jinan_real）的差异仍然存在；同流后 snapshot 与 test 同分布，学生本应更容易学到反事实增益——但结果依旧持平，说明源策略对 2500 流的决策已经足够好，反事实标签没有提供超额信息。
2. **H120 K=118 仍是唯一不退化臂**，但绝对增益为 0；这与 Phase-7 跨网络「H120 K=118 最佳臂」的结论一致，只是在 held-out 目标上衰减到零。
3. **4 个快照被 dump 写坏 → K=118 实际用 116**，与 20260816 的 120 个相比缺 4 个，属可忽略差异；不改变结论。

## 6. 运营备注

- **CityFlow snapshot dump 偶发写坏**：0 字节或 196608 字节截断。`run_projection_online.py` 的重试机制（5 次 + json.load 校验）在极端下仍失败；本次把 setup 校验步骤从「硬失败」改为「过滤损坏快照后继续」（备份到 `collected_states.jsonl.orig`），116 个 clean 快照足够跑 K=118。
- 新增/修改脚本：`scripts/run_jinan2500_pt8_{setup,train}_sameflow.sh`、`scripts/run_jinan_real_relabel{,_frozen_c30,_parallel}.sh`（`TRAFFIC_FILE` 参数化）。
- 完整流程：zero-shot → collect(2500) → relabel(3 教师) → train+eval（54 configs），23:44 全部完成 rc=0。
