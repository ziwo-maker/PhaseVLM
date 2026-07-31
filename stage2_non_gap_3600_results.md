# Stage 2 非 Gap 训练集 @ 3600s 在线测试 — 最佳效果汇总

> 生成日期：2026-06-01
> 远程服务器：`direct.virtaicloud.com:30022` (VirTAI Cloud)
> 项目路径：`/gemini/code/LLMSTSC`

---

## 筛选规则

1. **仅 Stage 2**（排除 Stage 1 对齐训练、Stage 3 PPO 微调）
2. **排除 gap/硬规则训练集**：目录名含 `rule_gap` 或 `gap` 的全部排除
3. **仅 3600 秒在线测试**：`run_counts=3600` 或 QV 行数 ≥100（120步 × 30s = 3600s）
4. **三个效果指标**：`test_avg_queue_len_over` + `test_avg_waiting_time_over` + `test_avg_travel_time_over`，求和越低越好
5. **仅同数据集**：训练数据与测试数据一致，排除跨数据集迁移

---

## 总览

| 数据集 | 同数据集非gap Stage 2 @3600s | Checkpoint 数量 | 最佳 Sum (Q+W+T) |
|--------|------------------------------|:---------------:|------------------|
| **jn_2000** | ✅ 有结果 | 12 | **455.14** |
| **jn_2500** | ✅ 有结果 | 1 | **1088.99** |
| **jn_real** | ✅ 有结果（训练eval） | 1 | **481.59** |
| **hz_real** | ✅ 有结果 | 16 | **568.87** |
| **hz_5816** | ✅ 有结果 | 3 | **578.52** |

**5/5 数据集均有测试结果。其中 jn_real 仅有 gap 训练的 checkpoint，非 gap 训练在 jn_real 上的测试数据缺失。**

---

## 各数据集详细结果

### jn_2000（Jinan 3×4，2000s 流量文件）

训练数据：`anon_3_4_jinan_real_2000.json`
路径前缀：`records/pipeline_runs/exp_jinan2000/`

| Rank | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | Reward↑ | 训练 Run |
|:----:|-------:|------:|--------:|---------:|--------:|----------|
| **#1** | 118.08 | 57.46 | 279.59 | **455.14** | -623.62 | `stage2_bc_val02` |
| #2 | 122.03 | 51.99 | 283.90 | 457.91 | -629.26 | `stage2_curriculum_realaux_real2000_20260516` |
| #3 | 123.19 | 52.23 | 284.74 | 460.16 | -632.52 | `stage2_norebalance_test_20260518_025903` |
| #4 | 119.54 | 60.57 | 281.78 | 461.89 | -624.87 | `stage2_v2` |
| #5 | 123.80 | 55.03 | 284.95 | 463.78 | -640.38 | `stage2_diag_beststage1_lr7e5_..._ep16_save2` |
| #6 | 122.85 | 62.35 | 284.02 | 469.22 | -631.65 | `stage2_beststage1_deduped_temp010_lr1e4_vr01_bs8ga8_ep20` |
| #7 | 125.28 | 61.19 | 286.63 | 473.10 | -640.40 | `stage2_diag_beststage1_lr7e5_...` (full_online) |
| #8 | 134.73 | 69.98 | 295.02 | 499.74 | -652.79 | `stage2_beststage1_deduped_temp010_lr5e5_min5e6_...` |
| #9 | 134.11 | 79.08 | 292.77 | 505.96 | -654.08 | `stage2_actionbal_test_20260518_014824` |
| #10 | 165.74 | 105.70 | 324.25 | 595.69 | -698.10 | `stage2_expB_test_20260518_052337` |
| #11 | 202.80 | 153.01 | 353.84 | 709.65 | -733.83 | `stage2_expA_test_20260518_041542` |
| #12 | 172.82 | 273.52 | 319.41 | 765.74 | -670.06 | `stage2_prefix8` |

**最佳 checkpoint 路径**：
```
records/pipeline_runs/exp_jinan2000/stage2_bc_val02/checkpoints/best.pt
```

---

### jn_2500（Jinan 3×4，2500s 流量文件）

训练数据：`anon_3_4_jinan_real_2500.json`
路径前缀：`records/pipeline_runs/exp_jinan2500/`

| Rank | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | Reward↑ | 训练 Run |
|:----:|-------:|------:|--------:|---------:|--------:|----------|
| **#1** | 329.38 | 385.29 | 374.32 | **1088.99** | -718.01 | `stage2_norebalance_prefix16_stage1temp005_...` |

⚠️ 仅 1 个 checkpoint，且效果很差（Sum > 1000）。基本等于没有有效的非 gap 结果。

**最佳 checkpoint 路径**：
```
records/pipeline_runs/exp_jinan2500/stage2/stage2_norebalance_prefix16_stage1temp005_lr7e5_min7e6_alpha03_temp07_vr01_bs4ga16_ep16_20260519_041910/checkpoints/best.pt
```

---

### jn_real（Jinan 3×4，真实流量文件）✅

训练数据：`anon_3_4_jinan_real.json`
路径前缀：`records/pipeline_runs/exp_jinan/`

> 📌 结果来自 `stage2_bc` 训练流程的 **内部 eval**（`test_runs/`），`run_counts=0`（跑完流量文件），test wall time = 890s。这是 jinan_real 上**唯一的非 gap Stage 2 测试结果**。

| Rank | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | Reward↑ | 训练 Run |
|:----:|-------:|------:|--------:|---------:|--------:|----------|
| **#1** | 162.39 | 47.89 | 271.31 | **481.59** | -612.42 | `stage2_bc` |

**最佳 checkpoint 路径**：
```
records/pipeline_runs/exp_jinan/stage2_bc/checkpoints/best.pt
```

> ⚠️ 注：`online_test_norule_3600` 中有 3 个结果也是跑在 jinan_real 上的（最佳 Sum=423.39），但它们测试的是 gap 训练的 checkpoint（`rule_gap2`/`rule_gap3`），已排除。

---

### hz_real（Hangzhou 4×4，真实流量文件）

训练数据：`anon_4_4_hangzhou_real.json`
路径前缀：`records/pipeline_runs/exp_hangzhou_real/`

| Rank | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | Reward↑ | 训练 Run |
|:----:|-------:|------:|--------:|---------:|--------:|----------|
| **#1** | 81.91 | 150.43 | 336.53 | **568.87** | -708.68 | `stage2_manual_no_balance_lr7e5_alpha02_temp08_ep14_save1` |
| #2 | 87.35 | 209.33 | 342.32 | 639.00 | -687.96 | `stage2_wait_travel_sweep` — W2 epoch10 |
| #3 | 101.48 | 211.97 | 354.85 | 668.30 | -735.14 | `stage2_w2_guarded_online3600` — epoch 12 |
| #4 | 99.39 | 229.26 | 353.81 | 682.45 | -762.24 | `stage2_wait_travel_sweep` — W1 epoch12 |
| #5 | 100.78 | 233.02 | 353.87 | 687.68 | -724.54 | `stage2_manual_no_balance_from_ep10_best` |
| #6 | 96.44 | 239.79 | 354.81 | 691.04 | -719.85 | `stage2_wait_travel_sweep` — W2 epoch11 |
| #7 | 116.72 | 280.45 | 369.09 | 766.26 | -795.16 | `stage2_w2_guarded_online3600` — epoch 10 |
| #8 | 110.85 | 292.88 | 365.72 | 769.46 | -775.30 | `stage2_w2_guarded_online3600` — epoch 11 |
| #9 | 110.61 | 309.96 | 368.02 | 788.59 | -781.35 | `stage2_param_sweep` — C epoch9 |
| #10 | 115.13 | 342.18 | 369.77 | 827.08 | -794.95 | `stage2_w2_guarded_online3600` — epoch 13 |
| #11 | 126.92 | 334.73 | 380.12 | 841.76 | -799.37 | `stage2_norebalance_lr7e5_alpha02_temp08_ep14` — epoch 11 |
| #12 | 129.75 | 330.05 | 383.63 | 843.44 | -808.65 | `stage2_norebalance_lr7e5_alpha02_temp08_ep14` — last.pt |
| #13 | 131.03 | 325.26 | 388.49 | 844.78 | -833.03 | `stage2_norebalance_lr7e5_alpha02_temp08_ep14` — best.pt |
| #14 | 140.28 | 323.87 | 393.65 | 857.80 | -858.83 | `stage2_param_sweep` — B epoch11 |
| #15 | 136.52 | 404.64 | 389.86 | 931.02 | -774.93 | `stage2_w2_guarded_online3600` — epoch 14 |
| #16 | 460.57 | 749.53 | 737.25 | 1947.35 | -1391.54 | `stage2_w2_guarded_online3600` — epoch 1 (未收敛) |

**最佳 checkpoint 路径**：
```
records/pipeline_runs/exp_hangzhou_real/three_stage_20260520_035801/stage2_manual_no_balance_lr7e5_alpha02_temp08_ep14_save1_20260520_065240/checkpoints/best.pt
```

---

### hz_5816（Hangzhou 4×4，5816s 流量文件）

训练数据：`anon_4_4_hangzhou_real_5816.json`
路径前缀：`records/pipeline_runs/exp_hangzhou_5816/`

| Rank | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | Reward↑ | 训练 Run |
|:----:|-------:|------:|--------:|---------:|--------:|----------|
| **#1** | 188.30 | 56.73 | 333.49 | **578.52** | -666.64 | `stage2_from_stage1temp007_lr1e4_a025_bal_bs9ga7_ep14_trainonly` |
| #2 | 222.28 | 105.96 | 352.04 | 680.29 | -735.83 | `stage2_from_stage1temp007_lr7e5_a010_bal_macrof1_bs9ga7_ep10_trainonly` |
| #3 | 266.90 | 221.74 | 369.14 | 857.78 | -821.18 | `stage2_from_stage1temp007_lr7e5_a010_nobal_bs9ga7_ep10_trainonly` |

> ⚠️ 之前排名第一的 q=69.06 结果是 **跨数据集**（5816训练 → hz_real测试），已排除。

**最佳 checkpoint 路径**：
```
records/pipeline_runs/exp_hangzhou_5816/stage2_from_stage1temp007_lr1e4_a025_bal_bs9ga7_ep14_trainonly_20260521_165725/checkpoints/best.pt
```

---

## 所有数据集全局排名（同数据集 only，Top 10）

| Rank | 数据集 | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | Reward↑ | 备注 |
|:----:|--------|-------:|------:|--------:|---------:|--------:|------|
| **#1** | jn_2000 | 118.08 | 57.46 | 279.59 | **455.14** | -623.62 | |
| #2 | jn_2000 | 122.03 | 51.99 | 283.90 | 457.91 | -629.26 | |
| #3 | jn_2000 | 123.19 | 52.23 | 284.74 | 460.16 | -632.52 | |
| #4 | jn_2000 | 119.54 | 60.57 | 281.78 | 461.89 | -624.87 | |
| #5 | jn_2000 | 123.80 | 55.03 | 284.95 | 463.78 | -640.38 | |
| #6 | jn_2000 | 122.85 | 62.35 | 284.02 | 469.22 | -631.65 | |
| #7 | jn_2000 | 125.28 | 61.19 | 286.63 | 473.10 | -640.40 | |
| #8 | **jn_real** | 162.39 | 47.89 | 271.31 | **481.59** | -612.42 | |
| #9 | jn_2000 | 134.73 | 69.98 | 295.02 | 499.74 | -652.79 | |
| #10 | jn_2000 | 134.11 | 79.08 | 292.77 | 505.96 | -654.08 | |

---

## 关键发现

1. **Jinan 2000 非 gap 绝对占优**：非 gap Top 7 全是 jn_2000，Sum 在 455-473 之间。但 Jinan 是 3×4 路口网，Hangzhou 是 4×4，场景复杂度不同，不宜直接对比绝对值。

2. **jinan_real 有结果但 rank #8**：`exp_jinan/stage2_bc` 是唯一一个在 jinan_real 上有非 gap eval 的 Stage 2，Sum=481.59。效果不如 jn_2000 的最佳，但优于 jn_2000 的后续尝试（#9、#10）。

3. **Hangzhou Real 最佳 Queue 最低**：`stage2_manual_no_balance_lr7e5` 的 Queue=81.91 是所有同数据集训练的全局最低，但 Waiting Time（150.43）比 Jinan（~55）高很多。

4. **jn_2500 基本无效**：唯一的结果 Sum=1088.99，效果极差。

5. **跨数据集迁移能力**（排除在排名外但值得注意）：hz_5816 训练的 `stage2_from_stage1temp007_lr1e4_a025_bal` 在 hz_real 上测试达 q=69.06 sum=447.10，跨数据集泛化能力很强。

---

## 排除的 Stage 2 训练集（gap/硬规则）

以下 Stage 2 目录因含 gap 规则被排除，不做统计：

| 目录 | 城市 |
|------|------|
| `exp_jinan/stage2_jinan_real_queue_wait_rule_gap2_p60_ml416_ep12_bs2_ga32_20260529` | Jinan |
| `exp_jinan2000/stage2_queue_wait_rule_gap2_p60_20260529_014632` | Jinan 2000 |
| `exp_jinan2000/stage2_queue_wait_rule_gap2_p60_ml412_bs2_ga32_20260529_035106` | Jinan 2000 |
| `exp_jinan2500/stage2_queue_wait_rule_gap3_p60_ml412_bs2_ga32_20260529_073336` | Jinan 2500 |
| `exp_hangzhou_real/stage2_queue_wait_rule_gap2_p60_20260528_055503` | Hangzhou Real |
| `exp_hangzhou_real/stage2_queue_wait_rule_gap2_p60_ml412_qw_20260530_115914` | Hangzhou Real |
| `exp_hangzhou_5816/stage2_queue_wait_rule_gap2_p60_ml416_qw_20260531` | Hangzhou 5816 |
