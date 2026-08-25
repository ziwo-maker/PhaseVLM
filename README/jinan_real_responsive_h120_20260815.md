# jinan_real 响应式 H120 反事实蒸馏复现（in-network）

> 日期：2026-08-15　|　任务：用 `llmstsc-responsive-repro` skill 在 `jinan_real` 上复现 Phase-7 最优臂（policy-responsive SimExpert H120）
> 输出目录：`records/pipeline_runs/jinan_real_responsive_h120_20260814/`

## 1. 实验性质

这是 **in-network（Jinan→Jinan）复现**，不是跨网络。源 Stage-1/Stage-2（`records/pipeline_runs/stage1_pt32_hd256_20260812/best.pt` 与 `stage2_pt32_hd256_20260812/best.pt`）的训练数据是 `jinan_real` + `jinan_real_2000` + `jinan_real_2500` 三个文件，因此 target 在分布内。

科学问题：**响应式 H120 反事实蒸馏在 in-network 是否同样有效**（Hangzhou 的 67.75→48.88 是跨网络结果）。

## 2. 配置（已通过 Stage-1/Stage-2 source-policy gate）

| 项 | 值 |
|---|---|
| 源 checkpoint | pt32 / hd256、`queue_wait_short`、`max_length=64`（gate 通过） |
| collection flow | `anon_3_4_jinan_real_2000.json`（4365 veh，较轻） |
| test flow | `anon_3_4_jinan_real.json`（6295 veh，匹配 Phase-6 in-network baseline） |
| grid | `--num_rows 3 --num_cols 4`（12 路口） |
| sim_horizon | 120（`min_action_time=30`, `traffic_series_interval=2`） |
| cost_scope | global，权重 att/aql/awt=1.0、spillback=0.2 |
| 损失 | mixed（hard CE + soft α=0.3 + ranking w=0.05） |
| label budgets | K=32、K=118 |
| 训练种子 | 2024 42 123 7 11 99 314 2718 9001（9 个，3 GPU 并行） |
| eval 种子 | 2024 42 123 |

## 3. 结果（n=9 训练种子，mean±std）

| Arm | ATT | AQL | AWT | ΔAWT vs zero-shot |
|---|---|---|---|---|
| zero-shot | 274.07 | 165.98 | **43.42** | — |
| H120 K=32 | 275.13 ± 1.91 | 166.82 ± 3.06 | 43.18 ± 0.90 | −0.24（噪声内，n.s.） |
| H120 K=118 | 274.91 ± 2.29 | 166.68 ± 3.72 | **41.59 ± 0.76** | **−1.84（4.2%）** |

**关键结论：**
- 响应式 H120 反事实蒸馏在 in-network **也有效**，且 **K=118 > K=32**，与 Phase-7 跨网络结论一致。
- K=118 的 **9/9 种子全部优于 zero-shot**（sign test p≈0.004），是最可靠的信号。
- 但 in-network 头部空间很小：−1.84 ≈ 4.2% 相对，远小于跨网络的 67.75→48.88（约 −28%）。ATT/AQL 变化在噪声内，**可靠信号只有 AWT**。

### per-seed AWT

**K=32**（5/9 优于 zero-shot）：
`123:42.04, 2024:43.59, 42:44.31, 11:42.98, 7:43.95, 99:43.28, 2718:42.97, 314:44.07, 9001:41.42`

**K=118**（9/9 优于 zero-shot）：
`123:40.11, 2024:41.00, 42:41.59, 11:43.13, 7:42.13, 99:41.45, 2718:41.52, 314:41.60, 9001:41.74`

## 4. 解释

1. **K=118 单调优于 K=32**：与跨网络 Phase-7 的「高 K 退化」现象相反，in-network 下 K=118 是稳定的正收益臂。原因可能是 in-network 无分布漂移，高 K 只是增加反事实监督密度而不会放大 OOD 伪标签。
2. **信号几乎只落在 AWT**：cost 权重 att/aql/awt=1.0 是等权的，但 ATT/AQL 的 Δ 都在 ±1 以内的噪声带，说明当前 H120 反事实标签对「平均等待」最敏感、对「吞吐/队列长度」几乎没有边际信息。
3. **相对收益小（−4.2%）**：因为源策略本身在 in-network 已经接近较优（zero-shot AWT 43.42 ≈ Phase-6 的 43.78），蒸馏能挤出的空间天然有限。

## 5. 运营备注

- 聚合脚本 `scripts/aggregate_jinan_real_h120.py`，结果 `h120_aggregate.json`。
- 新增脚本：`scripts/run_jinan_real_{zero_shot,collect,relabel,train}.sh`。
- `run_phase7_crcd.py` 已加 `--dataset`/`--test_flow`（默认 hangzhou，向后兼容），`run_phase7_policy_responsive.py` 转发。
- checkpoint 已用 `--checkpoint_slim`（303MB，去 optimizer state），`save_every=100`，磁盘占用可控（当前 1.1T/1.8T）。
