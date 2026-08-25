# jinan_real pt8/hd64 完整矩阵（6 arm × K={32,118} × 9 seed）

> 日期：2026-08-15　|　任务：在用户指定 epoch `records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt` 上，对 `jinan_real` 跑全部实验（完整矩阵）
> 输出目录：`records/pipeline_runs/jinan_real_pt8_full_matrix_20260815/`
> 聚合文件：`full_matrix_aggregate.json`

## 0. 重要口径修正（先读这条）

源 checkpoint 是 **pt8 / hd64**（`prefix_tokens=8, hidden_dim=64`），不是 skill gate 默认的 pt32/hd256。此前 `refine-logs/EXPERIMENT_FINAL_REPORT.md` 把 pt8 判定为「AWT=1020 灾难性退化」——**这是旧评估代码的视觉维度 bug，不是真的退化**。

根因：两个 checkpoint 用了不同的视觉特征维度（`vision_projector.proj.weight` 输入维）：
- pt8（`jinan3_qwshort_targeted_20260805`）→ **4096 维**（vision merger/pooler 输出）
- pt32（`stage2_pt32_hd256_20260812`）→ **1152 维**（raw ViT last_hidden_state）

旧代码把 pt8 投影器按 4096 重建却喂 1152 维特征 → 静默垃圾 → 网格锁。当前 `models.py`（vision_dim 查找顺序改为 `hidden_size` 优先）+ `prefix_action_policy.py`（`pooler_output` 回退）已正确兼容两者。用修正后代码，**pt8 零样本 AWT=43.98，与 pt32（≈44）几乎一致——pt8 本身健康**。

因此本实验的科学问题是：**对一个健康的小容量 in-network 策略，长视界 responsive 重标注能否进一步提升**（不是「拯救退化策略」）。

## 1. 实验性质

in-network（Jinan→Jinan）。源 checkpoint 训练数据含 `jinan_real` 系列文件，target 在分布内。三臂对比：

- `frozen c30`：policy-free 邻居 SimExpert，30s 局部（负对照）
- `responsive H30`：policy-responsive SimExpert，horizon 30（短视界对照）
- `responsive H120`：policy-responsive SimExpert，horizon 120（主复现臂）

## 2. 配置

| 项 | 值 |
|---|---|
| 源 checkpoint | pt8 / hd64、`queue_wait_short`、`max_length=64` |
| collection flow | `anon_3_4_jinan_real_2000.json` |
| test flow | `anon_3_4_jinan_real.json` |
| grid | 4×3（12 路口） |
| 损失 | mixed（hard CE + soft α=0.3 + ranking w=0.05） |
| label budgets | K=32、K=118 |
| 训练种子 | 2024 42 123 7 11 99 314 2718 9001（9 个） |
| eval 种子 | 2024 42 123 |

## 3. 结果（n=9 训练种子，mean±std）

| Arm | ATT | AQL | AWT | ΔAWT vs zero-shot |
|---|---|---|---|---|
| zero-shot | 268.94 | 157.88 | **43.98** | — |
| frozen c30 K=32 | 269.36 ± 0.77 | 159.67 ± 1.21 | 50.87 ± 1.99 | +6.89 ❌ |
| frozen c30 K=118 | 269.67 ± 1.04 | 160.14 ± 1.66 | 53.92 ± 1.54 | +9.94 ❌ |
| responsive H30 K=32 | 268.46 ± 0.85 | 158.26 ± 1.64 | 48.95 ± 2.02 | +4.97 ❌ |
| responsive H30 K=118 | 269.75 ± 0.94 | 160.05 ± 1.59 | 53.54 ± 1.23 | +9.56 ❌ |
| responsive H120 K=32 | 268.70 ± 1.18 | 158.14 ± 1.62 | 42.68 ± 0.98 | −1.30 ✅ |
| **responsive H120 K=118** | 269.98 ± 0.68 | 159.27 ± 1.10 | **40.97 ± 0.84** | **−3.00（−6.8%）** ✅ |

**关键结论：**
- **只有 responsive H120 能超越零样本**，且 **K=118 是最佳臂**（AWT 40.97，−3.00）。
- **frozen c30 与 responsive H30 全是负面**（AWT +5~+10），短视界（30s 局部、min-max 归一化）重标注会**损伤**一个本来健康的策略。
- 时程单调性清晰：H120 > H30 ≈ c30；H120 内 K118 > K32。
- 可靠信号几乎只落在 **AWT**：ATT/AQL 的 Δ 都在 ±2 以内（噪声带），只有 AWT 有方向一致的差异。

### per-seed AWT（vs zero-shot 43.98）

**responsive H120 K=32**（8/9 优于 zero-shot）：
`123:40.92, 42:41.80, 314:41.89, 7:42.24, 2718:42.69, 2024:43.26, 11:43.51, 9001:43.82, 99:43.99`

**responsive H120 K=118**（9/9 优于 zero-shot）：
`2024:39.44, 123:40.39, 2718:40.40, 9001:40.67, 11:40.88, 42:40.93, 99:41.94, 314:41.96, 7:42.14`

K=118 的 **9/9 种子全部优于 zero-shot**（sign test p ≈ 0.002），是最可靠的信号。

## 4. 综合最好的一组

**responsive H120 K=118**：`AWT=40.97±0.84, ATT=269.98±0.68, AQL=159.27±1.10`。

- 主指标 AWT 是全矩阵唯一有意义的改进（−3.00 = −6.8%），且 9/9 种子一致。
- ATT/AQL 相对零样本仅 +0.4%/+0.9%，可忽略（AWT 收益远大于此）。
- 若严格零回归（travel/queue 也要求不涨），`responsive H120 K=32` 是更保守的备选（ATT/AQL 基本持平、AWT −1.30），但 AWT 收益弱于 K=118。

单种子最优：`responsive H120 K=118 / seed 2024`（AWT=39.44）。

## 5. 解释

1. **H120 单调优于 H30/c30**：长视界反事实标签能捕捉「当前动作对未来 120s 等待」的影响，短视界（30s 局部、min-max 归一化）标签反而引入偏差，把策略往坏的方向拉。
2. **in-network 头部空间小**：pt8 零样本 AWT=43.98 已接近较优，蒸馏挤出的空间有限（−3.0 ≈ −6.8%），远小于跨网络 Phase-7 的 −28%。这与 pt32 in-network 复现（−4.2%）同量级。
3. **K=118 在 in-network 是正收益**：与跨网络 Phase-7「高 K 退化」相反，in-network 无分布漂移，高 K 只增加监督密度、不放大 OOD 伪标签。
4. **与 Phase-7 结论一致**：policy-responsive H120 K=118 再次确认为最佳臂，frozen c30/H30 为负对照。

## 6. 运营备注

- 新增/修改脚本：`scripts/run_jinan_real_pt8_{setup,train}.sh`、`scripts/run_jinan_real_{zero_shot,collect,relabel,relabel_frozen_c30}.sh`、`scripts/aggregate_jinan_pt8_full_matrix.py`。
- `run_phase7_crcd.py` 已加 `--source_stage1_ckpt`/`--source_stage2_ckpt`/`--hidden_dim`/`--prefix_tokens`，支持 pt8 源。
- checkpoint 用 `--checkpoint_slim`（~268MB，去 optimizer state）；最终仅保留 54 个 `best.pt`，`last.pt`/`align_epoch_*.pt` 已清理。
