# Stage 2 消融实验：New York 28×7 — 数据质量 vs 损失函数

> 日期：2026-08-07 ~ 2026-08-08
> 目标：定位 DAgger + Cost Ranking 在 New York 28×7 double traffic 上的性能退化根因
> 基线 Sum (Q+W+T)：2822（qwctx 纯 BC）

---

## 方法说明

### 基线方法

| 方法 | 数据 | Stage 2 Loss | 说明 |
|------|------|-------------|------|
| **qwctx BC (基线)** | qwctx SimExpert 数据 | 纯 BC (hard) | 使用 SimExpert 生成的专家标注（queue_wait 上下文），23,520 个决策点。Stage 1 对齐 + Stage 2 纯行为克隆。这是 7 月 31 日跑出的最强基线 |
| **DAgger + cost rank** | DAgger 混合数据 | BC + soft target + cost ranking + base KL | 原始论文方法。数据由 `build_dagger_stage2_dataset.py` 聚合：stage12_newmethod 专家数据 + double_true_awt 策略 rollout（经 SimExpert 重新标注），35,480 个决策点。Loss = hard BC + cost soft-target BC (α=0.3) + 0.15 × pairwise cost ranking + 0.02 × conditional KL(policy ‖ base-action prior) |

### 消融实验

| # | 方法 | 数据 | Stage 2 Loss | 研究问题 |
|---|------|------|-------------|---------|
| **A** | DAgger 纯 BC | DAgger 混合数据 | 纯 BC (hard) | 去除所有 loss 组件后，DAgger 数据本身是否有问题？ |
| **B** | qwctx 纯 BC | qwctx SimExpert 数据 | 纯 BC (hard) | qwctx 基线是否可复现？数据质量是否可靠？ |
| **C** | qwctx + rank + KL | qwctx SimExpert 数据 | BC + soft target + cost ranking + base KL | 在好数据上，cost ranking 能否进一步提升？ |

### 实验控制变量

所有实验统一使用：
- **Backbone**: Qwen-VL-8B + prefix projection (8 tokens)
- **Prompt mode**: `queue_wait_short`
- **超参数**: lr=1e-4, batch_size=4, grad_accum=8, epochs=15, seed=2024, bf16
- **测试**: `anon_28_7_newyork_real_double.json`, run_counts=3600 (120 steps × 30s)
- **硬件**: 2× RTX 5090 (32GB) 训练 + 1× RTX 5090 online test

> 注：消融 A 使用 Aug 6 训练的 Stage 1 (mean_recall=0.166)，消融 B/C 使用 qwctx 自有 Stage 1。消融 A 和 DAgger+costrank 共用同一 Stage 1，确保对比公平。

---

## 训练指标对比

### 消融 A vs B：数据质量差异

| Epoch | A: DAgger 纯 BC (val_acc) | B: qwctx 纯 BC (val_acc) | 差距 |
|-------|--------------------------|--------------------------|------|
| 1 | 45.8% | **75.4%** | +29.6% |
| 3 | 59.6% | **80.3%** | +20.7% |
| 5 | 63.2% | **79.9%** | +16.7% |
| 15 (final) | 67.9% | **82.7%** | +14.8% |

> qwctx 数据训练收敛速度远快于 DAgger 数据。Epoch 1 准确率差距高达 29.6 个百分点，最终差距仍为 14.8 个百分点。

### 消融 B vs C：损失函数差异

| Epoch | B: 纯 BC (val_acc) | C: BC+rank+KL (val_acc) | C val_regret |
|-------|-------------------|------------------------|-------------|
| 1 | 75.4% | 71.7% | 0.217 |
| 3 | 80.3% | 80.2% | 0.150 |
| 5 | 79.9% | 81.1% | 0.142 |
| 15 (final) | 82.7% | 83.0% | **0.116** |

> Cost ranking 初期降低 val_acc（epoch 1: 71.7% vs 75.4%），但后期追上并略超（83.0% vs 82.7%）。val_cost_regret 持续下降（0.217→0.116），表明模型确实学会了候选动作的代价排序。

---

## Online Test 最终结果

**New York 28×7, Double Traffic, 3600s (120 steps × 30s)**

| # | 方法 | 数据 | Loss | Queue↓ | Wait↓ | Travel↓ | **Sum↓** | vs 基线 |
|---|------|------|------|--------|-------|---------|----------|---------|
| 🥇 | **B** | qwctx SimExpert | 纯 BC | 1657 | **211** | 937 | **2806** | **-0.6%** |
| 🥈 | 基线 | qwctx SimExpert | 纯 BC | 1642 | 220 | 959 | 2822 | — |
| 🥉 | C | qwctx SimExpert | BC+rank+KL | **1628** | 284 | 952 | 2864 | +1.5% |
| 4 | DAgger | DAgger 混合 | BC+rank+KL | 2270 | 248 | 1168 | 3686 | +30.6% |
| 5 | A | DAgger 混合 | 纯 BC | 2294 | 281 | 1182 | 3758 | +33.2% |

> Sum = Queue + Wait + Travel，越低越好。三个指标分别衡量：排队长度（Queue）、车辆等待时间（Wait）、车辆行程时间（Travel）。

---

## 核心发现

### 1. 数据质量是决定性因素（+33% 差距）

```
同一 Loss (纯 BC):
  qwctx 数据 → Sum = 2806  ← 最优
  DAgger 数据 → Sum = 3758  ← 差 33.2%
```

**原因分析**：
- qwctx 数据由 SimExpert 使用 **Gemma-3-27B** 生成，带有 queue_wait 上下文，专家动作质量高
- DAgger 数据 base 部分来自 **stage12_newmethod**（训练方法/数据源不同），rollout 部分来自 double_true_awt 策略（本身性能有限），经 SimExpert 重新标注后质量仍不及原始专家数据
- DAgger 数据还包括大量 rollout 行（316,932 行原始数据，其中仅 11,960 行被选中重新标注），引入噪声

### 2. Cost Ranking 改善 Queue 但恶化 Wait（Queue-Wait 权衡）

```
同一数据 (qwctx):
  纯 BC:              Q=1657  W=211  T=937  → Sum=2806  ← 最优
  BC+rank+KL:         Q=1628  W=284  T=952  → Sum=2864  ← 差 2.1%
```

- Cost ranking 让 Queue 改善 **1.8%**（1628 vs 1657）
- 但 Wait 恶化 **34.6%**（284 vs 211）
- 整体 Sum 退化了 2.1%

**原因分析**：
Cost ranking loss 使用候选动作的 `score`（SimExpert 复合得分）进行成对排序训练。该 score 可能更偏重 queue 相关指标，导致模型学会优先降低排队长度，但忽略了等待时间。这种不公平的代价权重引导模型走向了错误的策略方向。

### 3. 纯 BC + 高质量数据 = 当前最优方案

简单纯 BC 在 qwctx 数据上的表现（Sum=2806）已经：
- 略优于原始 qwctx 基线（2822，-0.6%）
- 显著优于 DAgger + cost ranking（3686，-23.9%）
- 无需复杂 loss 组件（cost ranking、base KL、soft target）

---

## 建议

### 短期
1. **使用 qwctx SimExpert 数据作为标准训练数据**，替换当前 DAgger 混合数据
2. **纯 BC 作为默认 Stage 2 训练方法**，简单且效果最好

### 中期
1. **改进 DAgger 数据 pipeline**：
   - 使用 qwctx SimExpert 数据作为 base（而非 stage12_newmethod）
   - 提高 rollout 策略质量（当前 double_true_awt 性能不够好）
   - 调整 SimExpert relabeling 阈值和优先级权重
2. **重新设计 Cost Ranking**：
   - 当前 cost score 的 Queue-Wait 权重需要重新校准
   - 考虑直接对 Wait 和 Travel 做多目标排序
   - 在好数据上做 ranking 权重 sweep（0.05, 0.10, 0.20）

### 长期
1. 在 Jinan 和 Hangzhou 数据集上验证纯 BC 的泛化性
2. 如果 cost ranking 的 Queue-Wait 权衡可以在多数据集上复现，考虑重新设计 cost function

---

## 复现命令

### Ablation B (qwctx 纯 BC — 最佳)

```bash
torchrun --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
    --data_path /mnt/data/LLMSTSC/generated/newyork28x7_double_simexpert_qwctx_20260730_162857/raw/traffic_prompt_series_2s.jsonl \
    --llm_path /home/data/model/qwen-vl-8B \
    --align_ckpt /home/code/LLMSTSC/records/pipeline_runs/newyork28x7_double_qwctx_20260731_040000/stage1/best.pt \
    --output_dir <output_dir> \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 --prompt_mode queue_wait_short \
    --batch_size 4 --grad_accum_steps 8 --epochs 15 \
    --lr 1e-4 --min_lr 1e-5 --warmup_ratio 0.1 --lr_schedule cosine \
    --weight_decay 1e-4 --max_grad_norm 1.0 \
    --val_ratio 0.1 --split_strategy stratified --num_workers 4 --seed 2024 --bf16 \
    --log_interval 20 --save_every 1 --best_metric val_acc --best_mode max \
    --target_mode hard
```

### Online Test

```bash
python projection_train/run_projection_online.py \
    --memo <memo> \
    --projection_ckpt <best.pt> \
    --llm_path /home/data/model/qwen-vl-8B \
    --dataset newyork_28x7 \
    --traffic_file anon_28_7_newyork_real_double.json \
    --run_counts 3600 --num_rounds 1 --seed 2024 \
    --prompt_mode queue_wait_short \
    --seq_len 16 --image_size 224 --periodicity 16 --input_dim 8 --hidden_dim 64 \
    --prefix_tokens 8 --prefix_mode prefix --max_length 64 \
    --inference_batch_size 64 --bf16
```

---

## 实验产物路径

| 实验 | 训练目录 | Online Test 结果 |
|------|---------|-----------------|
| A: DAgger 纯 BC | `records/pipeline_runs/newyork28x7_ablation_pure_bc_20260807_run01/` | `stage2/checkpoints/best.pt` → result.json |
| B: qwctx 纯 BC | `records/pipeline_runs/newyork28x7_ablation_qwctx_pure_bc_20260807_run01/` | `stage2/checkpoints/best.pt` → result.json |
| C: qwctx+rank+KL | `records/pipeline_runs/newyork28x7_ablation_qwctx_costrank_20260808_run01/` | `stage2/checkpoints/best.pt` → result.json |
| DAgger+costrank | `records/pipeline_runs/newyork28x7_dagger_costrank_20260807_stage2_run01/` | `stage2/checkpoints/best.pt` → result.json |
| qwctx 基线 | `records/pipeline_runs/newyork28x7_double_qwctx_20260731_040000/` | `stage2/checkpoints/best.pt` → result.json |
