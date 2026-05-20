# Stage 2 实验总结：Action Balance 与 Rebalance 的影响

**日期**: 2026-05-18

## 背景问题

Stage 2 BC 训练中存在「动作饿死」问题——某些动作（NLSL、ELWL）预测频率极低，模型偏向 ETWT。尝试通过 loss 正则化和数据分析来解决。

## 实验设计

### 代码修改

**文件**: `projection_train/train_projection_bc.py`

新增两个动态 loss 正则项（不硬编码动作名）：

1. **`_compute_action_balance_loss`**: 对 batch 预测分布的熵做正则，惩罚任何动作的过度集中
2. **`_compute_overconf_penalty`**: 对超过 uniform 的 mean probability 做惩罚

新增超参：`--action_balance_weight`、`--overconf_penalty_weight`、`--global_action_weights`（默认均为 0/None，向后兼容）。

### Bug 修复

`_compute_overconf_penalty` 初版在 raw logits 上操作（值域可达数百），导致 loss 爆炸。修复为 softmax 后用概率计算。

### 实验矩阵

| 实验 | 数据 | lr | action_balance | val_f1 | Queue | Wait | Travel |
|------|------|-----|---------------|--------|-------|------|--------|
| baseline (diag) | rebalanced | 7e-5 | 0 | 0.850 | 125.3 | 61.2s | 286.6s |
| best-lr1e4 | rebalanced | 1e-4 | 0 | 0.844 | 122.9 | 62.3s | 284.0s |
| best-lr5e5 | rebalanced | 5e-5 | 0 | 0.895 | 134.7 | 70.0s | 295.0s |
| action_balance | rebalanced | 7e-5 | 0.05 | 0.864 | 134.1 | 79.1s | 292.8s |
| **no-rebalance** | **cleaned only** | **7e-5** | **0** | **0.567** | **123.2** | **52.2s** | **284.7s** |

## 核心发现

### 1. val_f1 与交通性能负相关

```
val_f1 ↑  ⟹  traffic ↓

lr=5e-5:      val_f1=0.895  →  travel=295.0s  (最差)
action_bal:   val_f1=0.864  →  travel=292.8s
lr=7e-5:      val_f1=0.850  →  travel=286.6s
lr=1e-4:      val_f1=0.844  →  travel=284.0s
no-rebalance: val_f1=0.567  →  travel=284.7s, wait=52.2s  (最好)
```

更好的 expert 模仿（val_f1 更高）并不能转化为更好的交通流。Expert action 本身不是最优策略。

### 2. Rebalance 是问题，不是解决方案

**数据流水线分析**：

```
SimExpertCollector
  └→ 每个 state 模拟 4 个 action (ETWT/NTST/ELWL/NLSL)
  └→ Min-max normalize ATT/AQL/AWT/spillback 到 [0,1] 区间
  └→ score = att_norm + aql_norm + awt_norm + 0.2×spillback_norm
  └→ 选 score 最低者 → expert_action

clean_jsonl (含 rebalance)
  └→ oversample ELWL/NLSL 等低频动作
  └→ 自然分布: ETWT=35.7% NTST=31.4% ELWL=17.0% NLSL=15.8%
  └→ Rebalanced: ETWT=26.8% NTST=24.4% ELWL=24.4% NLSL=24.4%
```

**Rebalance 有害的三个原因**：

1. **Min-max 归一化抹平了真实差异**：4 个候选在每个 state 内归一化，使得每个动作约 25% 概率被选为 expert。真实的质量差异被抹平。

2. **过采样复制了低质量样本**：ELWL/NLSL 的样本被大量复制（17%→24%, 16%→24%），但这些多出来的样本没有新信息，只是在重复同样的（可能次优的）expert 决策。

3. **人为的均匀分布不反映真实交通**：真实交通流是不均匀的。某些方向（如东西向）在特定时段确实流量更大，ETWT 应该更频繁。强制均匀让模型学到了错误的先验。

### 3. 转向均衡不是瓶颈

原始假设「NLSL 被饿死导致性能差」被证伪。去掉 rebalance 后模型对 NLSL/ELWL 预测更少（val_f1 更低），但 traffic 显著改善（wait -15%）。

## 最佳结果

**No-rebalance, lr=7e-5, epoch 10**

- val_macro_f1: 0.567
- avg_queue: 123.2
- avg_wait: **52.2s** (-15% vs baseline)
- avg_travel: 284.7s

**Checkpoint 路径**:
```
records/pipeline_runs/exp_jinan2000/stage2_norebalance_lr7e5_ep16_20260518_020759/checkpoints/best.pt
```

## 建议

### 短期
- 使用 no-rebalance 数据训练所有 stage 2 模型
- 移除 `--rebalance_targets` 参数或用 `--train_balance_mode none`

### 中期
- 修改 `_score_candidates` 中的 min-max 归一化 → 使用绝对值或全局归一化，保留动作间的真实质量差异
- 考虑用 `expert_action_margin` 过滤低置信度的 expert 样本

### 长期
- BC 的固有局限：最小化与 expert 的 KL 散度 ≠ 最小化 queue/wait/travel
- Stage 3 PPO 需要修复 value function 收敛问题（value loss 从 7 爆炸到 7000+）
- 或考虑 DAgger / offline RL 替代方案

## Stage 3 PPO 状态

两次尝试均失败，根因是 value function 不收敛。可能需要：
- 增加 value_hidden_dim（512 → 1024）
- 降低 reward_alpha（0.1 → 0.02）
- 增加 rollout_steps（8 → 16）
- 预热 value head
