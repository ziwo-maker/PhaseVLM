# 2026-04-12 Stage3 显存与 Batch 优化说明

## 背景

`train_projection_ppo_traj.py` 的 Stage3 PPO 之前主要有三个问题：

- 没有 `grad_accum_steps`，每个 minibatch 都会立刻 `backward + all_reduce + step`
- 动作 logits 默认一次性对 4 个 action 全量打分，显存峰值接近按 `4 x mini_batch_size` 增长
- 没有显式的 activation checkpointing 开关，显存压力主要落在冻结 backbone 的前向激活上

这会导致：

- 本地 `mini_batch_size` 很难做大
- PPO 的有效 batch 只能靠直接增大本地 batch，容易先撞显存
- 8 卡/4 卡环境都无法稳定地把有效 batch 做上去

## 本次改动

### 1. 新增 `--grad_accum_steps`

Stage3 现在支持梯度累积。

行为上改成：

- 每个 minibatch 先只做 `backward`
- 到了累积边界再做一次 `sync_gradients + optimizer.step`

因此有效 optimizer batch 现在变成：

```text
effective_batch = mini_batch_size * world_size * grad_accum_steps
```

这允许在几乎不增加单步激活显存的前提下，把有效 batch 做大。

### 2. 新增 `--action_eval_chunk_size`

Stage3 里的动作打分原来会把每个样本扩成 4 份，一次性评估 4 个动作。

现在支持把动作分块打分，例如：

- `--action_eval_chunk_size 1`
- `--action_eval_chunk_size 2`

含义是：

- `1`：4 个动作逐个打分，峰值显存最低
- `2`：每次算 2 个动作，在速度和显存之间折中
- `0`：保持原来的“一次性算完全部动作”

### 3. 新增 `--gradient_checkpointing`

Stage3 现在支持：

```bash
--gradient_checkpointing on
```

这会显式对 backbone 的重型前向路径启用 activation checkpointing，用重算换显存。

这里的目标不是省参数显存，而是降低：

- vision tower 前向激活
- 动作 tag 打分时的 LLM 前向激活

## 使用建议

### 8 x 4090

建议先从下面这组开始：

```bash
--mini_batch_size 1
--grad_accum_steps 8
--action_eval_chunk_size 1
--gradient_checkpointing on
```

此时有效 batch 约为：

```text
1 * 8 * 8 = 64
```

### 4 x 5090

建议先从下面这组开始：

```bash
--mini_batch_size 1
--grad_accum_steps 8
--action_eval_chunk_size 1
--gradient_checkpointing on
```

此时有效 batch 约为：

```text
1 * 4 * 8 = 32
```

如果显存还有余量，再尝试：

```bash
--mini_batch_size 2
--grad_accum_steps 8
--action_eval_chunk_size 1
--gradient_checkpointing on
```

## 兼容性

- 旧命令不加这三个参数也能继续跑
- `--action_eval_chunk_size 0` 等价于旧版行为
- `--grad_accum_steps 1` 等价于旧版行为
- `--gradient_checkpointing off` 等价于旧版行为

## 涉及代码

- `projection_train/train_projection_ppo_traj.py`
- `projection_train/prefix_action_policy.py`
- `models.py`
