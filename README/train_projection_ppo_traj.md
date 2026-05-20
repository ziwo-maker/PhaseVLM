# `train_projection_ppo_traj.py` 修改说明

## 本次改动

- 修正了 Stage 3 PPO 中训练目标与 checkpoint 选择目标脱节的问题
  - `--selection_queue_weight`
  - `--selection_wait_weight`
  - `--selection_travel_weight`
  现在不再只用于 `best.pt` 的复合打分，也会参与 PPO reward shaping
- 修正了 `queue_reward_coef` / `waiting_reward_coef` 符号失效的问题
  - 之前 reward 计算里对这两个系数取了 `abs()`
  - 导致 `-0.35` 和 `0.35` 实际效果相同
  - 现在会保留原始符号
- 把原先没有真正进入 shaped reward 的 travel proxy 补进了训练目标
  - `system_vehicle_reward_coef`
  - `throughput_reward_coef`
  现在都会参与每一步 PPO reward
- 扩展了训练日志
  - 新增 `sv`（system vehicle reward）
  - 新增 `thr`（throughput reward）
  - 新增 `trv`（travel reward）
- 新增 reward shaping 的独立辅助模块
  - `projection_train/reward_shaping.py`
  - 负责 objective weight 归一化和局部 shaped reward 组装

## 修改目的

这次修改主要是解决两个实际问题：

1. 命令行里设置了 `selection_queue_weight / selection_wait_weight / selection_travel_weight`，但训练时并不会影响 PPO loss，只会影响 `best.pt` 选谁。
2. 命令行里设置了负数 `queue_reward_coef / waiting_reward_coef`，但由于实现里直接取绝对值，符号配置并不生效。

这会导致：

- 用户以为自己在“偏向 queue / wait / travel”训练
- 但实际上只是在改 checkpoint 选择标准
- 同时 reward 系数的正负号语义也和命令行直觉不一致

因此这次改动的目标是让 Stage 3 的参数语义重新闭合：

- 相对目标权重要真实进入训练 reward
- reward 系数符号要真实决定优化方向
- travel proxy 要真正进入 shaped reward，而不是只停留在 eval 指标或 checkpoint 选择层

## 主要行为变化

### 1. `selection_*` 从“只选 best”改为“训练 + 选 best”

当前：

- `selection_queue_weight`
- `selection_wait_weight`
- `selection_travel_weight`

会先被解析为 queue / wait / travel 三个 objective 的相对权重。

这些权重有两个用途：

1. 用于 PPO step reward 的目标加权
2. 用于 eval 后 `best.pt` 的 composite score 计算

也就是说，它们现在既决定：

- 训练时 reward 更偏向什么目标
- 也决定评估时更偏向保存什么 checkpoint

### 2. `selection_*` 会先做归一化

为了避免你只是想改“相对偏好”，却顺手把整体 reward 尺度也放大或缩小，新增了统一权重归一化逻辑。

例如：

```bash
--selection_queue_weight 2.0
--selection_wait_weight 1.0
--selection_travel_weight 2.0
```

会被归一化成平均 active weight 为 `1.0` 的一组权重，大约等价于：

- queue: `1.2`
- wait: `0.6`
- travel: `1.2`

这样：

- 相对偏好被保留
- reward 总体量级不会被无意放大

### 3. `queue_reward_coef / waiting_reward_coef` 现在保留符号

现在这两个参数作用在：

```text
current - previous
```

的指标变化量上。

也就是：

- `queue_delta = current_queue - previous_queue`
- `wait_delta = current_wait - previous_wait`

因此：

- `queue_reward_coef < 0`：奖励 queue 下降，惩罚 queue 上升
- `waiting_reward_coef < 0`：奖励 waiting 下降，惩罚 waiting 上升
- 正值则含义相反

这和命令行直觉现在是一致的。

### 4. travel proxy 现在真实参与训练 reward

之前虽然脚本里有：

- `system_vehicle_reward_coef`
- `throughput_reward_coef`

但它们没有真正进入最终的 shaped reward 组装。

现在 travel proxy 被拆成两部分：

1. `system_vehicle_reward`
   - 基于 local active vehicle count 的变化
   - 负系数表示奖励“局部进口车道活跃车辆数下降”
2. `throughput_reward`
   - 基于 local throughput count
   - 正系数表示奖励“局部进口车道车辆被放行”

最终：

```text
travel_reward = system_vehicle_reward + throughput_reward
```

并由 `selection_travel_weight` 统一参与目标加权。

## 当前 shaped reward 结构

Stage 3 默认使用 `--reward_mode stable_delta`。单个路口的 reward 现在可以概括为：

```text
queue_term      = -clip((current_queue - previous_queue) / queue_delta_scale, -1, 1)
wait_term       = -clip((current_wait - previous_wait) / wait_delta_scale, -1, 1)
active_term     = -clip((current_active - previous_active) / active_delta_scale, -1, 1)
throughput_term =  clip(local_throughput_count / throughput_scale, 0, 1)

travel_term = 0.5 * active_term + 0.5 * throughput_term

raw_reward =
  queue_weight  * queue_term
  + wait_weight * wait_term
  + travel_weight * travel_term
  - switch_penalty
  - starvation_penalty

reward = clip(raw_reward * global_reward_scale, -reward_clip, reward_clip)
```

其中：

- `queue_weight / wait_weight / travel_weight` 来自 `selection_*`
- `current_wait` 仍来自当前等待车辆的等待秒数聚合，但进入 reward 的是 `current_wait - previous_wait`
- `queue_delta_scale / wait_delta_scale / active_delta_scale / throughput_scale` 用于把各分量压到相近量级
- `reward_clip` 防止极端拥堵或 starvation penalty 让 value target 爆炸
- `switch_penalty` 和 `starvation_penalty` 仍然保持原来的抑制作用

如需复现实验旧行为，可以显式指定：

```bash
--reward_mode legacy_absolute
```

旧模式仍使用绝对 wait signal：

```text
wait_reward = wait_weight * waiting_reward_coef * current_wait
```

## 日志变化

`train_projection_ppo_traj.py` 的 `[update ...]` 日志现在会记录：

- `wait_sig`：当前聚合后的绝对 wait signal，仅用于观察交通状态
- `wait_delta`：实际进入 stable reward 的 wait 变化量
- `rq / rw / rt`：queue、wait、travel 三个 reward 分量
- `qterm / wterm / aterm / thterm`：归一化后的 queue、wait、active、throughput term
- `raw / clip / mean_r`：裁剪前单步均值、裁剪后单步均值、rollout buffer 中的 reward 均值
- `ret_mean / ret_std / adv_std`：value target 和 advantage 的尺度

这样训练时可以直接观察：

- queue / wait 是否在按预期起作用
- travel proxy 是否真的进入了 PPO reward
- reward 是否被某一项单独主导
- value target 是否仍然发生尺度爆炸

## 新增文件

本次为了让 reward 逻辑更清晰，额外新增了两个文件：

### `projection_train/reward_shaping.py`

职责：

- 解析 objective 权重
- 对 queue / wait / travel 相对权重做归一化
- 统一组装单路口 shaped reward

### `tests/test_reward_shaping.py`

职责：

- 校验 objective weight 的归一化逻辑
- 校验负号 reward coefficient 会真实奖励指标下降
- 校验 `selection_*` 会真实改变 queue / wait / travel 的训练目标配比
- 校验 `stable_delta` 不会因为绝对 wait signal 变大而漂移
- 校验 `stable_delta` 的 reward clip 对极端输入生效

## 兼容性说明

### 1. 命令行参数名没有变

这次改动没有新增新的训练入口参数名，已有命令基本可以继续跑。

但需要注意：

- 旧命令里 `selection_*` 的含义已经扩大
- 它们现在不只是“checkpoint 选择权重”
- 还是“训练 reward 的相对目标权重”

### 2. 旧命令的 reward 方向可能会变化

如果你之前用了：

```bash
--queue_reward_coef -0.35

--waiting_reward_coef -0.01
```

在旧实现里，负号实际上会被吃掉。

在新实现里，这些负号会真正生效。

因此：

- 新旧版本即使命令相同
- 实际 reward 方向也可能不同

这属于预期行为变化，不是回归。

## 当前结论

`train_projection_ppo_traj.py` 现在的 Stage 3 PPO 训练逻辑已经从：

- “selection 权重只影响 best checkpoint”
- “reward 系数符号不生效”
- “travel proxy 没真正进入 shaped reward”

改成了：

- `selection_*` 同时影响训练 reward 和 checkpoint 选择
- `queue / wait / system_vehicle` 系数符号真实生效
- `throughput` 和 `travel proxy` 真实参与 PPO reward
- 训练日志能直接观察 reward 组成

这样 Stage 3 的超参语义会更一致，也更接近命令行配置本身表达的优化意图。
