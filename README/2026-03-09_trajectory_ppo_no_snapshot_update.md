# 2026-03-09 Trajectory PPO 去除 Snapshot 依赖的修改说明

这次调整的目标很直接：

- 训练可以只依赖 `traffic_file`
- 环境从 `t=0` 启动完整 episode
- PPO 每次只截取一小段 rollout 做更新
- 不再要求必须准备 `snapshot_jsonl` 和 `snapshot_traffic_file`

## 为什么要改

原来的训练主循环里，每个 update 开始前都会先做一次 reset。

如果传了 snapshot 数据，就会随机抽一个 snapshot 起点重新开始。
如果没传 snapshot 数据，就会直接执行：

- `train_env.reset(args.traffic_file)`

这在有 snapshot 的情况下问题不算大，因为起点本来就是分散的。
但在完全不使用 snapshot 的时候，会有一个很明显的问题：

- 每个 update 都从同一个 `traffic_file` 的 `t=0` 开始
- `rollout_steps` 又是短轨迹
- 结果就是训练反复采样 episode 前几分钟

以当前默认配置为例：

- `rollout_steps = 8`
- `min_action_time = 30`

一次 rollout 只覆盖 `240s`。如果每轮都 reset，那么 PPO 实际上长期盯着 episode 开头那一小段状态分布，后面的时段几乎看不到。

这不是我们想要的“短轨迹 PPO”，而是“每次都从头切一刀”的近似训练。

## 这次具体改了什么

改动点只在：

- `projection_train/train_projection_ppo_traj.py`

主循环从“每个 update 都 reset”改成了“只有 episode 结束时才 reset”。

现在的逻辑是：

1. 如果环境还没启动，或者上一个 episode 已经结束
2. 才决定这次是：
   - 用 snapshot reset
   - 还是用 `traffic_file` 做 full reset
3. 如果 episode 还没结束，就接着当前状态继续 rollout

这样一来，在不传 snapshot 参数时，训练流程会变成：

1. 用 `traffic_file` 从 `t=0` 启动一个 CityFlow episode
2. 采一段短 rollout
3. 做一次 PPO update
4. 继续沿着这个 episode 往后采下一段 rollout
5. 直到 episode 结束，再重新 reset

这更符合短轨迹 PPO 的本意。

## snapshot 现在是什么状态

这次没有把 snapshot 相关代码删掉，只是把它从“必须依赖”改成了“可选能力”。

也就是说：

- 不传 `snapshot_jsonl`
  - 训练完全可以正常运行
- 传了 `snapshot_jsonl`
  - 仍然可以在新 episode 开始时按概率使用 snapshot reset

保留这层兼容的原因很简单：

- 以后如果还想做更强的起点打散，不需要再把这套能力补回来
- 当前先解决“纯 `traffic_file` 在线训练”的主路径问题

## 现在三个参数怎么理解

### `traffic_file`

这是在线环境真正使用的流量文件。

如果你现在要训练济南 `3x4` 真实流量，核心参数就是：

- `traffic_root = data/Jinan/3_4`
- `traffic_file = anon_3_4_jinan_real.json`
- `roadnet_path = data/Jinan/3_4/roadnet_3_4.json`

### `snapshot_jsonl`

这是可选的 snapshot 种子索引文件。它不是 CityFlow 的流量文件，而是以前记录下来的 `traffic_prompt_series_2s.jsonl`。

如果不传，训练不会受影响。

### `snapshot_traffic_file`

这个参数只在启用 snapshot reset 时才有意义，用来限制“从哪一种 traffic file 对应的 snapshot 里采样”。

如果不使用 snapshot，这个参数可以直接忽略。

## 推荐用法

如果当前目标就是先把 PPO 跑通，而且只基于真实流量文件在线训练，那么建议直接这样用：

```powershell
python projection_train/train_projection_ppo_traj.py `
  --stage2_ckpt <你的stage2_ckpt> `
  --dataset jinan `
  --traffic_root data/Jinan/3_4 `
  --traffic_file anon_3_4_jinan_real.json `
  --roadnet_path data/Jinan/3_4/roadnet_3_4.json `
  --eval_traffic_file anon_3_4_jinan_real.json `
  --rollout_steps 8 `
  --num_updates 100
```

这条命令的含义很明确：

- 只使用 `anon_3_4_jinan_real.json`
- 不依赖 snapshot 数据
- 让 PPO 在同一个 episode 里连续前进

## 当前边界

这次改动解决的是训练流程上的一个偏差，不是把 snapshot 机制彻底移除。

所以当前状态是：

- `traffic_file` 单独就能跑
- snapshot 仍然兼容
- 代码结构改动很小，风险也比较可控

如果后面确认这条纯在线路径稳定，再决定要不要进一步删掉：

- `--snapshot_jsonl`
- `--snapshot_traffic_file`
- `SnapshotSeedDataset`

这会更稳一些。先把训练闭环跑顺，比现在就把所有分支都砍掉更重要。
