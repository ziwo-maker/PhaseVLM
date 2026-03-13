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

## `projection_train/train_projection_ppo_traj.py` 的整体设计思路

这份脚本的核心目标，不是从零训练一个 traffic signal policy，而是把已经做过 `stage2` 监督学习的 projection 模型，当作 PPO 的初始化策略，然后在真实 CityFlow 环境里继续做在线强化学习微调。

可以把它理解成 5 个层次：

### 1. 启动层：先从 `stage2_ckpt` 恢复运行时配置

脚本启动后，先做几件基础工作：

- 解析命令行参数
- 从 `stage2_ckpt` 里恢复 `llm_path`、`seq_len`、`image_size`、`prefix_tokens` 等运行时超参
- 根据 `dataset` 自动补齐 `roadnet_path`、`traffic_root`、`run_counts`
- 初始化随机种子、device、DDP/FSDP、AMP

这一步的设计意图很明确：

- PPO 训练不是孤立脚本，而是直接接在前面的 projection 训练结果之后
- 只要 checkpoint 里保存过训练参数，PPO 阶段就不需要用户重复手填一大串模型配置

### 2. 策略层：用 stage2 投影模型做 actor，再额外挂一个 value head

真正的策略网络是 `PrefixActionPolicy`。它内部不是普通的 MLP policy，而是三段式结构：

- `Qwen3VLBackbone`
  - 提供冻结的大模型 backbone
- `TimeSeriesProjector`
  - 把每个路口最近一段交通计数序列编码成“伪图像”
- `VisionPrefixProjector`
  - 再把视觉特征映射成给 LLM 使用的 prefix tokens

然后脚本构造了两个 policy：

- `policy`
  - 用来真正采样动作和做 PPO 更新
  - 带 `value_head`
- `reference_policy`
  - 只做 reference logits，不参与训练
  - 和 `policy` 共享同一个 backbone

这里的设计很关键：

- actor 并不是直接输出离散 action logits
- 它是把“交通时序 + prompt”拼成 LLM 输入，然后计算每个 `<signal>...</signal>` action tag 的概率
- value 函数不走 LM token head，而是直接从 prefix 表征上接一个独立的 value head

所以它本质上是在做：

- 用 stage2 学到的 prefix 表征初始化 RL policy
- 训练时主要更新 projector 和 value head
- 用 reference policy 约束当前策略不要偏离 stage2 初始分布太远

### 3. 环境层：`TrajectoryCityFlowEnv` 把 CityFlow 包成“短轨迹、多路口并行”的 PPO 环境

环境封装并不是直接裸调 `CityFlowEnv`，而是额外做了几层适配：

- 每个路口维护一个固定长度的历史序列 `history_buffers`
- 每次 `step` 后按 `traffic_series_interval` 从环境统计里追加新的 movement count
- `observe()` 返回的是：
  - 每个路口的时序序列
  - 对应的 prompt
  - 路口 id
  - 当前仿真时间
  - 当前 reset 来源（full / snapshot）

这意味着训练视角是：

- 一个 CityFlow episode 里有很多 intersection
- 每个 intersection 在同一时刻都视为一个并行样本
- policy 每次输出的是“所有路口当前该切到哪个 signal phase”

另外，奖励不是直接照搬环境原始 reward，而是做了轻量 shaping：

- 基础 reward 来自 `CityFlowEnv.step()`
- 再加一项 `reward_alpha * global_mean_reward`
  - 让局部决策对全局交通状态更敏感
- 再减一项相位切换惩罚 `switch_penalty_beta`
  - 避免策略无意义频繁切灯

所以这里的设计目标不是追求复杂 reward engineering，而是让 PPO 在稳定性和交通控制常识之间先取得一个够用的平衡。

### 4. 数据流层：每个 update 只收集短 rollout，但 GAE/ppo_update 都是标准 PPO 结构

单次训练更新的流程是：

1. 从当前环境状态拿到所有路口的 `sequences + prompts`
2. `policy.act()` 采样动作，并得到：
   - actions
   - log_probs
   - values
3. 同时用 `reference_policy` 前向一次，缓存 reference logits
4. 环境执行一步，得到 reward / done / traffic metrics
5. 把这一批 transition 写入 `TrajectoryRolloutBuffer`
6. 收集到 `rollout_steps` 后，用最后一个状态的 value 做 bootstrap
7. 用 GAE 计算 `advantages` 和 `returns`
8. 展平 buffer，按 mini-batch 做若干轮 PPO update

`TrajectoryRolloutBuffer` 的实现也是按这个设计来的：

- 存的是整段 trajectory，而不是单步样本
- `compute_returns_and_advantages()` 反向计算 GAE
- `flattened()` 再把“时间维 x 路口维”拉平成 PPO mini-batch

也就是说，这个脚本虽然环境是多路口联合推进的，但优化视角仍然是标准 PPO：

- rollout 收集
- GAE
- clipping policy loss
- clipped value loss
- entropy bonus

### 5. 约束层：PPO 不是完全自由更新，而是带 reference KL 的“受限微调”

这份实现和教科书 PPO 最大的区别，是它额外加了一项 KL：

- 当前策略 logits vs `reference_policy` logits
- 损失项为 `args.kl_coef * kl`

因此 `ppo_update()` 优化的是：

- policy loss
- value loss
- entropy bonus
- KL regularization

这样做的实际含义是：

- 允许在线 RL 去修正 stage2 监督策略
- 但不希望策略一下子漂得太远，导致语言模型动作分布崩掉

这和它的初始化方式是配套的：

- 起点来自监督训练
- 在线阶段是 constrained fine-tuning
- 而不是从随机策略开始探索

## 把这些模块串起来后，一次完整训练长什么样

如果按代码路径顺着看，一次完整训练基本就是：

1. 读 checkpoint，恢复 projection policy 运行时配置
2. 构建 `policy` / `reference_policy`
3. 构建 `train_env` / `eval_env`
4. 可选加载 `SnapshotSeedDataset`
5. 进入 update 循环
6. 如果 episode 结束了才 reset；否则继续沿当前 episode 向前 rollout
7. 收集短轨迹，做一次 PPO 更新
8. 定期在完整 episode 上做 evaluation
9. 保存 `last.pt`，如果评估更优则保存 `best.pt`

所以这份脚本的整体设计可以概括成一句话：

- 用 stage2 projection policy 作为初始化
- 用 CityFlow 在线交互产生短轨迹
- 用带 KL 约束的 PPO 做稳定微调
- 用“episode 内连续推进、只在结束时 reset”的方式避免长期只训练在 `t=0` 附近

这也是为什么这次“去掉 snapshot 强依赖”的修改虽然代码改动不大，但意义比较大：

- 它没有改变这套 PPO 训练框架的主体结构
- 改的只是 rollout 起点管理方式
- 但这个点正好决定了训练到底是在“连续轨迹上学习”，还是在“反复采样 episode 开头”

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
