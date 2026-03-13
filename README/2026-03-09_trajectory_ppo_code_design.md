# 2026-03-09 Trajectory PPO 代码实现思路

本文档记录这次在 `projection_train/` 下补齐 trajectory PPO 的写法思路，重点不是重复解释 PPO 教科书，而是说明这版代码为什么这样拆、如何尽量复用 Stage 1 / Stage 2 的稳定链路，以及每个新增文件具体承担什么职责。

## 目标

这次实现的目标很明确：

1. 不再把 Stage 3 做成单步 snapshot bandit
2. 改成共享策略的短轨迹 PPO
3. 保持 Stage 2 的输入语义不变
4. 尽量复用已有的 online inference / CityFlow / grouped dataset 代码

对应到统一表示链路，仍然保持：

`movement sequence -> ts_projector -> frozen vision tower -> vision_projector(prefix) -> frozen LLM -> 4-action logits`

这条链路在这次实现里没有被改写，Stage 3 只是把训练目标从离线动作监督扩展成了真实环境中的轨迹回报优化。

## 总体判断

### 1. Stage 3 不应该重新发明一套策略前向

如果 trajectory PPO 再单独写一套 actor / critic 编码器，表面上看好像更“RL 标准”，但实际上会立刻带来两个问题：

- Stage 2 warm start 失去意义
- Stage 2 学到的 prefix 语义和 Stage 3 在线优化语义再次分叉

所以这次实现的第一优先级不是“PPO 功能最全”，而是先把统一 policy 接口补齐。

这也是为什么先补了：

- `projection_train/prefix_action_policy.py`

它负责把 Stage 2、在线推理、trajectory PPO 三个入口收拢到同一个策略接口上。

### 2. 动作空间不能做 joint action

环境里每个控制步是“所有路口同时动作”，但训练时没有做 `4^N` 的 joint-action actor。

原因很直接：

- joint action 维度会随着路口数指数爆炸
- 当前 Stage 2 policy 本来就是单路口语义
- 共享参数、按路口展开样本，更符合现有代码结构

因此 rollout 的组织方式是：

- 环境里：所有路口同时执行动作
- buffer 里：把每个时刻、每个路口展开成一条 transition
- 参数上：所有路口共享同一套 policy

这就是“共享策略 + 多路口展开”的具体落地方式。

### 3. 第一版必须先做短轨迹

这次代码没有一上来就做 full episode on-policy training，而是默认：

- `rollout_steps = 8` 或 `16`
- 每个控制步推进 `MIN_ACTION_TIME=30s`

原因不是理论上不能做长轨迹，而是工程上先把下面几件事跑通更重要：

- snapshot reset 是否稳定
- history window 是否和 Stage 2 对齐
- PPO update 是否能从 Stage 2 checkpoint 平稳起步
- reference KL 是否能防止策略被在线训练拉崩

## 这次实现新增了什么

### 1. `projection_train/prefix_action_policy.py`

这是这次实现里最关键的缺口。

仓库原本的 Stage 2 脚本和 online inference 脚本都在引用 `PrefixActionPolicy`，但源码文件并不在当前工作树里，所以 trajectory PPO 真正开始之前，必须先把这个接口恢复出来。

这层封装的职责是：

- 挂载 `Qwen3VLBackbone`
- 挂载 `ts_projector`
- 挂载 `vision_projector`
- 可选挂载 `value_head`
- 对外提供统一的 actor / critic 接口

具体补齐的方法包括：

- `prepare_policy_batch`
- `forward`
- `action_distribution`
- `act`
- `predict_values`
- `evaluate_actions`
- `action_logprob`
- `load_checkpoint`

这里一个重要判断是：

不是让 LLM 自由生成一大段文本，再从文本里抽动作；而是直接对四个 `<signal>...</signal>` tag 计算对数概率，把四个动作打分堆成 logits。

这样和 Stage 2 的动作监督语义是一致的，也和 trajectory PPO 需要的 categorical actor 一致。

### 2. `projection_train/snapshot_seed_dataset.py`

这个文件不是普通 dataset，而是“snapshot 起点管理器”。

它做的事情是：

- 读取已有 recorded jsonl
- 按 `(traffic_file, snapshot_path)` 把记录重新聚合
- 收集同一个 snapshot 下所有路口的 `sequence / prompt / meta`
- 形成一个可供 `reset_from_snapshot()` 使用的 seed sample

之所以单独拆一个文件，是因为 snapshot seed 和普通 BC 样本的粒度不一样：

- BC 样本粒度是“单路口一个监督样本”
- snapshot seed 粒度是“同一个 snapshot 下的整张路网起点”

这两者不能混成一个 dataset，否则 reset 语义会很乱。

### 3. `projection_train/trajectory_env.py`

这个文件负责把真实 CityFlow episode 封成适合短轨迹 PPO 的环境层。

它没有复用旧的 `snapshot_env.py` 单步 `simulate_action()` 逻辑，而是直接对接：

- `CityFlowEnv.reset()`
- `CityFlowEnv.step(action_list)`

原因是这次要学的就是“连续控制”，不是“对一个 snapshot 猜单步 reward”。

这个 wrapper 里主要处理了四件事：

1. 构造 full-grid CityFlow 环境
2. 支持 `reset()` 和 `reset_from_snapshot()` 两种启动方式
3. 为每个路口维护最近 `seq_len` 步 movement history
4. 在 `step()` 里做 reward shaping

这里有一个非常关键的设计点：

snapshot reset 时不能只恢复引擎状态，还要恢复该 snapshot 对应的时间序列窗口。

否则 policy 在第一个 PPO 决策步看到的输入就会退化成全 0 或不完整窗口，和 Stage 2 / recorded 数据分布不一致。

所以 `trajectory_env.py` 里额外维护了：

- `history_buffers`

并且在 snapshot reset 时，把 recorded jsonl 里的那段历史直接塞回 buffer 里。

### 4. `projection_train/rollout_buffer.py`

这个文件的设计原则是尽量简单。

它只负责：

- 存 rollout 期间的 `sequence / prompt / action / reward / value / logprob / done`
- 做 GAE
- 展平 `T x N`
- 切 mini-batch

没有把环境逻辑和策略逻辑写进去。

原因是 trajectory PPO 最容易变乱的地方就是 buffer 和 env 相互耦合，一旦耦合，后面很难排查到底是 reward、value、还是 batch flatten 出错。

所以这里保持成纯数据层。

### 5. `projection_train/train_projection_ppo_traj.py`

这是这次新增的主入口。

它负责把前面四层串起来：

1. 从 Stage 2 checkpoint 初始化 policy
2. 额外挂上 value head
3. 冻结一份 reference policy
4. 构造 trajectory env
5. 按 snapshot 或 full reset 采样短轨迹
6. 用 GAE 计算 advantage / return
7. 执行 PPO update
8. 周期性跑 full episode eval
9. 保存 `last.pt / best.pt`

这版训练脚本里最重要的两个损失项是：

- 标准 PPO loss
- `KL(policy || reference policy)`

reference policy 直接来自 Stage 2 checkpoint，对应你的“不要把 Stage 2 policy 在线训练带崩”这个要求。

## 为什么 reward 这样写

这次没有另起一套复杂 reward，而是继续复用 CityFlow 里已有的 per-intersection reward，再做很轻量的 shaping：

`r_i = local_reward_i + alpha * global_mean_reward - beta * switch_penalty`

这几个项分别对应：

- `local_reward_i`
  - 直接来自 `CityFlowEnv.step()` 返回的每路口平均 reward
- `global_mean_reward`
  - 把全局均值作为一个很小的辅助项加进去
- `switch_penalty`
  - 当前动作和上一步动作不同就加一个很小惩罚

这里的核心思路不是 reward engineering 做到极致，而是先让策略能感受到：

- 连续放同一相位可能有长期价值
- 切相不是免费的
- 局部动作不能完全忽略全局状态

## 2026-03-13 补充：reward 改造成局部 delta-queue + delta-wait + starvation penalty

上面这一节描述的是 2026-03-09 第一版 trajectory PPO 的 reward 设计。后续根据实际训练日志和在线回放结果，这个设计已经被更新；如果你现在在看当前代码，reward 部分请以后面这节为准。

### 为什么要改

第一版 reward 的问题不是“PPO 不收敛”，而是“优化方向和最终评估不够一致”：

- `local_reward_i` 本质上仍然来自 `CityFlowEnv` 的绝对 queue 长度
- `global_mean_reward` / waiting / throughput 这些 shaping 里有不少是全局共享信号
- 在多路口共享策略训练下，这类共享信号会弱化单路口 credit assignment
- 实际日志里出现了相位偏科：主干相位占比过高、左转相位长期饥饿
- 结果就是 `queue` 或 `travel` 有时变好，但 `avg_wait` 被持续做坏

因此这次改动的核心目标不是“再加几个 reward 项”，而是把 reward 从“状态有多差”改成“这个动作是否让本路口状态变好”。

### 新 reward 的定义

当前 `projection_train/trajectory_env.py` 里，PPO 实际使用的 reward 已改成按路口计算：

`r_i = w_q * (queue_prev_i - queue_now_i) + w_w * (wait_prev_i - wait_now_i) - w_s * starvation_i - w_sw * switch_penalty_i`

对应含义：

- `delta-queue`
  - 看这个路口这一步把排队减少了多少
  - 如果 queue 从 20 降到 12，就是正奖励
  - 如果 queue 从 20 升到 28，就是负奖励
- `delta-wait`
  - 看这个路口这一步把等待时间拉低了多少
  - 这样 reward 会更直接对齐 `avg_wait`
- `starvation penalty`
  - 如果某个相位对应的 demand 明明持续存在，但连续多个决策周期都没被服务，就额外扣分
  - 它是专门用来抑制“只顾主干道、把某些 movement 长期饿死”的
- `switch_penalty`
  - 保留一个很小的切相代价，避免为了追逐局部 delta 而出现无意义抖动

### 这次具体改了什么

#### 1. 在 `trajectory_env.py` 里新增了按路口、按相位的拓扑与局部统计

为了把 starvation penalty 写对，不能只知道“这个路口现在总 queue 多大”，还必须知道：

- 哪些 lane 属于 `ETWT`
- 哪些 lane 属于 `NTST`
- 哪些 lane 属于 `ELWL`
- 哪些 lane 属于 `NLSL`

所以这次在环境层新增了：

- 路口 -> movement -> lane 的映射
- 路口 -> phase -> lane 的映射
- lane -> intersection / phase 的反查表

有了这些表之后，才可以在每一步里真正计算：

- 每个路口自己的 `queue_sum`
- 每个路口自己的 `waiting_time_mean`
- 每个 phase 当前对应的 `phase_queue_sums`
- 每个 phase 当前对应的 `phase_wait_means`

这也是为什么这次 reward 改造是放在 `trajectory_env.py`，而不是直接去改 `CityFlowEnv.get_reward()`。

#### 2. PPO 不再直接吃 `CityFlowEnv` 返回的绝对 queue reward

底层 `CityFlowEnv.step()` 依然会返回原来的 reward，但在当前实现里它只保留为诊断信息：

- `base_rewards`

真正写入 PPO buffer 的已经是新的 shaped reward：

- `delta_queue_reward`
- `delta_wait_reward`
- `starvation_penalty`
- `switch_penalty`

也就是说，现在 Stage 3 的 trajectory PPO 已经不再由“绝对 queue 长度”主导。

#### 3. starvation 被做成显式状态，而不是一次性惩罚

这次不是简单写一个：

- “左转 queue 大就罚”

而是维护了每个路口、每个 phase 的 starvation counter：

- 如果该 phase 当前 demand 不明显，counter 清零
- 如果该 phase 有 demand，但这一步没被选，counter 增加
- 超过阈值后，开始按 demand 强度和 overdue 步数扣分

这样做的原因是：

- 不是所有没选中的 phase 都算 starvation
- 只有“长期不服务且确实有需求”的 phase 才应该被惩罚

### 新增超参

这次在 `train_projection_ppo_traj.py` 里新增了下面这些 starvation 相关参数：

- `--starvation_penalty_coef`
- `--starvation_threshold_steps`
- `--starvation_queue_threshold`
- `--starvation_wait_threshold`

默认值分别是：

- `0.05`
- `2`
- `8.0`
- `45.0`

另外，`queue_reward_coef` 和 `waiting_reward_coef` 现在在 trajectory PPO reward 里按“幅度”使用，也就是只取绝对值。

### 这次改动对命令行参数的影响

当前 reward 主公式里，真正影响 PPO 的主要是：

- `queue_reward_coef`
- `waiting_reward_coef`
- `switch_penalty_beta`
- 四个 starvation 参数

而下面这些旧参数虽然接口还保留着，但已经不再主导 PPO reward：

- `reward_alpha`
- `system_vehicle_reward_coef`
- `throughput_reward_coef`

保留这些参数的原因主要是向后兼容，避免旧脚本直接报错；但如果是新一轮 trajectory PPO 实验，应该优先调的是 delta reward 和 starvation 相关参数。

### 训练日志也跟着更新了

为了让 reward 改造后的训练更容易读，这次 rollout 日志里额外加入了三项：

- `dq`
  - 平均 `delta_queue_reward`
- `dw`
  - 平均 `delta_wait_reward`
- `starve`
  - 平均 starvation penalty

因此你在训练时，除了继续看：

- `reward`
- `queue`
- `wait`
- `policy`
- `value`
- `kl`

还应该特别看：

- `dq` 是否大多数时间为正
- `dw` 是否不是长期为负
- `starve` 是否长期居高不下

### 这次改动后的验证状态

相对于 2026-03-09 第一版“只有结构检查”的状态，这次 reward 改造已经额外完成了：

- `projection_train/trajectory_env.py`
  - 可以独立 import
- `projection_train/trajectory_env.py`
  - 通过 `py_compile`
- `projection_train/train_projection_ppo_traj.py`
  - 通过 `py_compile`

但完整 import `train_projection_ppo_traj.py` 时，当前 `.venv` 仍然会碰到现有的 `tensorflow / numpy` 兼容问题。

这个问题不是这次 reward 改动引入的，而是当前运行环境本身的问题；如果要真正开始训练，仍然应该使用之前已经跑通过 trajectory PPO 的那套 Python 环境。

## 为什么 prompt 继续保持 `image_only_short`

trajectory PPO 里我没有把原始长 prompt 重新拉回来，而是继续用：

- `image_only_short`

原因和 Stage 2 一样：

- 不希望策略学文本捷径
- 不希望 trajectory PPO 又回到“长文本分析 -> 抽动作”的旧路径
- 要让 Stage 3 看到的输入分布尽量贴近 Stage 2

所以在环境观察里，每个路口给 policy 的 prompt 都是短 prompt，真正有信息量的输入仍然是 movement sequence 对应的 prefix 表示。

## 为什么 value 直接挂在 prefix 表示上

这次 critic 没有单独写第二套编码器，而是直接复用 actor 的 prefix 表示。

实现上就是：

- `prefix_vec -> value_head -> scalar value`

这样做的原因有三个：

1. 保持 Stage 3 和 Stage 2 的表示语义一致
2. 减少额外参数和额外训练不稳定性
3. 避免 critic 变成另一套和 actor 脱节的编码器

这和“Stage 3 不是重造模型，而是在同一 policy 语义上做轨迹回报优化”的定位是一致的。

## 训练流程如何闭环

当前这版代码的训练闭环可以概括成：

1. 加载 Stage 2 checkpoint 到 `PrefixActionPolicy`
2. 复制一份 frozen reference policy
3. 按概率选择：
   - snapshot reset
   - full reset
4. 在 CityFlow 里 rollout 若干控制步
5. 每步收集所有路口的 transition
6. 计算 GAE / returns
7. 展平成 `T x N` mini-batch
8. 做若干轮 PPO update
9. 周期性从 `t=0` 跑完整 episode eval
10. 保存 best / last

这里最重要的不是 PPO 算法本身，而是 rollout 看到的观测和 Stage 2 训练时看到的观测是一致的。

如果这点做不到，trajectory PPO 很容易变成“在线阶段其实在优化另一套输入分布”，那 warm start 的收益就会明显下降。

## 这版刻意没做什么

为了把第一版边界控制住，这次代码没有做下面这些事情：

- 多卡 PPO
- joint-action critic
- 八相位动作空间
- 长 episode on-policy 全量采样
- 图网络 credit assignment
- BC auxiliary batch 混训

这些方向后续都可以扩，但不应该在第一版里一起上。

因为当前最需要先确认的是：

- policy 接口统一后，Stage 2 checkpoint 能不能顺滑接 Stage 3
- trajectory env 和 history 恢复是不是稳定
- reference KL 能不能把在线训练控制在可接受范围内

## 代码层面的几个取舍

### 1. 先补接口，再补训练脚本

这次实现顺序上，先补 `PrefixActionPolicy`，再写 trajectory PPO。

原因是如果共享策略接口不先补齐，后面的 PPO 脚本一定会被迫把 actor、critic、checkpoint 兼容、tag logprob 这些细节重复实现一遍。

那样不仅代码重复，而且后面 Stage 2 / online inference / trajectory PPO 会再次分叉。

### 2. 环境层尽量不碰 `run_projection_online.py`

虽然 online inference 脚本里已经有一部分 history 恢复逻辑，但这次没有直接硬改原脚本，而是单独提炼 trajectory env。

这样做的好处是：

- 部署逻辑和训练逻辑不会互相污染
- PPO 需要的 snapshot reset / reward shaping 可以独立演进
- 线上测试脚本仍然保持部署用途，不被训练代码绑住

### 3. buffer 层保持纯数据角色

PPO 代码最容易写乱的地方，就是把 env 逻辑、loss 逻辑、buffer 逻辑揉到一起。

所以这里特意把 `rollout_buffer.py` 保持得很窄，只做数据收集和 GAE。

这样后面即使要改：

- reward
- GAE
- mini-batch 策略

也能单独改，不需要去碰 env 或 policy。

## 当前实现的限制

当前代码已经写完，但这次提交时有两个环境限制需要单独说明：

1. 当前 shell 没有可用的 Python 解释器
   - `python` 不存在
   - `py -3` 也不可用
2. 当前仓库有 `git safe.directory / ownership` 问题
   - 不能直接用正常的 `git status` / `git diff` 做收尾验证

所以这次只能做到：

- 文件级实现完成
- 结构和接口人工检查
- 基本的静态逻辑自查

还没有在这个环境里真正跑训练或 `py_compile`。

这意味着下一步最应该做的不是继续扩功能，而是先在有 Python / torch / transformers / cityflow 运行环境的机器上做一次最小 smoke test。

## 文件索引

这次新增的核心文件如下：

- `projection_train/prefix_action_policy.py`
- `projection_train/snapshot_seed_dataset.py`
- `projection_train/trajectory_env.py`
- `projection_train/rollout_buffer.py`
- `projection_train/train_projection_ppo_traj.py`

## 推荐阅读顺序

如果后续要继续改 trajectory PPO，建议按下面顺序看：

1. `projection_train/prefix_action_policy.py`
2. `projection_train/trajectory_env.py`
3. `projection_train/rollout_buffer.py`
4. `projection_train/train_projection_ppo_traj.py`

原因很简单：

- 先看策略接口，明确 actor / critic 到底是什么
- 再看环境，明确 rollout 到底采了什么
- 再看 buffer，明确数据怎么展平
- 最后再看 PPO 主循环，理解更新逻辑

## 总结

这次代码写法的核心不是“往项目里再塞一个 PPO 脚本”，而是先把 Stage 2 和 Stage 3 统一到同一个 policy 语义上，再在这个基础上把单步 snapshot bandit 推进成短轨迹 PPO。

换句话说，这次实现真正想解决的是两个问题：

1. Stage 2 学到的动作语义，能不能无缝进入 Stage 3
2. Stage 3 能不能在真实时序闭环里优化长期控制，而不是继续停留在单步近视 reward

所以这版 trajectory PPO 的重点不是追求功能最复杂，而是先把这条链路收拢成一套能演进、能验证、能继续扩展的实现骨架。
