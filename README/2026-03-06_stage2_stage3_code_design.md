# 2026-03-06 Stage 2 / Stage 3 代码思路

本文档记录本次围绕 `data/recorded/align_train.cleaned.jsonl` 的 Stage 2 / Stage 3 重写思路，重点解释为什么这样拆、各模块分别负责什么，以及训练时的数据和梯度如何流动。

## 目标

当前整体路线固定为：

1. Alignment 预训练
2. Stage 2: 动作监督 SFT
3. Stage 3: 在线 PPO

这次重写的核心目标不是“把脚本能跑起来”而已，而是把 Stage 2 和 Stage 3 收敛到同一条表示链路上，避免之前 BC 和 PPO 走两套前向、导致 warm start 语义不一致。

最终统一成：

`time series -> ts_image_adapter -> frozen Qwen vision tower -> vision_projector(prefix) -> frozen LLM -> 4动作分布`

其中只有下面两部分训练：

- `ts_image_adapter`
- `vision_projector`

LLM 本体和 vision tower 冻结。

## 设计判断

### 1. 数据不是逐行样本

`align_train.cleaned.jsonl` 不是一行一个样本，而是固定 `16` 行构成一个样本组：

- 前 `15` 行主要提供 `movement_counts`
- 第 `16` 行提供监督字段：
  - `prompt`
  - `clip_caption`
  - `expert_action`
  - `snapshot_path`

所以 Stage 2 / Stage 3 都必须先做 grouped dataset，不能直接把 jsonl 按行喂给 dataloader。

### 2. 禁用原始长 prompt 作为主输入

原始 `prompt` 里包含大量文本态交通状态描述，模型很容易学文本捷径，而不是看由时间序列生成的图像。

因此默认使用 `image_only_short`：

- 只保留任务定义
- 只保留 4 个合法动作 tag
- 明确要求只输出一个 `<signal>...</signal>`

这样能把训练压力集中回图像表示。

### 3. 以真实第一阶段配置为准

虽然数据里的遗留 `meta/prompt` 还写着旧的 `15 steps / size 56 / periodicity 24`，但当前代码以真实第一阶段配置为准：

- `seq_len=16`
- `image_size=224`
- `periodicity=16`
- `prefix_tokens=16`

这样 Stage 2 / 3 的 adapter 语义才和 alignment checkpoint 保持一致。

### 4. `projection_train/snapshot_env.py`

职责：

- 从 `snapshot_path` 恢复 CityFlow 引擎状态
- 从 snapshot 中读取当前信号灯 phase
- 结合 `roadnet_path + traffic_root + traffic_file` 构造 full-grid 环境
- 对某一路口某个动作做短时仿真并返回 reward

关键点：

- 不再依赖旧脚本里零散的环境恢复逻辑
- 直接把 Stage 3 的“动作评估”抽成独立模块

当前 reward 定义：

- 统计目标路口入口车道上速度低于阈值的车辆
- 计算短时平均等待时间
- 用其负值作为 reward

## Stage 2 思路

入口：

- `projection_train/train_projection_bc.py`

训练流程：

1. 从 grouped dataset 读取样本
2. 用 `image_only_short` 构造短任务 prompt
3. `PrefixActionPolicy` 输出 4 个动作 logits
4. 用 `expert_action` 做交叉熵监督
5. 只更新：
   - `ts_image_adapter`
   - `vision_projector`

为什么不再做旧版 `0-3` 数字分类：

- 旧版和 Stage 3 的动作接口不一致
- 现在直接学 `<signal>ETWT</signal>` 这种动作 tag，更贴近最终推理形式

训练指标：

- `loss`
- `acc`
- `macro_f1`

之所以保留 `macro_f1`，是因为动作分布不均衡，单看 accuracy 不够。


## 为什么 Stage 2 和 Stage 3 现在是同构的

这是这次改造最重要的部分。

之前的问题是：

- Stage 2 训练的并不是 Stage 3 真正会用到的动作前向
- Stage 3 warm start 只能“勉强加载参数”，但前向语义不完全一致

现在两者共用：

- 同一个 grouped dataset 结构
- 同一个 prompt 约束
- 同一个动作空间
- 同一个 prefix-action policy
- 同一个 adapter / projector 参数

因此：

- Stage 2 学到的是“如何让 prefix 表示支持动作决策”
- Stage 3 继续在同一个动作头语义上做 reward 优化

这才符合 `Alignment -> SFT -> RL` 的预期链路。

## 设备策略

### Stage 2

优先支持：

- 单卡 A800
- 多卡 `torchrun`

因为 Stage 2 是纯监督，天然更适合先扩到多卡。

### Stage 3

当前优先支持：

- 单卡 A800

原因：

- 在线仿真本身就比较重
- snapshot 恢复 + CityFlow 仿真更适合先把单卡逻辑跑稳
- 多卡 PPO 后续再扩，否则排查问题成本太高

## 当前实现的边界

当前已经实现：

- 统一 grouped dataset
- 统一 prefix-action policy
- Stage 2 监督训练入口
- Stage 3 单卡在线 PPO 入口
- snapshot 驱动的短时 reward 评估

当前还没有做的事情：

- Stage 3 多卡 DDP
- 更长 horizon 的轨迹式 PPO
- 按路口留出的正式泛化评估脚本
- 端到端 smoke 运行验证

最后一点尤其重要：当前终端环境缺少 `torch` 运行依赖，所以这次只完成了代码重构和 `py_compile` 级别静态校验，还没有在这个 shell 内直接跑训练。

## 文件索引

- `projection_train/action_space.py`
- `projection_train/grouped_dataset.py`
- `projection_train/prefix_action_policy.py`
- `projection_train/snapshot_env.py`
- `projection_train/train_projection_bc.py`


## 推荐阅读顺序

如果后续要继续改，建议按这个顺序读代码：

1. `grouped_dataset.py`
2. `prefix_action_policy.py`
3. `train_projection_bc.py`
4. `snapshot_env.py`


这样最容易先把数据流看明白，再看监督和 RL 两个入口如何共享同一策略前向。
