# 2026-03-20 SimExpert 复合打分标签设计

本文档记录本次对 `run_sim_expert_action.py` / `utils/sim_expert_collector.py` 的修改思路。目标不是改 Stage 2 训练器，而是只改训练集生成链路，把 `expert_action` 的选择标准从原先的 `avg_wait` 单目标，切换成基于 `ATT / AQL / AWT / spillback` 的复合打分。

## 1. 改动目标

之前的仿真驱动 expert 采集流程里，每个 snapshot 下对 4 个候选相位做短时仿真，然后：

- 只计算 `avg_wait`
- 选择 `avg_wait` 最小的动作作为 `expert_action`
- 将每个候选动作的 `avg_wait` 写入 `candidate_action_waits`

这条链路足够简单，但它会让 Stage 2 学到的监督信号更偏向“短时平均等待最优”，而不是同时兼顾：

- `ATT`: average travel time
- `AQL`: average queue length
- `AWT`: average waiting time
- `spillback`: one-hop downstream spillback proxy

因此这次改动的目标是：

1. 保持 Stage 2 / grouped dataset / BC 训练代码不变
2. 只改 SimExpert 的候选动作打分逻辑
3. 让 `expert_action` 由 `ATT + AQL + AWT + spillback` 的复合目标决定
4. 继续输出兼容 Stage 2 的 `candidate_action_waits`

## 2. 总体原则

### 2.1 只改训练集生成，不改 Stage 2 训练器

当前 Stage 2 训练器只依赖两类字段：

- `expert_action`
- `candidate_action_waits`

其中：

- `expert_action` 用作 hard label
- `candidate_action_waits` 在 `target_mode != hard` 时用来构造 soft target

所以只要训练集仍然继续产出这两个字段，Stage 2 代码就可以不动。

### 2.2 保留旧字段名，内部语义升级

这次改动后：

- `candidate_action_waits` 这个字段名保留不变
- 但它内部不再一定表示纯 waiting
- 它现在表示“候选动作的复合 score，越小越好”

这样做的原因很直接：

- 避免改 `projection_train/grouped_dataset.py`
- 避免改 `projection_train/train_projection_bc.py`
- 避免把兼容性问题扩散到清洗脚本和后续训练脚本

同时，为了便于分析与回溯，新增：

- `candidate_action_metrics`

它会记录每个动作对应的：

- `att`
- `aql`
- `awt`
- `spillback`
- `score`
- 归一化结果
- 打分权重

## 3. 当前实现口径

### 3.1 AQL

`AQL` 定义为：候选短时仿真窗口内，目标路口进入车道的平均 queue length。

实现上按每个仿真秒记录目标路口进入车道上的等待车辆数，再取时间均值。

### 3.2 AWT

`AWT` 定义为：候选短时仿真窗口内，目标路口进入车道上等待车辆的平均 waiting time。

实现上维护一个局部 `waiting_vehicle_times` 表，对进入目标路口 entering lanes 的低速车辆累加等待时间，再对窗口内每个时刻的局部等待时间求平均。

### 3.3 ATT

这里的 `ATT` 不是完整 episode 结束后的全局 average travel time，而是一个**局部短窗口代理指标**：

- 只看目标路口 entering lanes 上的车辆
- 统计它们在候选仿真窗口内的进入时刻和离开时刻
- 若窗口结束时车辆仍未离开，则用窗口结束时刻截断

所以这里更准确地说，是：

- approach-level short-horizon travel time proxy

之所以采用这个口径，是因为当前 SimExpert 的标签采集本身就是“snapshot + 短时枚举候选”，不可能低成本拿到完整 episode 的真实 ATT。

### 3.4 Spillback

`spillback` 定义为：候选短时仿真窗口内，目标路口一跳下游出口车道的拥挤与堵塞风险代理指标。

实现上只看目标路口的 `exiting lanes`，原因是这些 lane 本身就是相邻路口的 entering lanes。相比直接把“整个邻居路口”的全局 queue 拉进来，这个口径更接近当前动作的直接因果影响，也更不容易把无关拥堵噪声混进标签。

具体会维护两条时间序列：

- `spillback_occupancy`: 出口车道车辆占用率的时间均值
- `spillback_blocked_ratio`: 出口车道中“高占用且已有等待车辆”的堵塞比例时间均值

最终：

```text
spillback = spillback_occupancy + 0.5 * spillback_blocked_ratio
```

## 4. 打分逻辑

当前实现固定使用同一个打分方案，不再保留 `wait-only` 或 `none-norm` 分支。

对同一个 snapshot 下的 4 个候选动作：

1. 分别算出每个动作的 `ATT / AQL / AWT / spillback`
2. 在 4 个动作内部对四项指标分别做 min-max 归一化
3. 计算：

```text
score = w_att * norm(ATT) + w_aql * norm(AQL) + w_awt * norm(AWT) + w_spillback * norm(spillback)
```

4. 选择 `score` 最小的动作作为 `expert_action`

### 4.1 为什么固定做归一化

如果直接对原始 `ATT + AQL + AWT + spillback` 相加，会有两个明显问题：

1. 四项量纲不同
2. 四项数值范围经常差很多，容易导致其中一项天然主导 score

因此这里直接固定使用 `minmax`，先在“同一个 snapshot 的 4 个候选动作内部”做相对比较，再聚合成最终分数。

## 5. 本次新增参数

`run_sim_expert_action.py` 新增了下面这些参数：

- `--expert_att_weight`
- `--expert_aql_weight`
- `--expert_awt_weight`
- `--expert_spillback_weight`

默认值都是：

```bash
--expert_att_weight 1.0
--expert_aql_weight 1.0
--expert_awt_weight 1.0
--expert_spillback_weight 0.2
```

## 6. 代码改动点

### 6.1 `run_sim_expert_action.py`

职责：

- 暴露必要的复合打分权重参数
- 将这些参数写入 `dic_agent_conf_extra`

### 6.2 `utils/sim_expert_collector.py`

职责：

1. 在 `_simulate_candidate()` 中返回 `ATT / AQL / AWT / spillback` 多指标结果，而不再只返回 `avg_wait`
2. 在候选 rollout 内新增下游 `exiting lanes` 的 spillback 统计
3. 新增 `_score_candidates()`，负责固定的 min-max 归一化、加权和 tie-break
4. 将：
   - `candidate_action_waits`
   - `candidate_action_metrics`
   - `expert_action_margin`
   一起写入训练集记录

### 6.3 `utils/llm_aft_trainer.py`

本次无需改动。它已经会把 `candidate_action_metrics` 作为任意字典透传到 jsonl 最后一行。

## 7. 为什么 Stage 2 不需要改

Stage 2 当前训练逻辑是：

1. 从 grouped dataset 读取 `expert_action`
2. 用 `expert_action` 做交叉熵
3. 若 `target_mode != hard`，则读取 `candidate_action_waits`
4. 将其作为“候选动作代价”构造成 soft target

注意这里并没有要求这个字段一定是“真实 waiting time”。

它只要求：

- 是一个 `action -> float` 的映射
- 越小越好
- 值之间的相对大小能反映动作优劣

因此把 `candidate_action_waits` 从“纯 waiting”升级成“复合 score”后，Stage 2 的 hard label 和 soft label 仍然都能正常工作。

## 8. 需要注意的副作用

### 8.1 `expert_action_margin` 的含义变了

现在的 `expert_action_margin` 不再表示“候选 waiting 的前二名差距”，而是：

- 候选复合 score 的前二名差距

这意味着如果你继续使用：

```bash
python utils/clean_align_jsonl.py --min_action_margin ...
```

那么这个阈值的数值含义已经改变了，可能需要重新校准。

当前更适合作为起点的经验值是：

- `--min_action_margin 1.0`：偏保守，通常能过滤掉一批过于模糊的样本
- `--min_action_margin 1.5`：更激进，适合做一版高质量对齐子集

### 8.2 ATT 仍然是代理指标

当前实现中，`ATT` 是短时局部代理，而不是完整仿真结束后的全局 ATT。

这不是 bug，而是当前训练集采集范式的边界：

- 现在是 snapshot-based candidate scoring
- 不是 full-episode expert search

如果未来要把 `ATT` 做成更严格的全局定义，需要把候选动作枚举从“短时局部仿真”提升到“更长 horizon 甚至完整 episode 评估”，采集成本会明显上升。

### 8.3 Spillback 也是一跳代理指标

当前实现中的 `spillback` 只看目标路口的一跳下游出口车道，而不是整个邻居路口甚至更远路网的全局拥堵。

这同样不是 bug，而是有意控制的因果范围：

- 只惩罚当前动作直接可能加压的下游 link
- 避免把邻居路口其他无关 approach 的拥堵噪声混进来
- 保持候选 rollout 的额外计算成本足够低

## 9. 推荐命令

如果你希望启用本次复合打分逻辑，推荐命令如下：

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --expert_att_weight 1.0 \
  --expert_aql_weight 1.0 \
  --expert_awt_weight 1.0 \
  --expert_spillback_weight 0.2
```

## 10. 当前结论

这次改动的核心不是“重写 Stage 2”，而是：

- 保持 Stage 2 的输入协议不变
- 只提升训练集标签的优化目标

也就是说：

1. Stage 2 训练器不需要改
2. 训练集生成逻辑已经升级到了 `ATT / AQL / AWT / spillback` 复合打分
3. `candidate_action_waits` 仍然保留，用于兼容当前 BC 训练
4. `candidate_action_metrics` 则提供更完整的分析信息，包括一跳下游 spillback 代理
