# Stage 1 精简数据集构建说明

本文说明 `scripts/build_stage1_compact_dataset.py` 的设计思路、适用范围和推荐使用方式。

## 1. 背景

第一阶段对齐训练使用的是 `(sequence, clip_caption)` 配对样本，而不是后续阶段更关注的 action 学习目标。

从代码上看：

- `train_align.py` 读取 `TrafficClipJsonlDataset`
- `TrafficClipJsonlDataset` 会把 jsonl 中“以非空 `clip_caption` 结尾的一组记录”构造成一个样本
- 每个样本的监督信号主要是：
  - 时序输入：最近 `seq_len` 步的 `movement_counts`
  - 文本监督：末尾记录上的 `clip_caption`

因此，Stage 1 可以单独使用一个更小、更干净、更均匀的子集，而不必完全沿用后续阶段使用的全量数据。

这个精简数据集只服务于 Stage 1 对齐训练，不替代 Stage 2 / Stage 3 数据。

## 2. 为什么需要精简

对 `data/align_train.cleaned.llm_caption.rebalanced.jsonl` 的检查显示：

- 原文件虽然有很多行，但真正用于 Stage 1 的是按 group 聚合后的样本
- 同一类交通模式存在较多重复序列
- caption 存在比较明显的模板化重复
- 数据分布对某些方向和模式存在偏置

如果直接把这类重复较高的数据全部用于第一阶段，通常会带来几个问题：

- 训练成本偏高，但有效信息增益有限
- 模型更容易记住高频模板表达
- 某些主导交通模式会压制长尾模式
- 对齐阶段更像“重复拟合主流描述”，而不是学习更稳的时序-语义对应关系

因此，脚本的目标不是简单随机采样，而是构造一个“更适合 Stage 1 对齐”的子集。

## 3. 脚本目标

`scripts/build_stage1_compact_dataset.py` 的目标可以概括为三点：

1. 去掉冗余重复样本
2. 保留语义信息更强的样本
3. 尽量维持覆盖面，而不是只保留高频模式

对应地，脚本采用了：

- 按完整时序窗口去重
- 基于信息密度打分
- 基于路口和交通属性做分桶选样

## 4. 核心思路

### 4.1 按 group 读取样本

脚本不会把 jsonl 每一行都当成独立样本，而是按与 `dataset.py` 一致的 group 方式读取：

- 连续读取记录
- 遇到非空 `clip_caption` 时，认为一个 group 结束
- 用这个 group 的最近 `seq_len` 步 `movement_counts` 构建一个样本

这样做的原因是，必须和 Stage 1 实际训练时的数据组织方式保持一致。

### 4.2 计算样本属性

对每个 group，脚本会计算一组结构化属性，用于后续筛选：

- `last_total`
  - 最后一步总流量
- `seq_total`
  - 整个窗口总流量
- `delta_total`
  - 首尾流量变化
- `variation_total`
  - 序列波动强度
- `ew_share`
  - 东西向流量占比
- `through_share`
  - 直行流量占比
- `dominance`
  - `east_west_dominant` / `north_south_dominant` / `balanced`
- `turn_balance`
  - `through_dominant` / `left_dominant` / `balanced`
- `action_margin`
  - 从 `expert_action_margin` 或 `candidate_action_waits` 估计动作区分度
- `caption_words`
  - caption 词数

这些属性并不是为了替代训练特征，而是为了给“哪些样本更值得保留”提供可解释的判断依据。

### 4.3 先按完整序列去重

脚本首先对完整 `seq_len x 8` 序列做精确去重。

原因：

- 许多样本在时序上完全相同或几乎相同
- 如果一个序列重复出现多次，继续全部保留对 Stage 1 的信息增益很小

去重后，只保留该重复序列中“分数更高”的那条样本。

这里的“分数更高”意味着它更可能是一个信息密度更高、描述更完整的训练样本。

### 4.4 为样本打信息分数

脚本不会只看流量大小，而是综合多个维度打分：

- 较高的 `last_total`
- 较高的 `variation_total`
- 较明显的 `abs(delta_total)`
- 较高的 `action_margin`
- 稍长一些的 `caption_words`

这些因素组合起来，是为了优先保留：

- traffic pattern 更明显的样本
- 时序变化更明显的样本
- 语义描述更完整的样本
- expert 决策区分度更清晰的样本

同时，对“低流量 + 低波动 + 稳定”的低信息样本会做轻微降权，避免它们占据过多预算。

### 4.5 建立 coarse bucket

为了避免精简后只剩“最常见、最明显”的模式，脚本会为每个样本构造一个粗粒度桶：

- `intersection_id`
- `dominance`
- `turn_balance`
- `volume_bin`
- `trend_bin`

其中：

- `volume_bin` 由 `last_total` 分成 `low / mid / high`
- `trend_bin` 由 `delta_total` 分成 `easing / stable / growing`

这个 bucket 的作用不是做严格分层统计，而是在选样时强制保留覆盖面。

### 4.6 两阶段选样

脚本的选样分两步：

第一步：

- 每个 coarse bucket 至少尝试保留 `min_per_bucket` 条
- 保证基础覆盖

第二步：

- 按总分从高到低继续补样，直到达到 `target_groups`
- 同时施加多个上限，避免某一类样本重新泛滥

## 5. 约束设计

为了避免“虽然去重了，但仍然被某些高频模板占满”，脚本加入了三个上限：

### 5.1 `per_caption_cap`

限制同一个标准化 caption 最多保留多少条。

作用：

- 压制模板化重复描述
- 减少模型只记语言模板而不看时序模式的风险

### 5.2 `per_bucket_cap`

限制同一个 coarse bucket 最多保留多少条。

作用：

- 防止某个高频模式占据过多样本
- 保证不同交通形态都有机会进入 Stage 1

### 5.3 `per_intersection_cap`

限制单个路口最多保留多少条。

作用：

- 防止精简后仍然由少数路口主导
- 让模型在 Stage 1 就见到更均匀的 intersection 变化

## 6. 为什么这是 Stage 1 专用方案

这个脚本的设计前提是：

- Stage 1 关注的是“时序交通状态”与“文字描述”的语义对齐
- 不要求保留所有后续阶段需要的 action 分布细节

因此它更偏向：

- 去重
- 去模板重复
- 保语义覆盖
- 保代表性

而不是尽可能保留所有样本。

对于 Stage 2 / Stage 3，不建议直接沿用这个思路，因为后续阶段更依赖：

- 动作标签分布
- action margin
- 更细的决策边界
- 与环境回报相关的样本密度

所以这个脚本明确只用于第一阶段。

## 7. 默认推荐参数

推荐先从以下参数开始：

```bash
python /home/code/LLMSTSC/scripts/build_stage1_compact_dataset.py \
  --input /home/code/LLMSTSC/data/align_train.cleaned.llm_caption.rebalanced.jsonl \
  --output /home/code/LLMSTSC/data/align_train.stage1_compact.balanced.jsonl \
  --seq_len 16 \
  --target_groups 2500 \
  --min_per_bucket 1 \
  --per_caption_cap 3 \
  --per_bucket_cap 18 \
  --per_intersection_cap 220
```

这组参数的意图是：

- 将 Stage 1 样本数从约 `6000` 压到约 `2500`
- 保留较高的唯一序列数
- 控制模板化 caption 重复
- 尽量让路口覆盖更均匀

## 8. 输出文件

脚本会生成两个文件：

### 8.1 精简后的 jsonl

即 `--output` 指定的文件，例如：

`data/align_train.stage1_compact.balanced.jsonl`

它仍然保持和原始训练数据兼容的 group jsonl 格式，可以直接喂给 `train_align.py`。

### 8.2 summary 文件

脚本还会额外生成：

`<output>.summary.json`

其中包含：

- 精简前统计
- 序列去重后统计
- 最终选样后统计

可以用来检查：

- 路口分布是否过斜
- action 是否过度偏移
- `low / mid / high` 流量是否保留合理
- caption 多样性是否下降过多

## 9. 调参建议

如果你希望子集更小：

- 降低 `target_groups`

如果你希望更多压缩重复语言：

- 降低 `per_caption_cap`

如果你希望分布更均匀：

- 降低 `per_bucket_cap`
- 降低 `per_intersection_cap`

如果你希望更多保留高信息样本：

- 保持较大的 `per_bucket_cap`
- 适当放宽 `per_intersection_cap`

如果你发现低流量样本太少：

- 提高 `target_groups`
- 降低路口上限约束的激进程度后，再检查 `volume_bin_counts`

## 10. 总结

`build_stage1_compact_dataset.py` 不是简单做随机抽样，而是在 Stage 1 目标下做了一次有偏好的样本重构：

- 用与训练一致的 group 方式读样本
- 先做完整序列去重
- 再按信息密度排序
- 再通过 bucket 和 cap 保证覆盖

它的核心目标是：

在不改动 Stage 1 训练接口的前提下，构造一个更适合做第一阶段语义对齐的小型数据集。
