# 2026-03-09 Stage 2 阶段成果汇报

## 1. 整体训练流程说明

本项目当前整体路线分为三个阶段：

### Stage 1: 时间序列到图像的表征对齐

Stage 1 的目标是把交通时间序列数据转换为适合视觉模型处理的图像表征，并完成基础对齐。这里主要通过 `CLIP` 风格的对齐方式，让时间序列编码结果能够和后续视觉语言模型的输入空间衔接起来。

这一阶段的作用是建立统一的输入表示链路，为后续动作监督和在线强化学习提供稳定的特征基础。

### Stage 2: 动作监督训练（SFT）

Stage 2 的目标是在已经完成表征对齐的基础上，利用专家动作对模型进行监督训练，让模型学会根据时间序列图像表征直接输出交通信号动作。

这一阶段的作用是先通过离线监督方式，让模型具备基础决策能力，并验证时间序列图像表征是否足以支撑交通控制任务。

### Stage 3: 在线强化学习优化（PPO）

Stage 3 的目标是在 Stage 2 的基础上进一步引入在线强化学习，通过 `PPO` 在仿真环境中继续优化策略。

这一阶段的作用是把模型从“会模仿专家动作”进一步推进到“能够结合环境反馈持续优化决策”，从而争取获得更高的控制收益。

## 2. 本次实验范围

本次结果仅覆盖 Stage 2：

- 提示词由 `Gemma-27B` 基于时间序列数据自动生成，重点描述交通状态中的波峰、波谷等变化特征。
- 时间序列到图像表征的转换采用 `CLIP` 对齐与 `SFT` 动作监督训练。
- 本次实验**没有使用第三阶段的 PPO**，因此结果反映的是离线表征学习与监督决策带来的收益。

## 3. 数据设置与 zero-shot 说明

本次训练集使用的是**济南数据**，测试集使用的是**杭州数据**。也就是说，模型并不是在杭州数据上继续训练后再测试，而是在跨城市场景下直接进行评估，因此这里的结果属于 **zero-shot** 泛化结果。

从实验结果来看，即使在 zero-shot 设置下，`Qwen-7B + Projection` 依然在奖励、排队长度、排队车辆数、平均等待时间和平均旅行时间等指标上取得了稳定改进，说明该方法已经具备较强的跨场景迁移能力。

## 4. 提示词设计对比

当前代码中实际存在两种提示词模式：

- 长提示词：`raw`
- 短提示词：`image_only_short`

其中，长提示词对应 `run_open_LLM.py` 调用链中 `getPrompt(state2text(...))` 生成的原始任务模板；短提示词对应 [prompt_utils.py](/home/code/LLMSTSC/projection_train/prompt_utils.py) 中的 `image_only_short` 模式。

### 长提示词示例（英文原始模板，依据 `run_open_LLM.py` 调用链）

```text
You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks.
IMPORTANT:
- The attached image is the ONLY reliable observation of the intersection state for the last 30 seconds.
- Do NOT assume the traffic numbers in text are correct. If any text conflicts with the image, trust the image.
- You MUST look at and use the image to make the decision.

### What the image encodes (read carefully)
The image encodes a 30-second time series sampled every 2 seconds (15 time steps) for 8 lanes at one intersection.
- X-axis: time from oldest (left) to most recent (right).
- Y-axis: lanes in this fixed order:
1) East Through (ET)
2) West Through (WT)
3) North Through (NT)
4) South Through (ST)
5) East Left (EL)
6) West Left (WL)
7) North Left (NL)
8) South Left (SL)
- Pixel intensity (or color strength): larger means more vehicles in that lane at that time.
Interpret the rightmost columns as the latest demand. Consider both the current magnitude and recent trend (growing vs shrinking).

### Instruction:
A traffic light regulates a four-section intersection with northern, southern, eastern, and western sections...
The traffic light has 4 signal phases...
Please answer:
Which is the most effective traffic signal that will most significantly improve the traffic condition during the next phase?

Requirements:
- Let's think step by step.
- You can only choose one of the signals listed above.
- You must follow the following steps to provide your analysis: Step 1: Provide your analysis for identifying the optimal traffic signal. Step 2: Answer your chosen signal.
- Your choice can only be given after finishing the analysis.
- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>.
```

### 长提示词中文说明

```text
你是一名交通管理专家，需要利用交通常识来完成交通信号控制任务。
输入中附带的图像是过去 30 秒路口状态的唯一可靠观测，如果图像与文本数字冲突，应以图像为准。
图像编码了 8 个车道在 30 秒内的时间序列变化，需要同时考虑当前车辆规模和最近的增长/下降趋势。
随后提示词会详细说明四相位信号、每个相位对应疏导的车道、排队车辆数量、不同路段上的来车情况，并要求模型分步骤分析，最终只能输出一个带有 <signal>...</signal> 标签的信号相位。
```

### 短提示词示例（英文原始模板）

```text
You are controlling a four-phase traffic signal.
The attached image is the only reliable observation of recent traffic dynamics at one intersection.
Choose the single best next phase to reduce near-term congestion.
Allowed outputs: <signal>ETWT</signal>, <signal>NTST</signal>, <signal>ELWL</signal>, <signal>NLSL</signal>.
Return exactly one <signal>...</signal> tag and nothing else.
```

### 短提示词中文说明

```text
你正在控制一个四相位交通信号灯。
附带图像是该路口近期交通动态的唯一可靠观测。
请从四个合法相位中选择一个最优动作，以尽可能缓解短时拥堵。
输出必须且只能是一个 <signal>...</signal> 标签，不能附带额外分析文本。
```

### 对比说明

长提示词包含了较完整的交通规则说明、车道定义、分段车辆统计和回答步骤约束，信息更丰富，更接近传统 LLM 推理方式。短提示词则只保留任务目标和合法动作集合，把决策压力更多地压回到时间序列图像表征本身，更适合当前的图像投影训练路线。

## 5. 对比设置

对比对象包括：

- `Gemma-27B`
- `Qwen-7B + Projection`

评估指标包括：

- `Reward`
- `Avg. Queue Length`
- `Queuing Vehicles`
- `Avg. Waiting Time`
- `Avg. Travel Time`

其中，`Reward` 越大越好；其余排队与通行指标越小越好。

## 6. 实验结果

| Dataset | Model | Reward | Avg. Queue Length | Queuing Vehicles | Avg. Waiting Time | Avg. Travel Time |
| :-- | :-- | --: | --: | --: | --: | --: |
| hangzhou_real_5816.json | Gemma-27B | -1417.3911 | 950.4417 | 114053 | 732.6571 | 742.9948 |
| hangzhou_real_5816.json | Qwen-7B + Projection | -1395.3796 | 887.7583 | 106531 | 540.4676 | 701.6153 |
| hangzhou_real.json | Gemma-27B | -1478.8355 | 798.4417 | 95813 | 880.7846 | 1124.8419 |
| hangzhou_real.json | Qwen-7B + Projection | -1440.5432 | 670.4250 | 88451.8 | 623.2206 | 975.0054 |

## 7. 结果分析

从两组杭州真实路网数据可以看到，`Qwen-7B + Projection` 在所有核心指标上都优于 `Gemma-27B`，表现出稳定一致的提升趋势。

### `hangzhou_real_5816.json`

- `Reward` 提升约 `1.55%`
- `Avg. Queue Length` 降低约 `6.60%`
- `Queuing Vehicles` 降低约 `6.60%`
- `Avg. Waiting Time` 降低约 `26.23%`
- `Avg. Travel Time` 降低约 `5.57%`

### `hangzhou_real.json`

- `Reward` 提升约 `2.59%`
- `Avg. Queue Length` 降低约 `16.03%`
- `Queuing Vehicles` 降低约 `7.68%`
- `Avg. Waiting Time` 降低约 `29.24%`
- `Avg. Travel Time` 降低约 `13.32%`

值得强调的是，上述结果是在**济南训练、杭州测试**的 zero-shot 条件下取得的。这说明当前方法并不只是对训练城市有效，而是在跨城市、跨交通流分布的情况下，仍然展现出了不俗的泛化性能。

## 8. 参数量与训练资源对比设想

### 当前 Projection 训练的参数量

当前 Stage 2 方案中，真正参与训练的模块主要是两部分：

- `ts_projector`
- `vision_projector`

其中，`ts_projector` 的参数量可以直接从代码结构精确计算出来。在当前默认配置 `input_dim=8`、`hidden_dim=64` 下：

- `trend_dw`: `8 x 1 x 5 = 40`
- `conv1d`: `64 x 3 x 3 + 64 = 640`
- `conv2d_1`: `32 x 64 x 3 x 3 + 32 = 18464`
- `conv2d_2`: `3 x 32 x 3 x 3 + 3 = 867`

因此，`ts_projector` 总参数量约为 **20,011**。

`vision_projector` 的参数量主要由下面这个线性层决定：

`vision_dim -> prefix_tokens x text_dim`

因此其参数量公式为：

`vision_dim x (prefix_tokens x text_dim) + (prefix_tokens x text_dim)`

如果按照当前实验中常用的 `prefix_tokens=16`，并以 **7B 级底座模型**做一个资源估算，假设：

- `text_dim ≈ 4096`
- `vision_dim ≈ 1280 ~ 1536`

那么 `vision_projector` 的参数量大约在：

- **84.02M ~ 100.73M**

再加上 `ts_projector` 的约 `0.02M`，当前 Projection 方案的总可训练参数量大约可以认为是：

- **84M ~ 101M**

这说明当前方案虽然冻结了 LLM 主体，但外部的 prefix projection 本身并不小，参数量的主体其实集中在 `vision_projector` 上。

### 如果采用 LoRA 训练的参数量估算

如果对每层的 `q_proj / k_proj / v_proj / o_proj` 注入 LoRA，并假设：

- 层数约为 `32`
- hidden size 约为 `4096`
- LoRA rank 分别取 `r=8` 或 `r=16`

那么单个线性层的 LoRA 参数量约为：

`r x (in + out)`

对 `4096 -> 4096` 的线性层来说，就是：

- `r=8` 时，每层每个线性层约 `65,536`
- `r=16` 时，每层每个线性层约 `131,072`

进一步按 `q/k/v/o` 四个投影层、`32` 层估算：

- `LoRA r=8`：约 **8.39M**
- `LoRA r=16`：约 **16.78M**

如果只对 `q_proj / v_proj` 做 LoRA，那么参数量还会进一步减半；如果再把 `MLP` 里的 `gate/up/down` 一起纳入 LoRA，参数量则会继续上升。

### 参数量对比结论

从参数量规模上看，当前的 `Projection` 方案在 7B 级设定下，很可能会比常见 attention-only LoRA 训练更多参数：

- `Projection`：约 **84M ~ 101M**
- `LoRA (r=8)`：约 **8.39M**
- `LoRA (r=16)`：约 **16.78M**

也就是说，Projection 方案的训练参数量并不一定比 LoRA 更小，它的优势主要不在“参数更少”，而在于**训练位置更集中、接口更清晰、保持主干 LLM 完全冻结**。

### 时间对比与显卡对比设想

下面这部分是基于当前代码结构给出的**资源设想**，还不是实测结论。

| 方案 | 可训练参数量 | 训练位置 | 时间设想 | 显卡设想 |
| :-- | :-- | :-- | :-- | :-- |
| Projection | 约 84M ~ 101M | 主要集中在 `ts_projector + vision_projector` | 单步训练预计更稳定，工程链路更直接；在相同 batch 下，整体训练时间有希望与 LoRA 持平或略快，但需要实测确认 | 2 x 4090 或 1 x A800 起步较现实；若继续放大 batch 或上下文长度，建议 2 x A800 |
| LoRA | 约 8M ~ 17M（attention-only） | 分散注入到每一层 Transformer 内部 | 虽然参数更少，但训练点分散在整网层内，通信和实现复杂度更高；单步未必一定更快，分布式下可能接近甚至略慢于 Projection | 7B 底座下可尝试 2 x 4090 / 2 x A800；若换成 27B 底座，通常需要 4 x A800 甚至更高配置 |

## 9. 方案优势

除了当前实验结果本身，Projection 路线还有几个比较明确的工程优势。

1. 训练数据更容易大规模生成。`caption` 只是对交通状态图像的简短描述，生成成本低；而监督动作可以直接通过仿真、专家策略或规则搜索得到，不需要依赖强教师模型逐条生成高质量长提示词。
2. 对高质量教师模型的依赖更弱。很多长提示词方法首先需要一个足够强的 LLM 把 prompt 写好，否则数据质量本身就不稳定。当前方案把监督重点放在“交通状态表征到动作”的映射上，减少了对教师模型写作能力的依赖。
3. 更适合自动扩数据和跨城市扩展。样本主要由时间序列、图像表征、简短 caption 和仿真动作组成，天然适合批量生成，也更适合后续混入济南、杭州等多城市数据做统一训练。
4. 更不容易走文本捷径。长提示词容易让模型依赖文本模式匹配，而不是理解真实交通动态。短提示词加 projection 的方式，会把训练压力更多压回时间序列图像表征本身。
5. 更有利于 zero-shot 泛化。因为模型学到的重点是交通状态表示和动作决策，而不是某一种固定提示词模板，所以更容易迁移到新城市和新流量分布。


## 10. 阶段性结论

本阶段结果说明，即使在**未引入 Stage 3 PPO 在线优化**的情况下，仅依赖时间序列描述生成、图像表征对齐以及动作监督训练，`Qwen-7B + Projection` 也已经能够在真实交通数据上取得稳定收益。

尤其是在平均等待时间和平均旅行时间两个直接反映通行效率的指标上，模型都取得了较明显下降，说明该方法已经具备较强的交通状态理解能力和一定的决策有效性。

因此，本阶段可以得出以下结论：

1. 基于时间序列生成语义描述，并结合视觉投影进行决策建模，是有效的技术路线。
2. `Qwen-7B + Projection` 相比直接使用 `Gemma-27B`，在交通控制任务上具有更好的整体性能。
3. 当前结果已经验证了 Stage 2 方案本身的价值，而且是在 zero-shot 跨城市测试条件下完成验证的。

## 11. 下一步工作

下一步将重点围绕以下几个方向继续推进：

1. 引入 `PPO`，验证在线强化学习是否能够在当前 Stage 2 基础上进一步提升控制效果。
2. 以 `Gemma-27B` 作为底座模型继续训练，观察更大底座模型是否能够带来更强的决策性能和泛化能力。
3. 在后续版本中融入更多交通信号控制规则与先验知识。当前模型还没有系统融入交通信号灯相关规则，后续将把这些规则进一步纳入训练与推理过程，看看是否能够带来进一步收益。
4. 在训练集中混入不同城市的数据，构造跨城市混合训练集，例如同时加入济南、杭州等不同城市样本，进一步提升模型的泛化能力和跨场景鲁棒性。
