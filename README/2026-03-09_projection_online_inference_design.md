# 2026-03-09 Projection 在线推理脚本设计思路

本文档记录 `projection_train/run_projection_online.py` 的设计思路，重点说明为什么它不再沿用 `run_open_LLM_with_vllm.py` 的生成式推理路线，而是直接接入 Stage 2 / Stage 3 训练好的 `PrefixActionPolicy`。

## 目标

当前要解决的问题不是“再写一个能跑的控制脚本”，而是把已经训练完成的 projection 真正接到在线仿真里，让线上动作选择和训练时的表示链路保持一致。

目标有三个：

1. 直接复用 Stage 2 / Stage 3 训练好的 `ts_projector + vision_projector`
2. 保证线上输入分布尽量贴近训练分布
3. 保持 CityFlow 在线测试、日志输出、snapshot 记录这些已有能力

## 为什么不继续走 vLLM 生成

原始 `run_open_LLM_with_vllm.py` 的核心逻辑是：

`时间序列 -> 图像 -> 长文本 prompt -> vLLM 生成回答 -> 正则抽取 <signal>`

这条路线适合通用生成式基座，但和当前 projection 训练后的真实策略并不完全一致。

现在 Stage 2 / Stage 3 学到的其实是：

`sequence -> ts_projector -> frozen vision tower -> vision_projector(prefix) -> frozen LLM -> 4动作 logits`

所以如果线上还继续让模型自由生成长文本，就会出现两个问题：

1. 训练时优化的是 4 动作打分，而不是自由生成整段分析文本
2. 线上会重新暴露旧 prompt 的文本捷径，削弱 projection 本身的作用

因此新脚本明确改成：

- 不再依赖 `vllm`
- 不再做开放式文本生成
- 直接调用 `PrefixActionPolicy` 输出 4 类动作 logits
- 用 `argmax` 作为线上控制动作

这才是和当前训练链路一致的部署方式。

## 与原脚本的关系

新脚本不是去修改 `run_open_LLM_with_vllm.py`，而是在 `projection_train/` 下新增独立入口：

- `projection_train/run_projection_online.py`

这样做有两个原因：

1. 原脚本仍然保留旧的 vLLM 生成式实验路径，不被破坏
2. projection 部署逻辑和训练代码放在同一目录，更容易一起维护

## 核心设计判断

### 1. 线上主模型应当是 `PrefixActionPolicy`

新脚本的主策略类不是 `LLM_Inference_VLLM`，而是训练侧已经存在的：

- `projection_train/prefix_action_policy.py`

这样可以直接复用：

- checkpoint 加载逻辑
- BC / PPO 权重兼容逻辑
- prompt 构造约束
- 4 动作 logits 输出接口

### 2. checkpoint 必须兼容 BC 和 PPO

线上部署不应该只绑定某一个阶段的产物。

因此脚本设计为：

- 统一使用 `--projection_ckpt`
- 允许传入 BC checkpoint
- 也允许传入 PPO checkpoint

兼容方式很简单：

- 在线推理只需要 `ts_projector + vision_projector`
- PPO 额外的 `value_head` 对纯部署不是必需

所以脚本统一按“只加载策略相关部分”处理，这样 BC / PPO 都能复用。

### 3. 底座模型路径从 checkpoint 和 CLI 共同决定

projection 权重和底座模型是强绑定的。

所以新脚本的模型路径解析顺序是：

1. 优先使用 `--llm_path`
2. 如果未传，则回退到 checkpoint 里保存的 `args.llm_path`
3. 再不行则尝试 `args.model_path`

这样既支持显式覆盖，也支持直接从训练产物恢复部署配置。

### 4. 线上输入必须对齐训练分布

这是这次设计里最重要的判断之一。

虽然旧脚本也会从仿真里取交通状态，但它取 history 的方式和 Stage 2 / Stage 3 训练数据并不一致。

训练侧真实使用的是：

- `2s` 间隔
- `30s` 窗口
- `seq_len=16`
- movement 维度顺序与 grouped dataset 对齐

所以新脚本在线构造输入时，不再复用旧的 `_get_traffic_history_until()` 风格，而是按训练录数逻辑对齐：

- 开启 `intersection_movement` traffic count
- 使用 `TRAFFIC_SERIES_INTERVAL=2`
- 使用 `TRAFFIC_SERIES_WINDOW`
- 从 `traffic_counts_by_interval` 恢复每个路口最近窗口的 movement counts
- 不足 `seq_len` 时向前补齐

这样可以最大程度保证：

线上策略看到的 `sequence`

和

训练时 grouped dataset 中的 `sequence`

语义一致。

### 5. prompt 改成训练侧短 prompt

原脚本主输入里有很长的文本状态说明。

新的线上脚本保留了原始长 prompt 的生成能力，但用途变了：

- `raw_prompt` 只作为兼容记录和调试信息保留
- 真正送进 policy 的 prompt，使用 `build_task_prompt(..., mode=image_only_short)`

这样做的原因是：

1. 训练默认就是 `image_only_short`
2. 短 prompt 更符合“让模型依赖 prefix 图像表示”的目标
3. 线上和训练的 prompt 分布更一致

## 新脚本的主要结构

### 1. 轻量独立入口

新脚本没有直接继承 `LLM_Inference_VLLM`。

原因不是功能上不能继承，而是当前工程里 `LLM_Inference_VLLM` 会把：

- `vllm`
- `transformers`
- 部分额外依赖

一并拉进来，运行环境更脆弱。

因此新脚本选择：

- 直接使用 `CityFlowEnv`
- 只复用轻量工具函数
- 自己维护部署所需的最小逻辑

这样依赖边界更清楚，也避免部署脚本被生成式推理栈绑死。

### 2. `ProjectionRuntimeConfig`

这个数据结构负责承接部署时真正需要的 projection 配置：

- `llm_path`
- `seq_len`
- `image_size`
- `periodicity`
- `input_dim`
- `hidden_dim`
- `prefix_tokens`
- `prefix_mode`
- `max_length`

作用是把“checkpoint 中保存的训练参数”和“CLI 临时覆盖参数”统一收口。

### 3. `ProjectionPolicyRunner`

这是新脚本的主运行器。

职责包括：

- 初始化 `CityFlowEnv`
- 初始化 traffic-series 记录配置
- 加载 `PrefixActionPolicy`
- 在线构造 `sequence + prompt`
- 输出动作并推进仿真
- 保存日志和统计结果

它本质上就是一个“projection 版的在线测试 runner”。

## 在线决策流程

每个控制步主要做下面几件事：

1. 从 `CityFlowEnv` 读取当前每个路口状态
2. 为每个路口恢复最近一段 movement history
3. 把 history 变成固定长度的 `sequence`
4. 用 `image_only_short` 构造 policy prompt
5. 调用 `PrefixActionPolicy` 得到 4 动作 logits
6. 取 `argmax` 作为当前动作
7. 执行 `env.step(action_list)`
8. 记录 reward、queue、travel time、state_action 日志

这里最关键的变化是：

- 不再“生成一段回答再抽动作”
- 而是“直接预测动作”

## 为什么仍然保留 snapshot 和 prompt-series 记录

即使线上动作已经不是自由生成，snapshot 和 series 记录仍然有价值。

保留这部分的原因有三个：

1. 便于后续把线上运行样本重新回灌成训练/诊断数据
2. 便于离线复盘某个时间点 policy 到底看到了什么输入
3. 便于和 Stage 3 的 snapshot 仿真逻辑形成统一的数据接口

所以新脚本仍然会：

- 周期性写 snapshot
- 记录 `traffic_prompt_series_*.jsonl`
- 在 `state_action.json` 里保存：
  - `raw_prompt`
  - `policy_prompt`
  - `action_logits`
  - `action_probs`
  - `snapshot_path`

这样线上行为就不只是“给出一个动作”，而是能完整追溯。

## 为什么日志里同时保留 `raw_prompt` 和 `policy_prompt`

这是为了区分两层语义：

- `raw_prompt`
  - 代表旧系统中面向 LLM 的原始文本描述
  - 方便和历史日志对齐
- `policy_prompt`
  - 代表当前 projection 真正消费的 prompt
  - 方便排查部署是否真的按 `image_only_short` 生效

这两个字段同时存在，有利于后续分析“文本描述变化”和“真实部署输入”之间的差别。

## 与动作空间的关系

新脚本明确只支持四相位：

- `ETWT`
- `NTST`
- `ELWL`
- `NLSL`

因此 `--eightphase` 在这个脚本里被直接禁止。

原因很直接：

1. Stage 2 / Stage 3 的训练动作空间就是固定四类
2. checkpoint 输出头维度也是 4
3. 如果线上放开八相位，只会让部署语义和训练语义错位

## 为什么使用贪心动作而不是采样

当前脚本在线推理选择 `argmax`，而不是从 categorical 分布里采样。

这是部署阶段更合理的默认设置，因为：

1. 在线测试目标是稳定评估当前策略质量
2. 采样会引入额外方差，影响结果比较
3. PPO 的采样需求属于训练阶段，不属于部署阶段

如果后续确实需要做 exploration 测试，再单独扩展 `--sample_action` 会更清楚。

## 当前实现的边界

当前脚本已经实现：

- projection checkpoint 的在线加载
- BC / PPO checkpoint 兼容
- 训练分布对齐的序列构造
- projection prompt 线上替换
- CityFlow 在线仿真接入
- state / prompt / logits / snapshot 日志记录

当前没有实现：

- 八相位 projection 部署
- 多种部署策略切换（如温度采样、top-k）
- 更复杂的线上 ensemble

## 环境层面的注意事项

这次脚本刻意把重型依赖延后到真正加载 `PrefixActionPolicy` 时再触发，并且先设置：

- `TRANSFORMERS_NO_TF=1`
- `USE_TF=0`

原因是当前项目环境里存在：

- `numpy`
- `scipy`
- `transformers`

之间的二进制兼容风险。

这不是这份设计文档的重点，但部署时需要意识到：

- 脚本入口可以轻量工作
- 真正模型加载仍然依赖一套和训练时一致的可用环境

## 总结

这次在线脚本的核心不是“把旧的 vLLM 推理代码改一改”，而是把部署语义彻底改成与 Stage 2 / Stage 3 一致的策略前向：

`traffic series -> projection policy -> 4动作 -> CityFlow`

这样做的结果是：

1. 训练和部署共享同一条动作决策链路
2. prompt、输入分布、动作头语义都尽量一致
3. projection 不再只是训练产物，而是真正进入在线仿真闭环

这也更符合原始目标：

- 把信号控制和在线仿真真正打通
- 而不是继续停留在“图像 + 长文本 + 生成式抽取动作”的过渡方案
