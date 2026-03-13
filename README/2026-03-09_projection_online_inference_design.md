# 2026-03-09 Projection 在线推理脚本设计思路

本文档记录本次 projection 在线推理代码的实际修改思路。目标不是继续沿用 `run_open_LLM.py` / `run_open_LLM_with_vllm.py` 的生成式路线，而是把已经训练好的 Stage 2 / Stage 3 projection 直接接到 CityFlow 在线控制里，并补一个和旧入口风格更接近的顶层脚本。

## 本次改动的最终落点

这次不是只改一个文件，而是拆成两层：

1. 顶层新增入口：`run_projection_LLM.py`
2. 底层复用并扩展：`projection_train/run_projection_online.py`
3. 共用 prompt 构造：`projection_train/prompt_utils.py`
4. 共用领域提示词资源：`prompts/prompt_domain_knowledge.json`

这样拆的原因是：

- 用户希望有一个和 `run_open_LLM.py` 同级、调用习惯更接近旧脚本的入口
- 但真正的 projection 在线推理逻辑已经沉淀在 `projection_train/run_projection_online.py`
- 所以更合理的做法不是重复实现一套 runner，而是让顶层脚本只负责参数组织、数据集选择、配置拼接，然后复用 projection runner

## 为什么新顶层入口不是继续走生成式回答

原始 `run_open_LLM.py` / `run_open_LLM_with_vllm.py` 的核心路线是：

`交通状态 -> prompt -> LLM生成文本 -> 正则抽取 <signal>`

这条路线和当前 projection 训练后的真实策略并不一致。

现在 Stage 2 / Stage 3 的真实链路是：

`sequence -> ts_projector -> frozen vision tower -> vision_projector(prefix) -> frozen LLM -> 4动作 logits`

因此这次顶层脚本仍然明确保持：

- 不做自由文本生成
- 不依赖 vLLM
- 直接输出 4 类动作 logits
- 使用 `argmax` 作为部署阶段动作

这样做的原因有三个：

1. 与训练时的动作头语义完全一致
2. 避免重新暴露旧长 prompt 的文本捷径
3. 线上评估更稳定，不引入采样方差

## 为什么要同时保留“顶层旧风格入口”和“底层 projection runner”

如果只保留 `projection_train/run_projection_online.py`，虽然逻辑上已经够用，但调用习惯更偏训练/实验目录，不够接近原来 `run_open_LLM.py` 的使用方式。

因此本次新增：

- `run_projection_LLM.py`

这个脚本的职责是：

- 提供和旧入口接近的 CLI 风格
- 根据 `dataset + traffic_file` 解析 roadnet/template
- 从 checkpoint 和 CLI 恢复 projection runtime 配置
- 复用 `utils/config.py` 中已有的红绿灯相位和时序参数
- 调用 `ProjectionPolicyRunner` 启动测试

底层 `projection_train/run_projection_online.py` 继续承担：

- CityFlow 初始化
- checkpoint 加载
- movement history 构造
- prompt 构造
- 在线动作预测
- 指标统计和日志输出

## 复用 `utils/config.py` 的原因

用户明确要求不要在新脚本里重新发明一套红绿灯控制定义，而要复用 `utils/config.py` 里的现有配置。

因此顶层脚本在配置上采取下面的策略：

### 1. 复用四相位定义

直接沿用 `dic_traffic_env_conf` 中已有的：

- `PHASE`
- `PHASE_LIST`
- `NUM_PHASES`
- `YELLOW_TIME`
- `ALL_RED_TIME`

这样可以保证 projection 在线脚本和项目里其它控制器看到的是同一套信号控制语义。

### 2. 复用动作时长相关默认值

直接沿用 `dic_traffic_env_conf` 中已有的：

- `MIN_ACTION_TIME`
- `MEASURE_TIME`

并且从 `DIC_FIXEDTIME_AGENT_CONF` 里恢复默认 `FIXED_TIME`。如果固定时长配置缺失或长度不匹配，再回退成按 `MIN_ACTION_TIME` 自动展开为 4 个相位时长。

这样做的好处是：

- 新脚本不再把 30 秒写死在新文件里
- 顶层入口和项目默认控制节奏一致
- 后续如果全局信号控制时间调整，projection 脚本也能自动跟上

## Prompt 为什么仍然挂在 `image_only_short`

用户要求“Prompt 添加在 `image_only_short`”，而不是新增一个独立 prompt mode。

因此这次没有再引入新的 mode，而是直接扩展了：

- `projection_train/prompt_utils.py` 中的 `image_only_short`

修改后的 `image_only_short` 不再只是极短约束，而是包含四类信息：

1. 领域 system prompt
2. 车道定义
3. 四相位定义
4. 图像编码规则和决策约束

具体包括：

- 说明 `ET/WT/NT/ST` 是直行，`EL/WL/NL/SL` 是左转
- 解释 `ETWT / NTST / ELWL / NLSL` 各自放行哪些 movement
- 明确图像才是唯一可靠观测
- 说明图像的时间轴、车道轴、强度含义
- 强调只能输出一个 `<signal>...</signal>` 标签

这样处理的原因是：

- 不改变训练和部署共用的 prompt mode 名称
- 仍然保持“短 prompt 优先依赖 prefix 表示”的总体思路
- 同时补足交通规则和车道语义，避免线上说明过于贫乏

## 为什么还要改 `prompts/prompt_domain_knowledge.json`

原先这个 JSON 里只有一句很短的 system prompt，不足以支撑新的 prompt 语义。

因此这次把它提升成更贴近 projection 部署的领域提示前缀，强调：

- traffic rules
- lane geometry
- protected movement definitions
- short-horizon demand trends

然后由 `projection_train/prompt_utils.py` 在构造 `image_only_short` 时读取这个 JSON，作为共用的 system prompt 前缀。

这样做的好处是：

- 领域知识入口不再硬编码在多个脚本里
- 后续如果继续调整全局 system prompt，只需要改 JSON
- prompt 模板和知识前缀分离，更容易维护

## 为什么顶层脚本默认禁用 snapshot 和 prompt-series

这是这次相对于最初设计文档最重要的一处调整。

最开始的 runner 设计里倾向于保留：

- snapshot
- `traffic_prompt_series_*.jsonl`

因为它们对后续回灌训练和离线诊断有价值。

但用户这次明确要求：

1. 不保留 snapshot 记录
2. 不保留 prompt-series 记录

因此实际落地时采用了“两层开关”的做法：

### 1. 底层 runner 仍然保留能力

`projection_train/run_projection_online.py` 没有把 snapshot / prompt-series 逻辑整体删掉，而是新增了三个控制开关：

- `record_snapshots`
- `record_traffic_series`
- `dump_cityflow_env_args`

默认值仍然是 `True`，这样原有实验型 runner 的功能不被破坏。

### 2. 顶层部署入口默认关闭这些记录

`run_projection_LLM.py` 在构造 `ProjectionPolicyRunner` 时显式传入：

- `record_snapshots=False`
- `record_traffic_series=False`
- `dump_cityflow_env_args=False`

这样顶层脚本的行为就符合本次用户需求：

- 不写 snapshot 文件
- 不写 `traffic_prompt_series_*.jsonl`
- 不额外导出 `cityflow_env_init`

### 3. `state_action.json` 仍然保留必要调试信息

虽然 snapshot / prompt-series 关闭了，但仍然保留 `state_action.json` 中这些字段：

- `raw_prompt`
- `policy_prompt`
- `action_logits`
- `action_probs`
- `response`
- `action`

只有 `snapshot_path` 改成在关闭 snapshot 时不再写入。

这样做的平衡点是：

- 减少不需要的重日志
- 保留线上动作可解释性
- 便于之后排查“某一步为什么选了这个相位”

## 为什么对 `ProjectionPolicyRunner` 做的是“参数化扩展”而不是重写

虽然用户最终主要使用的是新顶层脚本，但底层 runner 仍然是更稳的复用点。

因此本次没有复制一份新的 online runner，而是只对现有 `ProjectionPolicyRunner` 做最小必要扩展：

1. 为 snapshot / traffic-series / cityflow init dump 增加开关
2. 在关闭 snapshot 时不再写 `snapshot_path`
3. 保持其余 sequence 构造、checkpoint 加载、动作预测逻辑不变

这样做的原因是：

- 避免复制出第二套几乎相同的在线推理主循环
- 保持 BC / PPO checkpoint 兼容逻辑仍然只有一处
- 未来如果再改 projection runner，不需要同步改两套实现

## 顶层脚本为什么不用 `utils.utils.merge`

一个额外的实现细节是：

- 顶层 `run_projection_LLM.py` 没有复用 `utils.utils.merge`
- 而是在脚本内部写了一个轻量 `merge`

原因不是功能问题，而是依赖边界问题。

`utils.utils` 会连带导入更重的 pipeline / transformers 相关模块，在当前环境里容易过早触发 `numpy / scipy / sklearn` 的兼容问题。

因此顶层脚本只保留最小依赖：

- `torch`
- `projection_train.run_projection_online`
- `utils.config`

这样能让入口层本身尽量轻，真正的重模型依赖只在 projection policy 初始化时触发。

## 在线推理的实际控制流程

当前实际线上流程是：

1. 顶层脚本解析 `dataset / traffic_file / checkpoint`
2. 从 checkpoint 和 CLI 恢复 `ProjectionRuntimeConfig`
3. 复用 `utils/config.py` 中的四相位与时序配置
4. 初始化 `ProjectionPolicyRunner`
5. 在每个控制步读取当前路口统计状态
6. 从 traffic counter 恢复最近窗口的 movement history
7. 补齐到固定长度 `seq_len`
8. 用扩展后的 `image_only_short` 构造 policy prompt
9. 调用 `PrefixActionPolicy` 输出 4 类 logits
10. 对 logits 做 `argmax`
11. 执行 `env.step(action_list)`
12. 记录指标和 `state_action.json`

其中最关键的变化是：

- 线上不再“生成一段分析文本再抽动作”
- 而是“直接做动作分类并执行”

## 当前实现后的文件分工

### 1. `run_projection_LLM.py`

负责：

- 提供顶层 CLI 入口
- 对齐 `run_open_LLM.py` 的使用习惯
- 组织 dataset / roadnet / runtime / config
- 默认关闭 snapshot / prompt-series

### 2. `projection_train/run_projection_online.py`

负责：

- 复用 projection checkpoint 推理主逻辑
- 在线构造 sequence
- 调用 policy 输出动作
- 统计 reward / queue / travel time
- 在需要时仍支持 snapshot / prompt-series

### 3. `projection_train/prompt_utils.py`

负责：

- 统一管理 projection 训练/部署使用的 prompt mode
- 将更完整的交通规则说明注入 `image_only_short`

### 4. `prompts/prompt_domain_knowledge.json`

负责：

- 提供可维护的领域 system prompt 前缀

## 和最初设计文档相比，哪些判断被修正了

本次实现落地后，相比最初版本设计，有三处关键修正：

1. 不再只有 `projection_train/run_projection_online.py` 一个入口，而是额外补了顶层 `run_projection_LLM.py`
2. 不再把 snapshot / prompt-series 当作部署默认，而是只保留为 runner 的可选能力
3. `image_only_short` 不再是极短约束，而是被扩展成“短但信息足够”的规则化 prompt

## 总结

这次修改的核心，不是简单把旧 `run_open_LLM.py` 替换成另一个脚本，而是把 projection 在线部署真正收敛到和训练一致的动作决策链路上，同时满足用户对工程接口和日志行为的具体要求。

最终形成的方案是：

`run_projection_LLM.py -> ProjectionPolicyRunner -> PrefixActionPolicy -> 4动作argmax -> CityFlow`

并且同时满足：

1. 顶层入口风格接近旧脚本
2. 底层决策语义与 Stage 2 / Stage 3 保持一致
3. Prompt 在 `image_only_short` 内补充了完整车道/规则说明
4. 红绿灯相位与时间参数复用了 `utils/config.py`
5. snapshot 和 prompt-series 在顶层默认关闭

这样 projection 不再只是训练产物，而是真正以当前项目的既有控制配置接入到在线仿真闭环中。
