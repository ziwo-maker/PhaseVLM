# 研究方案初稿：策略响应式反事实教师构造（PRCT）

**日期：** 2026-08-18  
**定位：** 面向闭环多路口交通信号控制的教师构造方法；本文只主张一个机制创新。

## Problem Anchor

- **底线问题：** 如何利用仿真反事实监督对已部署的 PhaseVLM 交通信号策略进行少样本适配，而不因局部、协调不一致的教师标签破坏多路口闭环控制？
- **必须解决的瓶颈：** 现有 SimExpert 在枚举一个目标路口动作时冻结其他路口的快照相位。随着仿真视界或标签预算增大，这一反事实不再代表部署时其他路口会随策略重规划的系统，已观测到全直行偏置和高预算 AWT 恶化。
- **非目标：** 不提出新 VLM 主干、通用离线 RL 目标、主动查询方法或可靠性筛选器；不把 generic advantage weighting / KL 约束包装成创新。
- **约束：** 复用冻结 Qwen-VL 8B 和已有投影策略；只改变 CityFlow 标签生成，保持既有四动作代价记录和 Stage-3 mixed CE + soft-target + ranking 训练接口；标签生成成本必须可控。
- **成功条件：** 相对于冻结邻居教师，PRCT 在独立训练种子和闭环评测下同时 (a) 避免动作分布塌缩/饥饿风险，(b) 在存在源策略余量的目标设置提升 AWT 且不恶化 ATT/AQL，(c) 在无余量流量上至少不比冻结教师更具破坏性。

## 技术缺口

冻结邻居的逐动作 rollout 估计的是 `c_frozen(s,a)`：目标交叉口改变相位，但邻居持续停在快照相位。部署中的学生策略执行 `a` 后，邻居会在下一控制点依赖其历史与观测重新决策。因此 `c_frozen` 将局部相位变化放在不可能持续的网络环境中，特别容易把“目标路口短期放行”错排为全局收益。

扩大 rollout horizon、加长训练或叠加 CRCD 可靠性门都不能修正这个干预语义：现有证据中冻结 H300 / CRCD 使 through 行为占优且高 K 退化；PRCAT 的优势加权加 KL 也在与同教师的对比中失败。问题的最小修复是让非目标节点按部署策略响应，而不是再添加学生侧模块。

## 方法论断

**PRCT 将局部动作反事实定义为“固定目标相位干预 + 其余路口按冻结源策略闭环重规划”的 120 秒网络 rollout；由此生成与部署协调机制一致的四动作教师分布。**

这不是给学生加网络、记忆或奖励模型，而是修复教师询问的因果对象。

## 贡献聚焦

- **唯一主贡献：** Policy-Responsive Counterfactual Teacher Construction (PRCT)，即在枚举目标路口候选动作时，恢复同一 CityFlow 快照，并让非目标路口每 30 秒由同一冻结 source policy 决策，从全局多指标代价形成四动作监督。
- **支撑性贡献（非独立创新）：** 用 action entropy、per-action frequency、p95/max movement wait 将“避免协调崩溃”变成可检验的安全证据，而非只报告平均 AWT。
- **明确排除：** PRCAT、CRCD、OOD acquisition、DAgger 和新训练损失均不进入标题或主张；PRCAT 仅作为删除/负向消融。

## 复杂度预算

- **冻结并复用：** Qwen-VL 8B、Stage-2 source policy、PrefixActionPolicy、训练器及其 mixed CE + soft target (alpha=0.3) + rank loss (weight=0.05)。
- **新增可训练组件：** 0。
- **新增运行时组件：** 仅离线标签生成器 `projection_train/policy_responsive_simexpert.py`；部署时无额外模型、查询或规划开销。
- **主动拒绝：** 多视界投票、可靠性 hard gate、优势权重、KL-anchor、主动选样和额外 RL fine-tuning。

## 系统与数据流

```text
收集的 source-policy 快照 s
    -> 对每个目标路口 i、候选相位 a 属于 {ETWT, NTST, ELWL, NLSL}
       -> 从同一 CityFlow snapshot 恢复
       -> t=0: 仅将 i 强制设为 a（含 canonical yellow transition）
       -> t=30,60,90s: 所有非目标路口由冻结 pi_base 根据其滚动观测重新选择相位
       -> 汇总 120s 全局 ATT/AQL/AWT/spillback 代理代价 c_PRCT(s,i,a; pi_base)
    -> 每快照内部归一化四动作代价、Pareto/约束过滤
    -> 保持现有 candidate_action_waits / metrics schema
    -> 原 Stage-3 训练器学习四动作硬标签、soft cost distribution 与 pairwise ranking
    -> 部署：学生独立闭环控制；PRCT 不在测试时运行
```

令 `pi_b` 为 source policy，`j != i` 为非目标路口。对每个候选 `a`，

`c_PRCT(s,i,a;pi_b) = Sum_{tau=0}^{119} g(x_tau)`，其中 `u_i(0)=a`，且每个控制点 `u_j(t)=pi_b(h_j(t))`。

其中 `g` 是当前实现的加权全局 ATT/AQL/AWT/spillback 代理成本。教师动作是可行候选里的 `argmin_a c_PRCT`，而四个归一化代价同时用于软目标与排序。不做硬过滤，以免把教师机制与样本筛选混为一谈。

## 实现锚点

- `PolicyResponsiveSimExpertRelabeler` 以与旧 `SimExpertRelabeler` 相同记录 schema 输出标签，训练器无需分支。
- 每一候选从同一磁盘 snapshot 载入，并清理 Python-side phase / measurement cache，保证候选间状态可比。
- `force_target_action` 保持 source action 向量，只替换 target index；滚动序列 bank 使重规划读取随 rollout 更新的交通计数，而非静态快照。
- 随机种子和 CityFlow work directory 均隔离；标签 determinism 以重放测试验证。

## 训练与推理

训练数据是 source-policy 收集轨迹上按目标路口展开的记录。对 K 个快照做 snapshot-level train/validation split，保持原始 target loss；训练只更新既有 projector / policy head，冻结 VLM。部署时移除 teacher，学生以既有 30 秒决策周期控制全部路口。

PRCT 的核心比较只改变标签的反事实语义；主结果中冻结 c30 与 PRCT H120 必须使用同一 source checkpoint、相同 K、相同标签记录数、同一 loss、训练预算和评测协议。

## 已有证据与其边界

1. Hangzhou 迁移中，冻结 c30 K=118 的 AWT 为 83.24，冻结 H300 为 236；PRCT H120 K=118 的 9-training-seed 聚合为 48.88+-2.76，对 zero-shot 67.75 改善 18.87，并优于 H120 K=32 的 57.93+-4.57。该证据支持“协调语义而非 K 本身”是高预算退化的来源。
2. Jinan real 的健康 pt8 source 上，PRCT H120 K=118 已报告 40.97+-0.84，相对 zero-shot 43.98 有 6.8% AWT 改善。
3. 同一 Jinan-2500 流（collect=test）、9 个训练种子中，PRCT H120 K=118 为 41.47+-0.98，和 zero-shot 41.47 持平。因而不能声称无条件泛化；该流是“源策略/目标流无可用余量”的边界对照。
4. PRCAT full 的唯一有效训练 seed AWT=42.48+-0.22，劣于同 seed 的未加权 PRCT 39.44，故 PRCAT 是应删除的负向模块，而非贡献。

## 新颖性与可辩护性

与普通 local counterfactual / COMA / model-based imitation 的差异不是“做了反事实”本身，而是：在多路口、已部署 VLM policy 的交通控制中，教师明确估计局部 phase 干预在**策略诱导的网络响应**下的代价，并把该语义修复与冻结邻居导致的相位塌缩因果证据对应起来。论文必须避免“first counterfactual TSC”或“new offline RL loss”表述，并以最近文献检索确认是否有直接 policy-responsive teacher 先例。

## 失败模式与诊断

- **源策略已饱和：** PRCT 没有改变空间。诊断为 zero-shot 接近可行上界且所有学生臂与它持平；报告为边界，不以负样本调参。
- **源策略响应本身差：** 120s rollout 可传播 source 的系统性错误。检查 source-policy online performance、H30/H120 sensitivity 和 target action distribution。
- **局部代理与全局指标仍错配：** 使用独立 3600s closed-loop ATT/AQL/AWT 和 movement-level tail metrics，不能以 teacher cost 自证。
- **标签非确定/候选污染：** 同快照多次重放及每候选 cache-reset 测试必须全过；坏 snapshot 必须在选择前过滤并记录实际 K。
- **平均收益掩盖饥饿：** 任何 AWT 改善若 p95/max wait、spillback 或 action entropy 明显恶化，即判为失败。

## 最小验证草图

### 主张 C1：PRCT 修复冻结邻居协调偏差
- 最小实验：同一 source、同一 target、`K={32,118}`，对比 zero-shot、frozen c30、responsive H30、PRCT H120。
- 指标：AWT（主）、ATT/AQL、p95/max movement wait、spillback、action entropy/频率。
- 决定性证据：PRCT H120 在至少一个具有 headroom 的目标设置中相对 frozen c30 同时提高 AWT 和安全指标；高 K 不出现 frozen 的 action collapse。

### 主张 C2：收益来自响应式反事实语义，而非时界、预算或训练损失
- 最小实验：固定 H=120、K=118、同一 loss，仅改变 non-target handler（frozen vs `pi_base` responsive）；外加 H30 responsive 区分 horizon。
- 指标：每个候选的 teacher action / rank disagreement、action distribution 和在线指标。
- 决定性证据：frozen-vs-responsive 的 teacher disagreement 集中在后续产生高 wait / phase imbalance 的状态，且在线顺序随 handler 改变而非单纯随 horizon 改变。

## 实验交接输入

- **必须证明：** C1 的闭环收益和安全性；C2 的因果隔离；Jinan-2500 边界的诚实报告；PRCAT 删除检查。
- **必须运行：** 标签审计、paired teacher comparison、9-seed headroom confirmation、第二目标网络/流、PRCAT negative replication only if pipeline audit invalidates old pilot.
- **最高风险：** PRCT 正例目前集中于一个 Hangzhou 迁移设置，而 Jinan-2500 为零收益；若第二个非饱和设置不能复现，则论文只能定位为诊断/benchmark 工具，而非普遍适配方法。

