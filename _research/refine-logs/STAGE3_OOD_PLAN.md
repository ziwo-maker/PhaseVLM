# Stage 3 OOD-Guided Adaptation — Experiment Plan (2026-08-13)

> 状态：代码已实现 + 跨模型评审两轮 → 待部署
> 依据：refine-logs/EXPERIMENT_FINAL_REPORT.md + EXPERIMENT_RESULTS.md（Phase 2: OOD 为最优采集策略）

## 主方法：OOD-Guided SimExpert Adaptation

源策略在目标路网采集决策状态 → 仅用采集时特征（movement counts + phase queue/wait，16 维，严格无零填充）计算快照级 OOD 得分（到源分布的马氏距离，对角协方差、尺度感知方差下限）→ 选 top-K 快照 → **仅对选中的 K 个快照**做完整 SimExpert 反事实重标注（诚实 K-query 预算）→ 快照级 train/val 切分 → 从源 checkpoint 热启动 BC 微调（mixed 损失）→ 与采集流不相交的测试流上 3 种子在线评测（t 区间）。

## 实验矩阵

| Arm | 方法 | K | 说明 |
|-----|------|---|------|
| ZS | zero-shot | 0 | 源 pt32/hd256 策略直接部署 Hangzhou 测试流 |
| RND | random | 8 / 32 / 120 | 随机快照基线（相同协议） |
| OOD | ood | 8 / 32 / 120 | **主方法**：OOD top-K 快照 |

- 训练 seed=2024，评测 seeds={2024, 42, 123}（3 GPU 并行，每进程唯一 memo）
- 收集流 `anon_4_4_hangzhou_real.json` ≠ 测试流 `anon_4_4_hangzhou_real_5816.json`
- 同网络对照（Jinan→Jinan）**本轮不做**：三条 Jinan 流量全部用于源 Stage-2 训练，不存在 held-out Jinan 流（评审发现 #9，已记录）

## 成功标准

1. 同等 K 下 OOD 的 ATT/AQL/AWT 不低于 random（主效应）
2. AWT 不随 K 增大而退化（对比旧 pipeline 的 K=120 退化问题）
3. OOD K=120 在 Sum(ATT+AQL+AWT) 上最优

## 资源估计

- 6 次 SimExpert 重标注（K=8/32/120 × 2 方法，≈ 2+8+30 min/方法，CPU）
- 6 次微调（各 ~5-15 min，2× RTX 5090）
- 21 次在线评测（各 ~2-5 min，3 GPU 并行）+ 1 次采集 + 1 次 zero-shot
- 总计 ≈ 3-4 小时墙钟，~8 GPU-hours

## 架构（2026-08-12 重训练完成）

- prefix_tokens=32, hidden_dim=256, max_length=64, seq_len=16
- 源 checkpoint：`records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt`（Jinan，val_acc=73.5%）
- Stage-1 对齐：`records/pipeline_runs/stage1_pt32_hd256_20260812/best.pt`

## 跨模型评审修复记录（两轮）

| # | 级别 | 问题 | 修复 |
|---|------|------|------|
| 1 | CRITICAL | SimExpert 未模拟黄灯过渡（3s） | 反事实仿真按 canonical 语义模拟 phase 0 黄灯过渡 + 单测 |
| 2 | CRITICAL | 并行评测 memo+时间戳工作区冲突 | 每进程唯一 memo（tag+seed+uuid） |
| 3 | MAJOR | 源分布被零填充污染（1440/22860 记录完整） | 严格特征提取 + 覆盖率检查 + 尺度感知方差下限 |
| 4 | MAJOR | best.pt 指标丢弃不可行预测 | 不可行预测按惩罚成本计入 + invalid_action_rate |
| 5 | MAJOR | 记录级验证切分泄漏 | 快照级 train/val 切分（--train_data_path/--val_data_path） |
| 6 | MAJOR | 120 池嵌套子采样≠K 查询预算 | 重构为 per-(method,K) 独立选择+重标注 |
| 7 | MAJOR | 重标注失败静默回退伪标签 | verify_relabeled 强制全覆盖，失败即中止 |
| 8 | MAJOR | att/awt 命名与实际含义不符 | 文档声明为 cost 代理量（在线指标以 CityFlow 为准） |
| 9 | MAJOR | Jinan 网格映射 (3,4) 错误 + 对照流非 held-out | 网格修正为 (4,3)；同网络对照本轮移除 |
| 10 | MAJOR | resume 工件与配置无绑定 | config 指纹（配置+代码+checkpoint mtime）写入所有 sentinel，不匹配拒绝 resume |
| 11 | MINOR | 不可评分快照排第一 | 改为 -inf 排最后 |

已知遗留（与本实验无关，未触碰文件）：`tests/test_cost_ranking.py` 中权重覆盖路径（awt=10）的预存在失败；本实验训练权重恰为 DEFAULT_COST_WEIGHTS，走预计算分数捷径，不受影响。

## 实现文件

- `projection_train/ood_selection.py` — OOD 特征/源统计/快照打分/选择（13 单测通过）
- `scripts/run_stage3_ood_adaptation.py` — 全流程管线（zero-shot → 采集 → 选择 → 重标注 → 验证 → 切分 → 微调 → 评测）
- `scripts/relabel_simexpert.py` — 黄灯过渡仿真 + Jinan 网格修正（4 单测通过）
- `projection_train/train_projection_bc.py` — 不可行预测惩罚 + invalid_action_rate
