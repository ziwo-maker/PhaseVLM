# JSCD 混入 Hangzhou_5816 与训练集扩容实验

**日期：** 2026-08-20  
**状态：** 已完成启动前审计，待运行  
**性质：** 探索性数据规模/跨流量监督实验，不替代已失败的 JSCD P2 efficacy gate。

## 目的

检验 JSCD P2 的不稳定性是否与每种子只有 86 个 `Hangzhou_real` 快照（1,376 条交叉口样本）的训练规模有关。将可用的 `Hangzhou_5816` 监督池作为共同补充训练数据，以提高流量模式覆盖；在线评估保持在用户指定的 `anon_4_4_hangzhou_real.json`。

## 数据与隔离

- P2 primary：每个臂/训练种子固定使用既有 86 个 `Hangzhou_real` JSCD 第二轮快照；train/validation 为 77/9 个快照，即 1,232/144 条记录。
- 5816 supplement 原始文件：`data/processed/hangzhou_real_5816_stage3_queue_wait_rule_gap2_p60_20260531.jsonl`。
- 原文件有 30,480 行，但只有 1,920 行具备完整 expert action、4 个候选 wait 与 candidate metrics；审计后形成 120 个完整的 16 路口快照。无标签行不用于监督，但用于重建每条补充标签的 16 步运动序列。
- 每个臂/训练种子从该 120 快照池中以 `selection_seed=20260820` 固定抽取 86 个；再按训练 seed 拆分为 77/9 个快照。故每个臂实际训练样本从 1,232 增至 2,464，验证样本从 144 增至 288，总监督快照从 86 增至 172。
- `Hangzhou_real` 是唯一线上测试流。5816 在本轮被用户指定为训练来源，不能再被报告为该模型的独立泛化测试。

## 公平比较

| 臂 | primary 第二轮标签 | 5816 补充标签 | 初始化 | 唯一机制差异 |
|---|---|---|---|---|
| F | Frozen H120 | 同一固定 5816 子集 | 匹配的 P2 Frozen `pi_1` | 无 responsive continuation |
| P | Static-PRCT H120 | 同一固定 5816 子集 | 匹配的 PRCT `pi_1` | `pi_0` continuation |
| J | JSCD H120 | 同一固定 5816 子集 | 与 P 同一 `pi_1` | `pi_1` continuation |

所有臂保留 P2 的架构、mixed loss、5 epoch、优化器、selection metric、三训练 seed (`2024/42/123`) 与三卡 DDP。补充数据在三臂间完全相同，因此 J-P 仍只改变 JSCD 的 second-pass continuation；但相对原 P2 的数据扩容比较增加了样本量和总更新次数，必须作为数据规模实验解释。

## 运行与输出

- 启动器：`scripts/launch_jscd_mixed_5816_scale.sh`
- 运行器：`scripts/run_jscd_mixed_5816_scale.py`
- 线上评估：每个 checkpoint 在 `anon_4_4_hangzhou_real.json`、`run_counts=3600`、`eval_seed=2024` 下运行一次。该控制器/交通流组合是确定性的，训练 seed 才是独立统计单位。
- 产物：每个 run 的 `dataset_manifest.json` 固化 source hash、snapshot IDs、split 与 init checkpoint hash；根目录 `status.json` 支持后台检查；`aggregate.json` 同时给出混合结果、原 P2 结果、mixed-minus-P2 及 J-P/J-F 的 AWT 配对差。

## 解释边界

正向平均值不能恢复原 JSCD P2 的性能主张。最低报告标准仍是三训练 seed 的逐对方向、两侧绝对指标和数据 hash。若混合后 J-P 仍不稳定，应优先归因为 JSCD continuation 本身未产生可靠在线收益，而不是继续无约束增加训练数据。
