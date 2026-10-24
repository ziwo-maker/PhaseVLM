# Stage 2 No-Balance 在线评测分析

## 在线测试结果（旧 reward 体系，lj_lsj_1h_1Lane）

| Stage 2 模型 | Val F1 | Avg Queue | Avg Wait (s) | Avg Travel (s) | Final Queue | Reward |
|---|---|---|---|---|---|---|
| **lr1e4_vr01 (val 0.8425)** | 0.8425 | **122.85** | **62.35** | **284.02** | **14,742** | **-631.65** |
| actionbal_fix (val 0.8636) | 0.8636 | 134.11 | 79.08 | 292.77 | 16,093 | -654.08 |
| lr5e5_min5e6 (val 0.8947) | **0.8947** | 134.73 | 69.98 | 295.02 | 16,168 | -652.79 |

## 关键发现：Val F1 与在线性能反相关

- **lr5e5（val F1 最高 0.8947）→ 在线最差**：队列最长 16,168、travel 最高 295s、测试耗时最长 757s
- **lr1e4（val F1 中等 0.8425）→ 在线最好**：队列少 1,426 辆、wait 低 7.6s、travel 低 11s、测试仅 320s
- lr5e5 测试耗时 757s（lr1e4 仅 320s），说明 lr5e5 策略犹豫、决策慢

## 根因分析

lr5e5 可能**过拟合到了训练集的 action 分布**（BC 任务中预测动作准确），但在实际仿真中：
1. 策略过于刚硬，不能根据交通状态灵活调整
2. lr1e4 学习率更低，模型更加平滑，泛化更好
3. Val F1 衡量的是"能否复现训练集动作"，而非"能否做出好的控制决策"

## Stage 3 PPO 验证（新 stable_delta reward）

| Stage 3 配置 | Stage 2 基础 | Best Eval Reward | Avg Queue | Avg Wait |
|---|---|---|---|---|
| stable_delta_conservative | exp_jinan/stage2_bc (旧 baseline) | **+66.3** | ~106 | ~37s |
| stable_delta_full (发散) | exp_jinan/stage2_bc | -703 | 286 | 151s |
| from_expA_conservative | expA_noreb (val 0.452) | -42~-453 | 143~203 | 67~138s |

Stage 3 PPO 仍然是旧 baseline (val F1 更低) 表现最好。

## 最佳模型

**现阶段最佳 nobalance 模型: `stage2_beststage1_deduped_temp010_lr1e4_vr01_bs8ga8_ep20_20260516_165853`**
- Path: `/gemini/code/LLMSTSC/records/pipeline_runs/exp_jinan2000/stage2_beststage1_deduped_temp010_lr1e4_vr01_bs8ga8_ep20_20260516_165853/checkpoints/best.pt`

---

*分析日期: 2026-05-18*
