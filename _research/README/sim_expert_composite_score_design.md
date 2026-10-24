# sim_expert_composite_score_design 修改说明

## 本次改动

- 在 `expert_action_margin` 含义变化的说明后，补充了更具体的经验阈值建议：
  - `--min_action_margin 1.0`
  - `--min_action_margin 1.5`

## 修改目的

- 之前文档只说明了 `expert_action_margin` 的数值语义变了，但没有给出新的起始阈值。
- 这次补充后，Stage 1 的高质量子集清洗可以直接从两个更实用的阈值开始试验。

## 解释

- `1.0` 更偏稳妥，适合先过滤掉一批过于模糊的监督样本。
- `1.5` 更偏激进，适合构造更高置信度的一阶段对齐子集。
