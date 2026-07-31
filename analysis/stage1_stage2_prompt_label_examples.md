# Stage 1 / Stage 2 Prompt and Label Examples

本文从当前 Hangzhou 5816 compact 训练集抽取样例，说明 Stage 1 和 Stage 2 中“输入提示/监督文本/专家决策”的实际对应关系。

## 数据来源

- 主数据文件：`records/sim_expert_validation_search/anon_4_4_hangzhou_real_5816/best/align_train.stage1_compact_ds.jsonl`
- 原始 prompt series 参照：`records/sim_expert_validation_search/anon_4_4_hangzhou_real_5816/best/traffic_prompt_series_2s.jsonl`
- 有效 compact 样本数：`1335`
- 专家动作分布：
  - `ETWT`: `481`
  - `NTST`: `457`
  - `ELWL`: `228`
  - `NLSL`: `169`

## 字段口径

Stage 1 的训练目标是交通时序/图像表征与文本语义对齐。代码中 `TrafficClipJsonlDataset` 按一组连续记录构造 `sequence`，并使用组末尾的 `clip_caption` 作为文本监督。因此 Stage 1 的核心对应关系是：

```text
traffic sequence / image representation -> clip_caption
```

Stage 2 的训练目标是动作监督。当前 `prompt_mode=image_only_short` 时，模型实际看到的任务提示词为：

```text
You are controlling a four-phase traffic signal.
The attached image is the only reliable observation of recent traffic dynamics at one intersection.
Choose the single best next phase to reduce near-term congestion.
Allowed outputs: <signal>ETWT</signal>, <signal>NTST</signal>, <signal>ELWL</signal>, <signal>NLSL</signal>.
Return exactly one <signal>...</signal> tag and nothing else.
```

Stage 2 的硬标签是 `expert_action`。如果使用 mixed/soft target，还会使用 `candidate_action_waits` 形成软监督；这里的等待值越小代表该动作越优。

## Stage 1 样例：时序窗口到文本标注

### 样例 1

- intersection: `intersection_3_2`
- date: `1970-01-01 00:36:30`
- latest movement counts: `NT=0, NL=0, ST=0, SL=0, ET=11, EL=0, WT=8, WL=0`
- window trend: total vehicles `11 -> 19`, delta `+8`
- dominant lane over window: `ET`
- Stage 1 text label `clip_caption`:

```text
Traffic has been steadily building over the past half minute, with queues growing as more vehicles arrive at the intersection. The busiest lanes are east through and west through, both seeing heavy flow, while the north and south approaches remain completely empty.
```

### 样例 2

- intersection: `intersection_1_3`
- date: `1970-01-01 00:15:30`
- latest movement counts: `NT=10, NL=0, ST=4, SL=0, ET=1, EL=0, WT=0, WL=0`
- window trend: total vehicles `15 -> 15`, delta `0`
- dominant lane over window: `NT`
- Stage 1 text label `clip_caption`:

```text
Traffic flow has remained steady over the past half-minute, with no signs of congestion building or clearing. The busiest lanes are northbound through (NT) and southbound through (ST), both moving consistently, while the quietest is eastbound through (ET), with only light activity.
```

### 样例 3

- intersection: `intersection_2_2`
- date: `1970-01-01 00:13:00`
- latest movement counts: `NT=0, NL=0, ST=0, SL=0, ET=1, EL=0, WT=0, WL=2`
- window trend: total vehicles `6 -> 3`, delta `-3`
- dominant lane over window: `WL`
- Stage 1 text label `clip_caption`:

```text
East-west traffic dominates the scene, with left turns flowing freely and slightly more prominent than through movements. The westbound left turn (WL) is the most congested movement right now, steadily active, while the northbound and southbound directions (NT, NL, ST, SL) are nearly empty with zero demand.
```

### 样例 4

- intersection: `intersection_3_1`
- date: `1970-01-01 00:04:30`
- latest movement counts: `NT=0, NL=1, ST=0, SL=0, ET=0, EL=0, WT=1, WL=0`
- window trend: total vehicles `0 -> 2`, delta `+2`
- dominant lane over window: `NL`
- Stage 1 text label `clip_caption`:

```text
Traffic has been gradually building from a quiet start, with queues now forming at the intersection. The busiest movement is northbound left turns (NL), while the quietest lanes, including northbound through (NT), southbound through (ST), southbound left (SL), and eastbound through (ET), have seen no vehicles at all.
```

## Stage 2 样例：短提示词到专家动作

以下样例使用同一个 `image_only_short` 任务提示词，差异主要来自附带的交通图像/时序表示。`candidate_action_waits` 中最小等待值对应专家动作；`expert_action_margin` 是最优和次优动作的等待差距。

### 专家动作 ETWT

- intersection: `intersection_3_2`
- date: `1970-01-01 00:36:30`
- expert decision: `<signal>ETWT</signal>`
- candidate action waits: `ETWT=0.2, NTST=4.0, ELWL=4.0, NLSL=4.0`
- expert action margin: `3.8`
- caption evidence: east through and west through are heavy, north/south approaches are empty.

### 专家动作 NTST

- intersection: `intersection_1_3`
- date: `1970-01-01 00:15:30`
- expert decision: `<signal>NTST</signal>`
- candidate action waits: `ETWT=4.0, NTST=0.2, ELWL=4.0, NLSL=4.0`
- expert action margin: `3.8`
- caption evidence: northbound through and southbound through are the busiest movements.

### 专家动作 ELWL

- intersection: `intersection_2_2`
- date: `1970-01-01 00:13:00`
- expert decision: `<signal>ELWL</signal>`
- candidate action waits: `ETWT=4.0, NTST=4.0, ELWL=0.2, NLSL=4.0`
- expert action margin: `3.8`
- caption evidence: east-west traffic dominates, with westbound left turn most congested.

### 专家动作 NLSL

- intersection: `intersection_3_1`
- date: `1970-01-01 00:04:30`
- expert decision: `<signal>NLSL</signal>`
- candidate action waits: `ETWT=3.8555, NTST=3.8919, ELWL=3.8919, NLSL=0.2`
- expert action margin: `3.6555`
- caption evidence: queues are forming, with northbound left turns as the busiest movement.

## 说明

同一条 compact 样本同时携带 `prompt`、`clip_caption`、`expert_action`、`candidate_action_waits` 等字段，但两个阶段使用重点不同：

- Stage 1 主要学习交通时序/图像表征与 `clip_caption` 的语义对应。
- Stage 2 使用固定短任务提示词和交通图像/时序表示，学习输出专家动作 `<signal>...</signal>`。
- 原始长 `prompt` 仍存在于数据中，但当前 Stage 2 配置会通过 `prompt_mode=image_only_short` 替换为短提示词，以减少文本捷径并强化图像/时序表征的决策作用。
