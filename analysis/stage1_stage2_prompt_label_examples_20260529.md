# Stage 1 / Stage 2 Prompt and Label Examples

本文从当前本地训练集抽取样例，说明 Stage 1 与 Stage 2 中“提示词/输入上下文”与“标注或专家决策”的对应关系。

## 数据来源

Stage 1 使用 compact 对齐数据，核心监督是 `sequence / time-series image -> clip_caption`：

- `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl`
- `records/sim_expert_validation_search/anon_3_4_jinan_real_2000/best/align_train.stage1_compact_deduped.jsonl`
- `records/sim_expert_validation_search/anon_3_4_jinan_real_2500/best/align_train.stage1_compact_ds.jsonl`

Stage 2 使用最新 queue/wait rule processed 数据：

- `data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl`
- `data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
- `data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl`

这些 JSONL 不是每一行都带监督字段，而是连续多行 `movement_counts` 组成一个时序窗口，通常在窗口末尾记录写入 `prompt`、`clip_caption`、`expert_action` 等字段。下面样例均选取带监督字段的窗口末尾记录。

## 字段口径

Stage 1 的主要监督信号是 `clip_caption`，用于学习交通时序/图像表征与自然语言交通描述之间的对应关系。记录里也可能带有 `expert_action`、`candidate_action_waits`，但这些不是 Stage 1 的主训练目标。

Stage 2 的主要监督信号是 `expert_action`。当前训练代码会读取 `raw_prompt`、`prompt_context`、`candidate_action_waits`、`phase_queue`、`phase_wait` 等字段；如果命令设置 `prompt_mode=image_only_short`，实际输入给策略模型的任务提示词会统一为短提示词。processed 文件中的原始 `prompt` 字段仍保留在样例中，杭州最新文件已经是短提示词，济南 2000/2500 最新文件仍可见较长原始模板。

当前短任务提示词如下：

```text
You are controlling a four-phase traffic signal.
The attached image is the only reliable observation of recent traffic dynamics at one intersection.
Choose the single best next phase to reduce near-term congestion.
Allowed outputs: <signal>ETWT</signal>, <signal>NTST</signal>, <signal>ELWL</signal>, <signal>NLSL</signal>.
Return exactly one <signal>...</signal> tag and nothing else.
```

## Stage 1 样例

### S1-1 Hangzhou Real

- source: `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl`
- intersection/date: `intersection_1_4`, `1970-01-01 00:15:30`
- latest movement counts: `NT=7, NL=2, ST=3, SL=0, ET=7, EL=1, WT=23, WL=1`
- window total vehicles: `40 -> 44`
- prompt field excerpt:

```text
You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks.
IMPORTANT:
- The attached image is the ONLY reliable observation of the intersection state for the last 30 seconds.
- Do NOT assume the traffic numbers in text are correct. If any text conflicts with the image, trust the image.
- You MUST look at and use the image to make the decision.

### What the image encodes (read carefully)
The image encodes a 30-second time series sampled every 2 seconds (15 time steps) for 8 lanes at one intersection.
...
```

- Stage 1 label `clip_caption`:

```text
The westbound through lane (WT) shows a heavy, growing queue, while the eastbound through (ET) and northbound through (NT) lanes have steady but lighter flows. The westbound left-turn lane (WL) is clearing quickly, and the southbound left (SL) lane remains completely empty.
```

- auxiliary action fields: `expert_action=NTST`, `base_action=ELWL`, `candidate_action_waits: ETWT=1.2, NTST=0.8, ELWL=3.6, NLSL=3.2`

### S1-2 Jinan Real 2000

- source: `records/sim_expert_validation_search/anon_3_4_jinan_real_2000/best/align_train.stage1_compact_deduped.jsonl`
- intersection/date: `intersection_2_2`, `1970-01-01 00:17:00`
- latest movement counts: `NT=8, NL=0, ST=10, SL=0, ET=10, EL=2, WT=14, WL=1`
- window total vehicles: `39 -> 45`
- prompt field excerpt:

```text
You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks.
IMPORTANT:
- The attached image is the ONLY reliable observation of the intersection state for the last 30 seconds.
- Do NOT assume the traffic numbers in text are correct. If any text conflicts with the image, trust the image.
- You MUST look at and use the image to make the decision.
...
```

- Stage 1 label `clip_caption`:

```text
East-west through traffic is dominant and overall demand appears steady. The latest load is concentrated on west through (WT) and east through (ET) movements, with increasing volumes also observed on south through (ST) and north through (NT).
```

- auxiliary action fields: `expert_action=ETWT`, `base_action=NLSL`, `candidate_action_waits: ETWT=0.2, NTST=3.0, ELWL=3.4, NLSL=3.5`

### S1-3 Jinan Real 2500

- source: `records/sim_expert_validation_search/anon_3_4_jinan_real_2500/best/align_train.stage1_compact_ds.jsonl`
- intersection/date: `intersection_3_2`, `1970-01-01 00:39:00`
- latest movement counts: `NT=18, NL=0, ST=15, SL=1, ET=3, EL=1, WT=7, WL=1`
- window total vehicles: `31 -> 46`
- prompt field excerpt:

```text
You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks.
IMPORTANT:
- The attached image is the ONLY reliable observation of the intersection state for the last 30 seconds.
- Do NOT assume the traffic numbers in text are correct. If any text conflicts with the image, trust the image.
- You MUST look at and use the image to make the decision.
...
```

- Stage 1 label `clip_caption`:

```text
North-south through traffic dominates the scene, flowing freely as the primary movement. The north through (NT) movement is the most congested, while the east through (ET) movement is the emptiest.
```

- auxiliary action fields: `expert_action=NTST`, `base_action=ETWT`, `candidate_action_waits: ETWT=2.3, NTST=0.0, ELWL=2.2, NLSL=2.6`

## Stage 2 样例

### S2-1 Hangzhou Real Latest: Normal High-Queue ETWT

- source: `data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl`
- intersection/date: `intersection_1_4`, `1970-01-01 00:19:30`
- latest movement counts: `NT=3, NL=3, ST=2, SL=0, ET=2, EL=1, WT=39, WL=8`
- record prompt:

```text
You are controlling a four-phase traffic signal.
The attached image is the only reliable observation of recent traffic dynamics at one intersection.
Choose the single best next phase to reduce near-term congestion.
Allowed outputs: <signal>ETWT</signal>, <signal>NTST</signal>, <signal>ELWL</signal>, <signal>NLSL</signal>.
Return exactly one <signal>...</signal> tag and nothing else.
```

- `prompt_context`:

```text
Traffic q/w(s): lanes NT=0/0,NL=3/60.7,ST=0/0,SL=0/0,ET=2/20,EL=1/98,WT=19/13.7,WL=7/170.6; phases ETWT=21/16.8,NTST=0/0,ELWL=8/134.3,NLSL=3/30.3; maxQ=ETWT(21); maxW=ELWL(134.3); totalQ=32.
```

- `clip_caption`: `Highest demand on WT(39) and WL(8), top-2 share 81%. Through lanes high, left-turn lanes low; overall demand steady. East-West share 86%, North-South share 14% (east-west dominant).`
- expert decision: `<signal>ETWT</signal>`
- `base_action`: `ETWT`
- `phase_queue`: `ETWT=21, NTST=0, ELWL=8, NLSL=3`
- `phase_wait`: `ETWT=16.8, NTST=0, ELWL=134.3, NLSL=30.3`
- `candidate_action_waits`: `ETWT=-21, NTST=0, ELWL=-8, NLSL=-3`
- `max_phase_override`: `overridden=false`, `original_action=ETWT`, `final_action=ETWT`, `target_phase=ETWT`, `rule_gap=2`, `left_wait_protect=60`

### S2-2 Hangzhou Real Latest: Left-Wait Protected NLSL

- source: `data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl`
- intersection/date: `intersection_4_4`, `1970-01-01 00:36:00`
- latest movement counts: `NT=0, NL=0, ST=0, SL=6, ET=18, EL=2, WT=3, WL=0`
- record prompt: same short task prompt as S2-1
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=0/0,NL=0/0,ST=0/0,SL=6/362.2,ET=13/67.8,EL=1/27,WT=3/80.3,WL=0/0; phases ETWT=16/74.1,NTST=0/0,ELWL=1/13.5,NLSL=6/181.1; maxQ=ETWT(16); maxW=NLSL(181.1); totalQ=23.
```

- `clip_caption`: `Highest demand on ET(18) and SL(6), top-2 share 83%. Through lanes moderate, left-turn lanes moderate; overall demand steady. East-West share 79%, North-South share 21% (east-west dominant).`
- expert decision: `<signal>NLSL</signal>`
- `base_action`: `NLSL`
- `phase_queue`: `ETWT=16, NTST=0, ELWL=1, NLSL=6`
- `phase_wait`: `ETWT=74.1, NTST=0, ELWL=13.5, NLSL=181.1`
- `candidate_action_waits`: `ETWT=-16, NTST=0, ELWL=-1, NLSL=-6`
- `max_phase_override`: `overridden=false`, `original_action=NLSL`, `final_action=NLSL`, `target_phase=ETWT`, `protect_reason=left_wait_protect:NLSL=181.083>=60.000`, `gap=10`, `rule_gap=2`, `left_wait_protect=60`

### S2-3 Hangzhou Real Latest: Rule Override To ELWL

- source: `data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl`
- intersection/date: `intersection_1_4`, `1970-01-01 00:05:00`
- latest movement counts: `NT=5, NL=0, ST=1, SL=0, ET=0, EL=0, WT=15, WL=9`
- record prompt: same short task prompt as S2-1
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=3/60,NL=0/0,ST=1/59,SL=0/0,ET=0/0,EL=0/0,WT=3/8.7,WL=7/106.4; phases ETWT=3/4.3,NTST=4/59.5,ELWL=7/53.2,NLSL=0/0; maxQ=ELWL(7); maxW=NTST(59.5); totalQ=14.
```

- `clip_caption`: `Highest demand on WT(15) and WL(9), top-2 share 80%. Through lanes moderate, left-turn lanes moderate; overall demand rising. East-West share 80%, North-South share 20% (east-west dominant).`
- expert decision: `<signal>ELWL</signal>`
- `base_action`: `ETWT`
- `phase_queue`: `ETWT=3, NTST=4, ELWL=7, NLSL=0`
- `phase_wait`: `ETWT=4.3, NTST=59.5, ELWL=53.2, NLSL=0`
- `candidate_action_waits`: `ETWT=-3, NTST=-4, ELWL=-7, NLSL=0`
- `max_phase_override`: `overridden=true`, `original_action=ETWT`, `final_action=ELWL`, `target_phase=ELWL`, `gap=4`, `rule_gap=2`, `left_wait_protect=60`

### S2-4 Jinan Real 2000 Latest: Normal High-Queue NTST

- source: `data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
- intersection/date: `intersection_4_2`, `1970-01-01 00:06:00`
- latest movement counts: `NT=11, NL=1, ST=13, SL=0, ET=12, EL=0, WT=0, WL=0`
- record prompt excerpt:

```text
You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks.
IMPORTANT:
- The attached image is the ONLY reliable observation of the intersection state for the last 30 seconds.
- Do NOT assume the traffic numbers in text are correct. If any text conflicts with the image, trust the image.
- You MUST look at and use the image to make the decision.
...
```

- actual short task prompt when `prompt_mode=image_only_short`: same as the short prompt in the 字段口径小节
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=10/23.8,NL=0/0,ST=8/26.2,SL=0/0,ET=8/26.4,EL=0/0,WT=0/0,WL=0/0; phases ETWT=8/13.2,NTST=18/25.0,ELWL=0/0,NLSL=0/0; maxQ=NTST(18); maxW=NTST(25.0); totalQ=26.
```

- `clip_caption`: `Highest demand on ST(13) and ET(12), top-2 share 68%. Through lanes high, left-turn lanes low; overall demand steady. East-West share 32%, North-South share 68% (north-south dominant).`
- expert decision: `<signal>NTST</signal>`
- `base_action`: `NTST`
- `phase_queue`: `ETWT=8, NTST=18, ELWL=0, NLSL=0`
- `phase_wait`: `ETWT=13.2, NTST=25, ELWL=0, NLSL=0`
- `candidate_action_waits`: `ETWT=-8, NTST=-18, ELWL=0, NLSL=0`
- `max_phase_override`: `overridden=false`, `original_action=NTST`, `final_action=NTST`, `target_phase=NTST`, `rule_gap=2`, `left_wait_protect=60`

### S2-5 Jinan Real 2000 Latest: Left-Wait Protected NLSL

- source: `data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
- intersection/date: `intersection_2_2`, `1970-01-01 00:16:30`
- latest movement counts: `NT=6, NL=5, ST=5, SL=3, ET=10, EL=2, WT=8, WL=0`
- record prompt excerpt: same long raw prompt schema as S2-4
- actual short task prompt when `prompt_mode=image_only_short`: same as the short prompt in the 字段口径小节
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=3/13.7,NL=4/184,ST=2/4,SL=3/280.3,ET=6/33.2,EL=0/0,WT=8/10.4,WL=0/0; phases ETWT=14/21.8,NTST=5/8.8,ELWL=0/0,NLSL=7/232.2; maxQ=ETWT(14); maxW=NLSL(232.2); totalQ=26.
```

- `clip_caption`: `Highest demand on ET(10) and WT(8), top-2 share 46%. Through lanes moderate, left-turn lanes moderate; overall demand steady. East-West share 51%, North-South share 49% (balanced).`
- expert decision: `<signal>NLSL</signal>`
- `base_action`: `NLSL`
- `phase_queue`: `ETWT=14, NTST=5, ELWL=0, NLSL=7`
- `phase_wait`: `ETWT=21.8, NTST=8.8, ELWL=0, NLSL=232.2`
- `candidate_action_waits`: `ETWT=-14, NTST=-5, ELWL=0, NLSL=-7`
- `max_phase_override`: `overridden=false`, `original_action=NLSL`, `final_action=NLSL`, `target_phase=ETWT`, `protect_reason=left_wait_protect:NLSL=232.167>=60.000`, `gap=7`, `rule_gap=2`, `left_wait_protect=60`

### S2-6 Jinan Real 2000 Latest: Rule Override To ETWT

- source: `data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
- intersection/date: `intersection_1_3`, `1970-01-01 00:59:30`
- latest movement counts: `NT=3, NL=1, ST=4, SL=2, ET=4, EL=0, WT=10, WL=0`
- record prompt excerpt: same long raw prompt schema as S2-4
- actual short task prompt when `prompt_mode=image_only_short`: same as the short prompt in the 字段口径小节
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=2/36,NL=1/195,ST=4/7.8,SL=2/107.5,ET=3/42.7,EL=0/0,WT=7/31.7,WL=0/0; phases ETWT=10/37.2,NTST=6/21.9,ELWL=0/0,NLSL=3/151.2; maxQ=ETWT(10); maxW=NLSL(151.2); totalQ=19.
```

- `clip_caption`: `Highest demand on WT(10) and ST(4), top-2 share 58%. Through lanes high, left-turn lanes low; overall demand rising. East-West share 58%, North-South share 42% (east-west dominant).`
- expert decision: `<signal>ETWT</signal>`
- `base_action`: `ELWL`
- `phase_queue`: `ETWT=10, NTST=6, ELWL=0, NLSL=3`
- `phase_wait`: `ETWT=37.2, NTST=21.9, ELWL=0, NLSL=151.2`
- `candidate_action_waits`: `ETWT=-10, NTST=-6, ELWL=0, NLSL=-3`
- `max_phase_override`: `overridden=true`, `original_action=ELWL`, `final_action=ETWT`, `target_phase=ETWT`, `gap=10`, `rule_gap=2`, `left_wait_protect=60`

### S2-7 Jinan Real 2500 Latest: Normal High-Queue ELWL

- source: `data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl`
- intersection/date: `intersection_3_3`, `1970-01-01 00:37:30`
- latest movement counts: `NT=8, NL=0, ST=9, SL=3, ET=0, EL=6, WT=1, WL=7`
- record prompt excerpt: same long raw prompt schema as S2-4
- actual short task prompt when `prompt_mode=image_only_short`: same as the short prompt in the 字段口径小节
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=3/20,NL=0/0,ST=6/23.8,SL=1/5,ET=0/0,EL=5/73.6,WT=0/0,WL=7/64.7; phases ETWT=0/0,NTST=9/21.9,ELWL=12/69.2,NLSL=1/2.5; maxQ=ELWL(12); maxW=ELWL(69.2); totalQ=22.
```

- `clip_caption`: `Highest demand on ST(9) and NT(8), top-2 share 50%. Through lanes moderate, left-turn lanes moderate; overall demand declining. East-West share 41%, North-South share 59% (north-south dominant).`
- expert decision: `<signal>ELWL</signal>`
- `base_action`: `ELWL`
- `phase_queue`: `ETWT=0, NTST=9, ELWL=12, NLSL=1`
- `phase_wait`: `ETWT=0, NTST=21.9, ELWL=69.2, NLSL=2.5`
- `candidate_action_waits`: `ETWT=0, NTST=-9, ELWL=-12, NLSL=-1`
- `max_phase_override`: `overridden=false`, `original_action=ELWL`, `final_action=ELWL`, `target_phase=ELWL`, `rule_gap=3`, `left_wait_protect=60`

### S2-8 Jinan Real 2500 Latest: Normal NLSL

- source: `data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl`
- intersection/date: `intersection_1_1`, `1970-01-01 00:31:30`
- latest movement counts: `NT=5, NL=1, ST=6, SL=7, ET=6, EL=2, WT=9, WL=2`
- record prompt excerpt: same long raw prompt schema as S2-4
- actual short task prompt when `prompt_mode=image_only_short`: same as the short prompt in the 字段口径小节
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=5/4.4,NL=1/66,ST=1/6,SL=6/72.3,ET=1/42,EL=0/0,WT=6/24.3,WL=0/0; phases ETWT=7/33.2,NTST=6/5.2,ELWL=0/0,NLSL=7/69.2; maxQ=ETWT(7); maxW=NLSL(69.2); totalQ=20.
```

- `clip_caption`: `Highest demand on WT(9) and SL(7), top-2 share 42%. Through lanes moderate, left-turn lanes moderate; overall demand steady. East-West share 50%, North-South share 50% (balanced).`
- expert decision: `<signal>NLSL</signal>`
- `base_action`: `NLSL`
- `phase_queue`: `ETWT=7, NTST=6, ELWL=0, NLSL=7`
- `phase_wait`: `ETWT=33.2, NTST=5.2, ELWL=0, NLSL=69.2`
- `candidate_action_waits`: `ETWT=-7, NTST=-6, ELWL=0, NLSL=-7`
- `max_phase_override`: `overridden=false`, `original_action=NLSL`, `final_action=NLSL`, `target_phase=ETWT`, `rule_gap=3`, `left_wait_protect=60`

### S2-9 Jinan Real 2500 Latest: Rule Override To NTST

- source: `data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl`
- intersection/date: `intersection_1_2`, `1970-01-01 00:34:00`
- latest movement counts: `NT=9, NL=0, ST=9, SL=1, ET=8, EL=4, WT=3, WL=6`
- record prompt excerpt: same long raw prompt schema as S2-4
- actual short task prompt when `prompt_mode=image_only_short`: same as the short prompt in the 字段口径小节
- `prompt_context`:

```text
Traffic q/w(s): lanes NT=7/22.4,NL=0/0,ST=3/3.3,SL=0/0,ET=5/10.8,EL=3/98.7,WT=1/21,WL=4/79.5; phases ETWT=6/15.9,NTST=10/12.9,ELWL=7/89.1,NLSL=0/0; maxQ=NTST(10); maxW=ELWL(89.1); totalQ=23.
```

- `clip_caption`: `Highest demand on NT(9) and ST(9), top-2 share 45%. Through lanes moderate, left-turn lanes moderate; overall demand steady. East-West share 52%, North-South share 48% (balanced).`
- expert decision: `<signal>NTST</signal>`
- `base_action`: `ETWT`
- `phase_queue`: `ETWT=6, NTST=10, ELWL=7, NLSL=0`
- `phase_wait`: `ETWT=15.9, NTST=12.9, ELWL=89.1, NLSL=0`
- `candidate_action_waits`: `ETWT=-6, NTST=-10, ELWL=-7, NLSL=0`
- `max_phase_override`: `overridden=true`, `original_action=ETWT`, `final_action=NTST`, `target_phase=NTST`, `gap=4`, `rule_gap=3`, `left_wait_protect=60`

## 小结

- Stage 1 样例展示的是 `prompt/sequence` 与 `clip_caption` 的对应关系；`expert_action` 只作为辅助字段列出。
- Stage 2 样例使用最新 Hangzhou、Jinan2000、Jinan2500 processed 数据，并同时列出 `prompt`、`prompt_context`、`clip_caption`、`expert_action`、队列/等待统计、候选动作等待值和 rule override 信息。
- 因此 Stage 2 部分不是 prompt-only，而是完整展示了提示词、结构化交通状态和专家决策之间的对应关系。
