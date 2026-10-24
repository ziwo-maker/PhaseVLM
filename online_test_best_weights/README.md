# PhaseVLM Elsevier 论文结果与权重

| 场景 | ATT | AQL | AWT | 权重 |
| --- | ---: | ---: | ---: | --- |
| Jinan-1 | 267.54 | 156.55 | 44.14 | [paper_main_table_near_jinan1.pt](anon_3_4_jinan_real/paper_main_table_near_jinan1.pt) |
| Jinan-2 | 262.40 | 100.08 | 36.74 | 未找到 |
| Jinan-3 | 255.87 | 126.67 | 40.86 | [paper_main_table_near_jinan3.pt](anon_3_4_jinan_real_2500/paper_main_table_near_jinan3.pt) |
| Hangzhou-1 | 307.36 | 57.84 | 41.54 | 未找到 |
| Hangzhou-2 | 320.26 | 170.41 | 55.30 | [epoch 9 权重](anon_4_4_hangzhou_real_5816/epoch_9.pt) |

表中数值来自 Elsevier 论文主结果表。列出的权重是已找到的近似匹配权重，未确认为论文原始权重。

Hangzhou-2 仅归档 epoch 9。其 ATT 单项刷新了本地同流量历史记录；该单项更新不代表 AQL 或 AWT 也刷新了历史最优。评测和复现信息见 [Hangzhou-2 归档](anon_4_4_hangzhou_real_5816/reproduction.json)。
