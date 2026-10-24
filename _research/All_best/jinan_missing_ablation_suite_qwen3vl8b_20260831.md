# Jinan Missing Ablation Suite Qwen3-VL-8B Results

Generated: 2026-08-31

## Run Identity

- Run root: `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412`
- Model path: `/mnt/data/Qwen3-VL-8B-Instruct`
- Dataset: `jinan`
- Traffic file: `anon_3_4_jinan_real.json`
- Same-flow online evaluation: yes
- Training seeds: `2024`, `42`, `123`
- Online seeds per trained checkpoint: `2024`, `42`, `123`, `7`
- Online horizon: `run_counts=3600`, `num_rounds=1`
- Prompt mode: `queue_wait_short`
- Shared model settings: `seq_len=16`, `image_size=224`, `input_dim=8`, `hidden_dim=64`, `prefix_tokens=8`, `prefix_mode=prefix`, `max_length=64`, `bf16`
- Training data files:

  - `/home/code/LLMSTSC/data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
  - `/home/code/LLMSTSC/data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
  - `/home/code/LLMSTSC/data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl`

## Completion Status

- Experiment status: complete
- Completed online rows: `108/108`
- Completed arms: `control_shuffled_targets, stage1_random_projector, hard_bc, hard_plus_soft, hard_soft_ranking, full_loss, visual_shuffled_image, visual_constant_image, visual_zero_prefix`
- Final status row: `2026-08-31T22:48:14+08:00	suite	all	all	all	completed	/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/RESULTS.md	done`
- Result artifacts:
  - `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/aggregate.json`
  - `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/aggregate.csv`
  - `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/RESULTS.md`

| Arm | Training seeds | Online rows | Status |
| --- | ---: | ---: | ---: |
| `control_shuffled_targets` | 3 | 12 | complete |
| `stage1_random_projector` | 3 | 12 | complete |
| `hard_bc` | 3 | 12 | complete |
| `hard_plus_soft` | 3 | 12 | complete |
| `hard_soft_ranking` | 3 | 12 | complete |
| `full_loss` | 3 | 12 | complete |
| `visual_shuffled_image` | 3 | 12 | complete |
| `visual_constant_image` | 3 | 12 | complete |
| `visual_zero_prefix` | 3 | 12 | complete |

## Main Conclusion

The most defensible conclusion is that the Jinan Qwen3-VL-8B controller depends strongly on real state-conditioned input and real action labels. Shuffling labels destroys performance, and removing or corrupting the visual/prefix signal causes large closed-loop degradation. In contrast, the individual loss-component ablations are close to each other; they do not support a strong claim that every loss term is independently necessary.

Key points:

1. `control_shuffled_targets` is a strong negative control: AWT rises from `47.4457` in `hard_bc` to `500.0050`, and AQL rises from `168.4049` to `879.9951`.
2. Visual/prefix information is crucial: relative to `full_loss`, `visual_shuffled_image` worsens AQL by `17.90%` and AWT by `18.68%`; `visual_constant_image` worsens AQL by `78.66%` and AWT by `154.79%`; `visual_zero_prefix` collapses the policy.
3. `full_loss` is best on AQL and ATT, but not on AWT. `hard_bc` has the lowest AWT. Therefore the paper should not claim that full loss dominates all metrics.
4. `stage1_random_projector` is close to the trained Stage-1 variants, so this Jinan suite does not provide strong evidence that Stage-1 initialization is decisive after Stage-2 training.

Recommended paper-safe claim: real labels and state-conditioned visual/prefix input are necessary for good closed-loop Jinan performance; the full loss recipe gives the best queue length and travel-time averages, while waiting-time advantages are not uniformly improved over hard BC.

## Aggregate Online Results

Values are mean +/- standard deviation across the three training seeds, where each training-seed mean is computed from four online simulator seeds. Reward is higher-is-better; all other metrics are lower-is-better.

| Arm | Train seeds | Online rows | Reward | AQL | AWT | ATT | Wait p95 | Wait max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_shuffled_targets` | 3 | 12 | -1060.1300 +/- 60.1186 | 879.9951 +/- 157.1032 | 500.0050 +/- 118.3984 | 669.1900 +/- 85.7240 | 2198.5292 +/- 472.9962 | 3391.5000 +/- 147.1148 |
| `stage1_random_projector` | 3 | 12 | -616.2061 +/- 8.1985 | 169.1583 +/- 7.5602 | 47.8268 +/- 0.4348 | 276.2181 +/- 5.1439 | 186.9667 +/- 7.1557 | 759.0000 +/- 50.2295 |
| `hard_bc` | 3 | 12 | -617.1586 +/- 3.9432 | 168.4049 +/- 3.5301 | 47.4457 +/- 1.6046 | 275.6083 +/- 2.1255 | 190.3042 +/- 4.4871 | 732.5000 +/- 232.5581 |
| `hard_plus_soft` | 3 | 12 | -615.1164 +/- 9.6406 | 166.7500 +/- 5.4862 | 50.2881 +/- 5.8778 | 274.2548 +/- 2.8865 | 201.4333 +/- 17.2138 | 772.3333 +/- 201.9216 |
| `hard_soft_ranking` | 3 | 12 | -617.8069 +/- 7.9331 | 167.9528 +/- 4.8394 | 49.5287 +/- 3.1070 | 275.2351 +/- 3.3010 | 201.2500 +/- 14.1664 | 720.4167 +/- 55.4529 |
| `full_loss` | 3 | 12 | -616.5350 +/- 13.7546 | 166.2792 +/- 7.9498 | 51.2086 +/- 5.3686 | 273.9426 +/- 4.7307 | 204.0000 +/- 20.0749 | 768.3333 +/- 155.8728 |
| `visual_shuffled_image` | 3 | 12 | -658.1869 +/- 19.1547 | 196.0424 +/- 17.7876 | 60.7750 +/- 7.6061 | 292.8950 +/- 9.9806 | 243.8667 +/- 16.0266 | 851.0000 +/- 272.8351 |
| `visual_constant_image` | 3 | 12 | -776.9927 +/- 63.1304 | 297.0806 +/- 85.6339 | 130.4734 +/- 24.6769 | 351.3983 +/- 51.4678 | 504.0333 +/- 133.9029 | 1506.3333 +/- 339.9593 |
| `visual_zero_prefix` | 3 | 12 | -1333.8764 +/- 0.0000 | 1729.2000 +/- 0.0000 | 1019.2841 +/- 0.0000 | 1579.1160 +/- 0.0000 | 3431.0000 +/- 0.0000 | 3561.0000 +/- 0.0000 |

## Delta Versus Full Loss

Positive Reward delta is better. Negative AQL/AWT/ATT delta is better.

| Arm | Reward delta | AQL delta | AWT delta | ATT delta |
| --- | ---: | ---: | ---: | ---: |
| `control_shuffled_targets` | -443.5950 (-71.95%, worse) | 713.7160 (+429.23%, worse) | 448.7964 (+876.41%, worse) | 395.2474 (+144.28%, worse) |
| `stage1_random_projector` | 0.3289 (+0.05%, better) | 2.8792 (+1.73%, worse) | -3.3818 (-6.60%, better) | 2.2756 (+0.83%, worse) |
| `hard_bc` | -0.6235 (-0.10%, worse) | 2.1257 (+1.28%, worse) | -3.7630 (-7.35%, better) | 1.6657 (+0.61%, worse) |
| `hard_plus_soft` | 1.4186 (+0.23%, better) | 0.4708 (+0.28%, worse) | -0.9206 (-1.80%, better) | 0.3122 (+0.11%, worse) |
| `hard_soft_ranking` | -1.2718 (-0.21%, worse) | 1.6736 (+1.01%, worse) | -1.6800 (-3.28%, better) | 1.2925 (+0.47%, worse) |
| `visual_shuffled_image` | -41.6519 (-6.76%, worse) | 29.7632 (+17.90%, worse) | 9.5664 (+18.68%, worse) | 18.9524 (+6.92%, worse) |
| `visual_constant_image` | -160.4577 (-26.03%, worse) | 130.8014 (+78.66%, worse) | 79.2647 (+154.79%, worse) | 77.4557 (+28.27%, worse) |
| `visual_zero_prefix` | -717.3414 (-116.35%, worse) | 1562.9208 (+939.94%, worse) | 968.0755 (+1890.45%, worse) | 1305.1735 (+476.44%, worse) |

## Delta Versus Hard BC

This table is most useful for the fixed-data loss ablation. Positive Reward delta is better. Negative AQL/AWT/ATT delta is better.

| Arm | Reward delta | AQL delta | AWT delta | ATT delta |
| --- | ---: | ---: | ---: | ---: |
| `stage1_random_projector` | 0.9524 (+0.15%, better) | 0.7535 (+0.45%, worse) | 0.3811 (+0.80%, worse) | 0.6098 (+0.22%, worse) |
| `hard_plus_soft` | 2.0421 (+0.33%, better) | -1.6549 (-0.98%, better) | 2.8424 (+5.99%, worse) | -1.3535 (-0.49%, better) |
| `hard_soft_ranking` | -0.6483 (-0.11%, worse) | -0.4521 (-0.27%, better) | 2.0830 (+4.39%, worse) | -0.3732 (-0.14%, better) |
| `full_loss` | 0.6235 (+0.10%, better) | -2.1257 (-1.26%, better) | 3.7630 (+7.93%, worse) | -1.6657 (-0.60%, better) |
| `control_shuffled_targets` | -442.9714 (-71.78%, worse) | 711.5903 (+422.55%, worse) | 452.5593 (+953.85%, worse) | 393.5817 (+142.80%, worse) |

## Paired Differences Versus Full Loss

These are row-level paired differences over matched training seed and online seed. Positive Reward delta is better; negative AQL/AWT/ATT delta is better.

| Arm | Reward delta | AQL delta | AWT delta | ATT delta |
| --- | ---: | ---: | ---: | ---: |
| `hard_bc` | -0.6235 +/- 9.3073 | 2.1257 +/- 6.4102 | -3.7630 +/- 3.2971 | 1.6657 +/- 3.8827 |
| `hard_plus_soft` | 1.4186 +/- 4.5384 | 0.4708 +/- 2.7823 | -0.9206 +/- 0.4404 | 0.3122 +/- 1.7682 |
| `hard_soft_ranking` | -1.2718 +/- 16.9859 | 1.6736 +/- 9.8806 | -1.6800 +/- 7.0081 | 1.2925 +/- 6.0718 |
| `visual_shuffled_image` | -41.6519 +/- 24.6000 | 29.7632 +/- 19.1751 | 9.5664 +/- 9.6074 | 18.9524 +/- 11.2010 |
| `visual_constant_image` | -160.4577 +/- 64.0477 | 130.8014 +/- 78.2623 | 79.2647 +/- 24.2565 | 77.4557 +/- 47.2510 |
| `visual_zero_prefix` | -717.3414 +/- 11.7300 | 1562.9208 +/- 6.7796 | 968.0755 +/- 4.5783 | 1305.1735 +/- 4.0343 |

## Per-Training-Seed Summary

Each row is mean +/- standard deviation across the four online seeds for one trained checkpoint.

| Arm | Train seed | Online n | Reward | AQL | AWT | ATT | Wait p95 | Wait max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_shuffled_targets` | 2024 | 4 | -1129.4972 +/- 0.6657 | 1058.6771 +/- 2.0625 | 634.8726 +/- 0.1907 | 767.2588 +/- 0.6843 | 2733.3500 +/- 4.9000 | 3561.0000 +/- 0.0000 |
| `control_shuffled_targets` | 42 | 4 | -1023.1237 +/- 0.0000 | 763.5250 +/- 0.0000 | 413.1747 +/- 0.0000 | 608.5152 +/- 0.0002 | 1835.2000 +/- 0.0000 | 3297.0000 +/- 0.0000 |
| `control_shuffled_targets` | 123 | 4 | -1027.7690 +/- 1.7537 | 817.7833 +/- 1.5730 | 451.9677 +/- 15.4610 | 631.7960 +/- 1.7638 | 2027.0375 +/- 44.7101 | 3316.5000 +/- 69.0579 |
| `stage1_random_projector` | 2024 | 4 | -607.9983 +/- 0.0000 | 162.9000 +/- 0.0000 | 48.1983 +/- 0.0000 | 272.0252 +/- 0.0002 | 194.9000 +/- 0.0000 | 701.0000 +/- 0.0000 |
| `stage1_random_projector` | 42 | 4 | -624.3952 +/- 0.0000 | 177.5583 +/- 0.0000 | 47.9335 +/- 0.0000 | 281.9579 +/- 0.0001 | 181.0000 +/- 0.0000 | 788.0000 +/- 0.0000 |
| `stage1_random_projector` | 123 | 4 | -616.2249 +/- 0.0000 | 167.0167 +/- 0.0000 | 47.3486 +/- 0.0000 | 274.6713 +/- 0.0001 | 185.0000 +/- 0.0000 | 788.0000 +/- 0.0000 |
| `hard_bc` | 2024 | 4 | -617.2340 +/- 0.0000 | 170.9917 +/- 0.0000 | 46.2647 +/- 0.0000 | 277.3620 +/- 0.0000 | 188.5000 +/- 0.0000 | 668.0000 +/- 0.0000 |
| `hard_bc` | 42 | 4 | -613.1782 +/- 0.0000 | 164.3833 +/- 0.0000 | 46.7998 +/- 0.0001 | 273.2445 +/- 0.0002 | 187.0000 +/- 0.0000 | 539.0000 +/- 0.0000 |
| `hard_bc` | 123 | 4 | -621.0635 +/- 4.3465 | 169.8396 +/- 2.7687 | 49.2726 +/- 1.4095 | 276.2183 +/- 2.1717 | 195.4125 +/- 5.7603 | 990.5000 +/- 147.0000 |
| `hard_plus_soft` | 2024 | 4 | -612.5240 +/- 0.0000 | 165.7750 +/- 0.0000 | 46.3891 +/- 0.0000 | 273.8824 +/- 0.0001 | 192.0500 +/- 0.0000 | 606.0000 +/- 0.0000 |
| `hard_plus_soft` | 42 | 4 | -607.0371 +/- 0.0000 | 161.8167 +/- 0.0000 | 47.4264 +/- 0.0000 | 271.5726 +/- 0.0001 | 190.9500 +/- 0.0000 | 714.0000 +/- 0.0000 |
| `hard_plus_soft` | 123 | 4 | -625.7882 +/- 0.0000 | 172.6583 +/- 0.0000 | 57.0487 +/- 0.0000 | 277.3094 +/- 0.0002 | 221.3000 +/- 0.0000 | 997.0000 +/- 0.0000 |
| `hard_soft_ranking` | 2024 | 4 | -626.8260 +/- 0.0000 | 173.4500 +/- 0.0000 | 52.8059 +/- 0.0000 | 278.9588 +/- 0.0002 | 216.0000 +/- 0.0000 | 764.0000 +/- 0.0000 |
| `hard_soft_ranking` | 42 | 4 | -614.6850 +/- 1.0586 | 166.0729 +/- 1.0042 | 49.1542 +/- 0.1575 | 274.0780 +/- 0.6979 | 200.0000 +/- 4.0000 | 739.2500 +/- 26.5000 |
| `hard_soft_ranking` | 123 | 4 | -611.9096 +/- 2.6981 | 164.3354 +/- 0.3375 | 46.6258 +/- 0.6744 | 272.6684 +/- 0.1357 | 187.7500 +/- 3.5000 | 658.0000 +/- 20.0000 |
| `full_loss` | 2024 | 4 | -608.5160 +/- 0.0000 | 162.0417 +/- 0.0000 | 47.5702 +/- 0.0000 | 271.9719 +/- 0.0000 | 195.0000 +/- 0.0000 | 638.0000 +/- 0.0000 |
| `full_loss` | 42 | 4 | -608.6718 +/- 0.0137 | 161.3458 +/- 0.0048 | 48.6813 +/- 0.0010 | 270.5158 +/- 0.0000 | 190.0000 +/- 0.0000 | 726.0000 +/- 0.0000 |
| `full_loss` | 123 | 4 | -632.4172 +/- 0.0000 | 175.4500 +/- 0.0000 | 57.3744 +/- 0.0000 | 279.3400 +/- 0.0002 | 227.0000 +/- 0.0000 | 941.0000 +/- 0.0000 |
| `visual_shuffled_image` | 2024 | 4 | -647.7727 +/- 0.0000 | 183.9500 +/- 0.0000 | 57.1262 +/- 0.0000 | 286.0382 +/- 0.0001 | 238.0000 +/- 0.0000 | 689.0000 +/- 0.0000 |
| `visual_shuffled_image` | 42 | 4 | -680.2926 +/- 0.0000 | 216.4667 +/- 0.0000 | 69.5180 +/- 0.0000 | 304.3453 +/- 0.0002 | 262.0000 +/- 0.0000 | 1166.0000 +/- 0.0000 |
| `visual_shuffled_image` | 123 | 4 | -646.4954 +/- 0.1315 | 187.7104 +/- 0.3181 | 55.6809 +/- 0.1667 | 288.3015 +/- 0.0034 | 231.6000 +/- 1.2000 | 698.0000 +/- 0.0000 |
| `visual_constant_image` | 2024 | 4 | -774.1471 +/- 0.4447 | 273.4229 +/- 0.3292 | 123.5350 +/- 0.0747 | 336.9988 +/- 0.1075 | 439.0750 +/- 0.0500 | 1236.0000 +/- 0.0000 |
| `visual_constant_image` | 42 | 4 | -841.4978 +/- 1.8365 | 392.0562 +/- 2.1963 | 157.8767 +/- 0.9095 | 408.5323 +/- 0.9485 | 658.0250 +/- 0.9350 | 1888.0000 +/- 0.0000 |
| `visual_constant_image` | 123 | 4 | -715.3333 +/- 1.3933 | 225.7625 +/- 1.8620 | 110.0084 +/- 1.4029 | 308.6639 +/- 0.6886 | 415.0000 +/- 30.0222 | 1395.0000 +/- 167.4316 |
| `visual_zero_prefix` | 2024 | 4 | -1333.8764 +/- 0.0000 | 1729.2000 +/- 0.0000 | 1019.2841 +/- 0.0000 | 1579.1160 +/- 0.0000 | 3431.0000 +/- 0.0000 | 3561.0000 +/- 0.0000 |
| `visual_zero_prefix` | 42 | 4 | -1333.8764 +/- 0.0000 | 1729.2000 +/- 0.0000 | 1019.2841 +/- 0.0000 | 1579.1160 +/- 0.0000 | 3431.0000 +/- 0.0000 | 3561.0000 +/- 0.0000 |
| `visual_zero_prefix` | 123 | 4 | -1333.8764 +/- 0.0000 | 1729.2000 +/- 0.0000 | 1019.2841 +/- 0.0000 | 1579.1160 +/- 0.0000 | 3431.0000 +/- 0.0000 | 3561.0000 +/- 0.0000 |

## Full Raw Online Results

This table contains every completed online result row: 9 arms x 3 training seeds x 4 online seeds = 108 rows.

| Arm | Train seed | Online seed | Reward | AQL | AWT | ATT | Wait p95 | Wait max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_shuffled_targets` | 2024 | 2024 | -1129.8300 | 1059.7083 | 634.7772 | 767.6010 | 2735.8000 | 3561.0000 |
| `control_shuffled_targets` | 2024 | 42 | -1129.8300 | 1059.7083 | 634.7772 | 767.6010 | 2735.8000 | 3561.0000 |
| `control_shuffled_targets` | 2024 | 123 | -1128.4987 | 1055.5833 | 635.1586 | 766.2323 | 2726.0000 | 3561.0000 |
| `control_shuffled_targets` | 2024 | 7 | -1129.8300 | 1059.7083 | 634.7772 | 767.6008 | 2735.8000 | 3561.0000 |
| `control_shuffled_targets` | 42 | 2024 | -1023.1237 | 763.5250 | 413.1747 | 608.5154 | 1835.2000 | 3297.0000 |
| `control_shuffled_targets` | 42 | 42 | -1023.1237 | 763.5250 | 413.1747 | 608.5150 | 1835.2000 | 3297.0000 |
| `control_shuffled_targets` | 42 | 123 | -1023.1237 | 763.5250 | 413.1747 | 608.5154 | 1835.2000 | 3297.0000 |
| `control_shuffled_targets` | 42 | 7 | -1023.1237 | 763.5250 | 413.1747 | 608.5152 | 1835.2000 | 3297.0000 |
| `control_shuffled_targets` | 123 | 2024 | -1026.3359 | 815.7000 | 457.8030 | 629.1891 | 2053.2500 | 3325.0000 |
| `control_shuffled_targets` | 123 | 42 | -1030.3266 | 817.4333 | 471.1844 | 632.2395 | 2076.5000 | 3411.0000 |
| `control_shuffled_targets` | 123 | 123 | -1027.2068 | 819.0000 | 439.4418 | 632.8778 | 1989.2000 | 3265.0000 |
| `control_shuffled_targets` | 123 | 7 | -1027.2068 | 819.0000 | 439.4418 | 632.8775 | 1989.2000 | 3265.0000 |
| `stage1_random_projector` | 2024 | 2024 | -607.9983 | 162.9000 | 48.1983 | 272.0251 | 194.9000 | 701.0000 |
| `stage1_random_projector` | 2024 | 42 | -607.9983 | 162.9000 | 48.1983 | 272.0251 | 194.9000 | 701.0000 |
| `stage1_random_projector` | 2024 | 123 | -607.9983 | 162.9000 | 48.1983 | 272.0251 | 194.9000 | 701.0000 |
| `stage1_random_projector` | 2024 | 7 | -607.9983 | 162.9000 | 48.1983 | 272.0254 | 194.9000 | 701.0000 |
| `stage1_random_projector` | 42 | 2024 | -624.3952 | 177.5583 | 47.9335 | 281.9579 | 181.0000 | 788.0000 |
| `stage1_random_projector` | 42 | 42 | -624.3952 | 177.5583 | 47.9336 | 281.9579 | 181.0000 | 788.0000 |
| `stage1_random_projector` | 42 | 123 | -624.3952 | 177.5583 | 47.9335 | 281.9577 | 181.0000 | 788.0000 |
| `stage1_random_projector` | 42 | 7 | -624.3952 | 177.5583 | 47.9336 | 281.9581 | 181.0000 | 788.0000 |
| `stage1_random_projector` | 123 | 2024 | -616.2249 | 167.0167 | 47.3486 | 274.6713 | 185.0000 | 788.0000 |
| `stage1_random_projector` | 123 | 42 | -616.2249 | 167.0167 | 47.3486 | 274.6715 | 185.0000 | 788.0000 |
| `stage1_random_projector` | 123 | 123 | -616.2249 | 167.0167 | 47.3486 | 274.6712 | 185.0000 | 788.0000 |
| `stage1_random_projector` | 123 | 7 | -616.2249 | 167.0167 | 47.3486 | 274.6713 | 185.0000 | 788.0000 |
| `hard_bc` | 2024 | 2024 | -617.2340 | 170.9917 | 46.2646 | 277.3620 | 188.5000 | 668.0000 |
| `hard_bc` | 2024 | 42 | -617.2340 | 170.9917 | 46.2647 | 277.3620 | 188.5000 | 668.0000 |
| `hard_bc` | 2024 | 123 | -617.2340 | 170.9917 | 46.2647 | 277.3620 | 188.5000 | 668.0000 |
| `hard_bc` | 2024 | 7 | -617.2340 | 170.9917 | 46.2647 | 277.3620 | 188.5000 | 668.0000 |
| `hard_bc` | 42 | 2024 | -613.1782 | 164.3833 | 46.7999 | 273.2448 | 187.0000 | 539.0000 |
| `hard_bc` | 42 | 42 | -613.1782 | 164.3833 | 46.7998 | 273.2445 | 187.0000 | 539.0000 |
| `hard_bc` | 42 | 123 | -613.1782 | 164.3833 | 46.7998 | 273.2443 | 187.0000 | 539.0000 |
| `hard_bc` | 42 | 7 | -613.1782 | 164.3833 | 46.7996 | 273.2443 | 187.0000 | 539.0000 |
| `hard_bc` | 123 | 2024 | -619.6437 | 168.6417 | 48.6443 | 275.0637 | 193.0000 | 917.0000 |
| `hard_bc` | 123 | 42 | -627.4135 | 173.9750 | 51.3809 | 279.4726 | 204.0000 | 1211.0000 |
| `hard_bc` | 123 | 123 | -617.5530 | 168.1000 | 48.4208 | 275.2732 | 191.6500 | 917.0000 |
| `hard_bc` | 123 | 7 | -619.6437 | 168.6417 | 48.6444 | 275.0639 | 193.0000 | 917.0000 |
| `hard_plus_soft` | 2024 | 2024 | -612.5240 | 165.7750 | 46.3891 | 273.8824 | 192.0500 | 606.0000 |
| `hard_plus_soft` | 2024 | 42 | -612.5240 | 165.7750 | 46.3891 | 273.8824 | 192.0500 | 606.0000 |
| `hard_plus_soft` | 2024 | 123 | -612.5240 | 165.7750 | 46.3891 | 273.8824 | 192.0500 | 606.0000 |
| `hard_plus_soft` | 2024 | 7 | -612.5240 | 165.7750 | 46.3891 | 273.8823 | 192.0500 | 606.0000 |
| `hard_plus_soft` | 42 | 2024 | -607.0371 | 161.8167 | 47.4264 | 271.5727 | 190.9500 | 714.0000 |
| `hard_plus_soft` | 42 | 42 | -607.0371 | 161.8167 | 47.4264 | 271.5725 | 190.9500 | 714.0000 |
| `hard_plus_soft` | 42 | 123 | -607.0371 | 161.8167 | 47.4264 | 271.5725 | 190.9500 | 714.0000 |
| `hard_plus_soft` | 42 | 7 | -607.0371 | 161.8167 | 47.4264 | 271.5725 | 190.9500 | 714.0000 |
| `hard_plus_soft` | 123 | 2024 | -625.7882 | 172.6583 | 57.0487 | 277.3093 | 221.3000 | 997.0000 |
| `hard_plus_soft` | 123 | 42 | -625.7882 | 172.6583 | 57.0487 | 277.3093 | 221.3000 | 997.0000 |
| `hard_plus_soft` | 123 | 123 | -625.7882 | 172.6583 | 57.0487 | 277.3093 | 221.3000 | 997.0000 |
| `hard_plus_soft` | 123 | 7 | -625.7882 | 172.6583 | 57.0487 | 277.3096 | 221.3000 | 997.0000 |
| `hard_soft_ranking` | 2024 | 2024 | -626.8260 | 173.4500 | 52.8060 | 278.9590 | 216.0000 | 764.0000 |
| `hard_soft_ranking` | 2024 | 42 | -626.8260 | 173.4500 | 52.8059 | 278.9587 | 216.0000 | 764.0000 |
| `hard_soft_ranking` | 2024 | 123 | -626.8260 | 173.4500 | 52.8059 | 278.9589 | 216.0000 | 764.0000 |
| `hard_soft_ranking` | 2024 | 7 | -626.8260 | 173.4500 | 52.8059 | 278.9587 | 216.0000 | 764.0000 |
| `hard_soft_ranking` | 42 | 2024 | -613.0970 | 164.5667 | 49.3905 | 273.0311 | 194.0000 | 779.0000 |
| `hard_soft_ranking` | 42 | 42 | -615.2143 | 166.5750 | 49.0754 | 274.4267 | 202.0000 | 726.0000 |
| `hard_soft_ranking` | 42 | 123 | -615.2143 | 166.5750 | 49.0755 | 274.4270 | 202.0000 | 726.0000 |
| `hard_soft_ranking` | 42 | 7 | -615.2143 | 166.5750 | 49.0755 | 274.4270 | 202.0000 | 726.0000 |
| `hard_soft_ranking` | 123 | 2024 | -610.5606 | 164.1667 | 46.2886 | 272.6006 | 186.0000 | 668.0000 |
| `hard_soft_ranking` | 123 | 42 | -610.5606 | 164.1667 | 46.2886 | 272.6005 | 186.0000 | 668.0000 |
| `hard_soft_ranking` | 123 | 123 | -610.5606 | 164.1667 | 46.2886 | 272.6005 | 186.0000 | 668.0000 |
| `hard_soft_ranking` | 123 | 7 | -615.9567 | 164.8417 | 47.6375 | 272.8720 | 193.0000 | 628.0000 |
| `full_loss` | 2024 | 2024 | -608.5160 | 162.0417 | 47.5702 | 271.9719 | 195.0000 | 638.0000 |
| `full_loss` | 2024 | 42 | -608.5160 | 162.0417 | 47.5702 | 271.9719 | 195.0000 | 638.0000 |
| `full_loss` | 2024 | 123 | -608.5160 | 162.0417 | 47.5702 | 271.9719 | 195.0000 | 638.0000 |
| `full_loss` | 2024 | 7 | -608.5160 | 162.0417 | 47.5702 | 271.9719 | 195.0000 | 638.0000 |
| `full_loss` | 42 | 2024 | -608.6837 | 161.3500 | 48.6805 | 270.5158 | 190.0000 | 726.0000 |
| `full_loss` | 42 | 42 | -608.6837 | 161.3500 | 48.6805 | 270.5158 | 190.0000 | 726.0000 |
| `full_loss` | 42 | 123 | -608.6599 | 161.3417 | 48.6821 | 270.5158 | 190.0000 | 726.0000 |
| `full_loss` | 42 | 7 | -608.6599 | 161.3417 | 48.6821 | 270.5158 | 190.0000 | 726.0000 |
| `full_loss` | 123 | 2024 | -632.4172 | 175.4500 | 57.3745 | 279.3400 | 227.0000 | 941.0000 |
| `full_loss` | 123 | 42 | -632.4172 | 175.4500 | 57.3745 | 279.3400 | 227.0000 | 941.0000 |
| `full_loss` | 123 | 123 | -632.4172 | 175.4500 | 57.3744 | 279.3403 | 227.0000 | 941.0000 |
| `full_loss` | 123 | 7 | -632.4172 | 175.4500 | 57.3745 | 279.3400 | 227.0000 | 941.0000 |
| `visual_shuffled_image` | 2024 | 2024 | -647.7727 | 183.9500 | 57.1262 | 286.0381 | 238.0000 | 689.0000 |
| `visual_shuffled_image` | 2024 | 42 | -647.7727 | 183.9500 | 57.1262 | 286.0381 | 238.0000 | 689.0000 |
| `visual_shuffled_image` | 2024 | 123 | -647.7727 | 183.9500 | 57.1262 | 286.0381 | 238.0000 | 689.0000 |
| `visual_shuffled_image` | 2024 | 7 | -647.7727 | 183.9500 | 57.1262 | 286.0383 | 238.0000 | 689.0000 |
| `visual_shuffled_image` | 42 | 2024 | -680.2926 | 216.4667 | 69.5180 | 304.3454 | 262.0000 | 1166.0000 |
| `visual_shuffled_image` | 42 | 42 | -680.2926 | 216.4667 | 69.5180 | 304.3454 | 262.0000 | 1166.0000 |
| `visual_shuffled_image` | 42 | 123 | -680.2926 | 216.4667 | 69.5181 | 304.3453 | 262.0000 | 1166.0000 |
| `visual_shuffled_image` | 42 | 7 | -680.2926 | 216.4667 | 69.5181 | 304.3451 | 262.0000 | 1166.0000 |
| `visual_shuffled_image` | 123 | 2024 | -646.5589 | 187.8667 | 55.7646 | 288.2986 | 231.0000 | 698.0000 |
| `visual_shuffled_image` | 123 | 42 | -646.5589 | 187.8667 | 55.7646 | 288.2986 | 231.0000 | 698.0000 |
| `visual_shuffled_image` | 123 | 123 | -646.2982 | 187.2333 | 55.4308 | 288.3036 | 233.4000 | 698.0000 |
| `visual_shuffled_image` | 123 | 7 | -646.5655 | 187.8750 | 55.7634 | 288.3053 | 231.0000 | 698.0000 |
| `visual_constant_image` | 2024 | 2024 | -774.8141 | 273.9167 | 123.4230 | 337.1601 | 439.0000 | 1236.0000 |
| `visual_constant_image` | 2024 | 42 | -773.9247 | 273.2583 | 123.5723 | 336.9450 | 439.1000 | 1236.0000 |
| `visual_constant_image` | 2024 | 123 | -773.9247 | 273.2583 | 123.5723 | 336.9450 | 439.1000 | 1236.0000 |
| `visual_constant_image` | 2024 | 7 | -773.9247 | 273.2583 | 123.5723 | 336.9450 | 439.1000 | 1236.0000 |
| `visual_constant_image` | 42 | 2024 | -839.3700 | 389.6750 | 157.1533 | 407.6294 | 657.8000 | 1888.0000 |
| `visual_constant_image` | 42 | 42 | -841.4139 | 392.9000 | 158.6601 | 409.2261 | 658.8500 | 1888.0000 |
| `visual_constant_image` | 42 | 123 | -841.3510 | 390.9667 | 157.0273 | 407.8060 | 656.8000 | 1888.0000 |
| `visual_constant_image` | 42 | 7 | -843.8563 | 394.6833 | 158.6662 | 409.4676 | 658.6500 | 1888.0000 |
| `visual_constant_image` | 123 | 2024 | -714.1266 | 224.1500 | 108.7935 | 308.0675 | 389.0000 | 1540.0000 |
| `visual_constant_image` | 123 | 42 | -716.5399 | 227.3750 | 111.2234 | 309.2602 | 441.0000 | 1250.0000 |
| `visual_constant_image` | 123 | 123 | -714.1266 | 224.1500 | 108.7935 | 308.0675 | 389.0000 | 1540.0000 |
| `visual_constant_image` | 123 | 7 | -716.5399 | 227.3750 | 111.2234 | 309.2602 | 441.0000 | 1250.0000 |
| `visual_zero_prefix` | 2024 | 2024 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 2024 | 42 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 2024 | 123 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 2024 | 7 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 42 | 2024 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 42 | 42 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 42 | 123 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 42 | 7 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 123 | 2024 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 123 | 42 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 123 | 123 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |
| `visual_zero_prefix` | 123 | 7 | -1333.8764 | 1729.2000 | 1019.2841 | 1579.1160 | 3431.0000 | 3561.0000 |

## Interpretation Notes

- `control_shuffled_targets` isolates label validity under the same-state pool and shows that the policy needs real action supervision.
- `visual_shuffled_image` preserves the image distribution but breaks image-to-intersection correspondence, so it is the cleanest visual state-dependence test among the visual interventions.
- `visual_constant_image` and `visual_zero_prefix` are stronger out-of-distribution interventions; they demonstrate dependency but should not be described as retrained text-only baselines.
- Loss terms should be discussed cautiously: AQL/ATT favor `full_loss`, AWT favors `hard_bc`, and the non-visual loss variants are all close relative to seed variance.
- The script explicitly did not claim modality 2x2 or full data-source matrix results because the clean local artifacts for those comparisons were not available.

## Explicit Gaps Not Claimed

- `data_source_matrix`: The local Jinan RuleOnly/DAgger/SimExpert datasets are not present as a shared same-state, equal-budget label pool with identical candidate-cost metadata. The shell runs same-state real-vs-shuffled label controls instead.
- `modality_2x2_factorial`: The current code has no parameter-matched numeric-only prefix encoder or clean switch for removing compact queue/wait cues during training. The script records this gap instead of producing a confounded pseudo-ablation.

## Source Files

- Launcher script: `/home/code/LLMSTSC/paper_additions/run_jinan_missing_ablation_suite.sh`
- Aggregate JSON: `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/aggregate.json`
- Aggregate CSV: `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/aggregate.csv`
- Original short report: `/home/code/LLMSTSC/records/pipeline_runs/jinan_missing_ablation_suite_qwen3vl8b_20260830_221412/RESULTS.md`
