# NewYork28x7 AWT100 Teacher-Heavy Two-Stage

- Run: `/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt100_teacherheavy_twostage_20260830_111958`
- Protocol: same-flow NewYork 28x7, online seeds 3031/3032/3033/3034.
- Stage A: teacher-heavy wait-first distillation.
- Stage B: initialize from Stage A, then recover ATT using stronger ATT/base-KL guard.

## Results

| Policy | Seeds | AWT | ATT | AQL | P95 wait | Max wait | AWT vs baseline | ATT vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch11_baseline | 4 | 249.09 | 1008.53 | 1763.87 | 1531.56 | 2753.25 | +0.00% | +0.00% |
| teacher_wait7_p30_g05 | 4 | 99.86 | 1085.77 | 2055.29 | 883.07 | 1741.75 | -59.91% | +7.66% |
| prev_awt6_attguard | 4 | 232.00 | 1134.11 | 2090.04 | 1651.80 | 2564.75 | -6.86% | +12.45% |
| stageA_wait_teacherheavy | 4 | 266.26 | 1325.45 | 2611.58 | 1852.92 | 2563.00 | +6.89% | +31.42% |
| stageB_att_recover_mild | 4 | 238.28 | 1258.33 | 2449.43 | 1723.95 | 2576.75 | -4.34% | +24.77% |
| stageB_att_recover_strong | 4 | 261.08 | 1280.50 | 2496.88 | 1788.47 | 2746.50 | +4.82% | +26.97% |

## Current Best New Candidate

`stageB_att_recover_mild` has AWT `238.28` and ATT `1258.33`.
