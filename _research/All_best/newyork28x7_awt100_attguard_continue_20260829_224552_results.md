# NewYork28x7 AWT100 ATT-guard continuation

- Run: `/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt100_attguard_continue_20260829_224552`
- Reused failed run data: `/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt100_attguard_20260828_205628`
- Protocol: same NewYork 28x7 flow, online seeds 3031/3032/3033/3034.
- Selection note: primary target is lower AWT; ATT is compared with the epoch-11 baseline.

## References

| Reference | AWT | ATT | AQL | P95 wait | Max wait |
|---|---:|---:|---:|---:|---:|
| epoch_11 baseline | 249.09 | 1008.53 | 1763.87 | 1531.56 | 2753.25 |
| teacher wait7_p30_g05 | 99.86 | 1085.77 | 2055.29 | 883.07 | 1741.75 |

## Student Results

| Policy | Seeds | AWT | AWT std | ATT | ATT delta vs baseline | AQL | P95 wait | Max wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awt6_attguard | 4 | 232.00 | 25.12 | 1134.11 | +12.45% | 2090.04 | 1651.80 | 2564.75 |
| awt5_attguard | 4 | 234.40 | 10.91 | 1163.42 | +15.36% | 2170.59 | 1544.70 | 2576.75 |
| awt7_attguard | 4 | 275.59 | 4.63 | 1192.04 | +18.20% | 2280.77 | 1804.41 | 2655.50 |

## Current Best

`awt6_attguard` is the current best by the continuation selector: AWT=232.00, ATT=1134.11, AQL=2090.04.
