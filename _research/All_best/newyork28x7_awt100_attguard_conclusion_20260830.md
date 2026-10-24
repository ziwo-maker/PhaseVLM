# NewYork28x7 AWT100 ATT-Guard Conclusion

## Completed Run

- Run: `/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt100_attguard_continue_20260829_224552`
- Status: `success`
- Protocol: same-flow NewYork 28x7, online seeds `3031 3032 3033 3034`
- Data reused from: `/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt100_attguard_20260828_205628/phase3_data/base_plus_wait_triggers.jsonl`

## Metrics

| Policy | AWT | ATT | AQL | P95 wait | Max wait | ATT vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| epoch11 baseline | 249.09 | 1008.53 | 1763.87 | 1531.56 | 2753.25 | 0.00% |
| teacher `wait7_p30_g05` | 99.86 | 1085.77 | 2055.29 | 883.07 | 1741.75 | +7.66% |
| `awt5_attguard` | 234.40 +/- 10.91 | 1163.42 | 2170.59 | 1544.70 | 2576.75 | +15.36% |
| `awt6_attguard` | 232.00 +/- 25.12 | 1134.11 | 2090.04 | 1651.80 | 2564.75 | +12.45% |
| `awt7_attguard` | 275.59 +/- 4.63 | 1192.04 | 2280.77 | 1804.41 | 2655.50 | +18.20% |

## Conclusion

The AWT100 ATT-guard student sweep did not solve the target. The rule-based teacher can reach AWT near 100, but the trained student policies stayed around AWT 232-276 and paid a large ATT penalty.

Best policy inside this sweep is `awt6_attguard`: AWT `232.00`, ATT `1134.11`, AQL `2090.04`, P95 wait `1651.80`, Max wait `2564.75`. It improves AWT by only `6.86%` vs epoch11 baseline while worsening ATT by `12.45%`.

The previous joint optimization remains stronger than this AWT100 continuation. In particular, `awt3_aql1_tail1_r003` from `/home/code/LLMSTSC/records/pipeline_runs/newyork28x7_awt_att_opt_20260828_115847_retry` reached AWT `202.26`, ATT `927.48`, and AQL `1521.81`, which dominates this run on AWT, ATT, and AQL.

## Next Experiment Rationale

The failure mode suggests that increasing AWT cost weights is not enough. The next run should change the training curriculum:

1. Stage A: train a teacher-heavy wait-first student using repeated trigger data and weak ATT/base-KL constraints.
2. Stage B: initialize from Stage A and add ATT/base-KL guard to pull travel time back toward baseline.
3. Evaluate both Stage A and Stage B checkpoints online on the same NewYork 28x7 flow.

The immediate target for the next run is not yet AWT 100, but a clear intermediate gate: `AWT < 180` while keeping ATT closer than the rule teacher or at least not worse than the previous failed students.
