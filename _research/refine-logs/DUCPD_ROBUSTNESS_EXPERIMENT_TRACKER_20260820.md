# DUCPD Robustness Experiment Tracker

| Run ID | Milestone | Purpose | System | Split / scenario | Metrics | Priority | Status |
|---|---|---|---|---|---|---|---|
| R001-R012 | M1 | Extend primary paired comparison to nine seeds | hard BC / DUCPD AWT=2 | six new training seeds | validation accuracy, checkpoint artifact | MUST | QUEUED |
| R013-R048 | M2 | Full-demand robustness | frozen checkpoints | target-real, source-5816, synthetic-high | reward, queue, mean wait, travel, vehicle p95/max wait | MUST | QUEUED |
| R049-R108 | M3 | Temporal robustness | frozen checkpoints | three target-real non-overlapping 1200-s windows | reward, queue, mean wait, travel, vehicle p95/max wait | MUST | QUEUED |
| R109 | M4 | Aggregate paired effects | hard BC vs DUCPD AWT=2 | all completed scenario rows | paired deltas, descriptive Wilcoxon, exclusions | MUST | QUEUED |
