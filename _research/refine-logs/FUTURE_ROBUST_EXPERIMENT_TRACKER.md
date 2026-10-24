# Future-Robust Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Future-label mechanism audit | policy-responsive H120 with future-only traffic perturbations | Hangzhou real, four time-stratified snapshots | flip rate, delta CVaR, feasibility, acceptance | MUST | COMPLETED (NO-GO) | Valid run `future_robust_hangzhou_audit_20260820_143959`: 4/4 snapshots, 96 rollouts, 0/4 accepted, mean flip rate 0.0. All 72 non-source future-candidates failed `awt_exceeded`; see R001 result analysis. |
| R002 | M1 | Tail-metric preflight | per-movement p95/max wait extraction | same pool | tail metric integrity | MUST | DEFERRED | No student training follows R001 NO-GO. Required again before any later student experiment. |
| R003 | M2 | Matched label efficacy | deterministic vs robust vs random same acceptance | two flows, matched seeds | AWT/ATT/AQL/tails/spillback | MUST if R001 GO | NO-GO | Not authorized: R001 produced no non-source robust label and no future-rank instability. |
