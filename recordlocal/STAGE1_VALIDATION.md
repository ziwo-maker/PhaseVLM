# Stage I Effectiveness Validation

Launch the complete campaign in the background:

```bash
bash /home/code/LLMSTSC/recordlocal/run_stage1_validation.sh
```

The launcher detects idle GPUs and schedules at most one frozen 8B-model training process per GPU. Re-running the launcher resumes the campaign: completed Stage I and Stage II phases are skipped, and incomplete phases continue from their latest epoch-level `last.pt` checkpoint. If a phase failed before its first epoch checkpoint, it starts in a new numbered attempt directory. Existing logs and checkpoints are retained.

The campaign runs in tmux session `stage1_validation`; the pane process ID is recorded in `runs/stage1_validation/launcher.pid`, and the orchestrator log is `runs/stage1_validation/logs/orchestrator.log`.

## Protocol

- Dataset: `data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl` (Jinan-1).
- Split: chronological 70/15/15 train/validation/test with 30-second embargoes and 30-second sequence warmup exclusion at each split start.
- Training subsets: class-stratified nested 10%, 25%, 50%, 100% sample sets, repeated with seeds 2024, 2025, 2026.
- Experiment 1: Stage I + Stage II versus Stage II from random initialization. Best Stage II checkpoints are selected only by validation top-1 cost regret. The independent test split is evaluated after selection. Full training time includes Stage I.
- Experiment 2: normal Stage I, seeded random projector initialization, and Stage I with train image-text pairs shuffled. Retrieval is measured on the test split. A frozen image representation feeds single affine probes for congestion level, max phase queue, and max phase wait. The probe heads use the same initialization seed for all three representation arms within each experiment seed.
- Model: `/mnt/data/Qwen3-VL-8B-Instruct`, bf16, 16-step sequence, 224-pixel image, hidden dimension 256, 8 prefix tokens, max length 512.

## Outputs

All artifacts are written below `recordlocal/runs/stage1_validation/`. Each training phase stores its exact command, log, checkpoints, epoch metrics, and status summary. Aggregate tables and the analysis report are written under `reports/` and mirrored to the run root as `REPORT.md`; paper-style figures are stored under `figures/` as PDF and PNG.
