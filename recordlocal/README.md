# Stage I Convergence Efficiency · Jinan-1

This folder contains a self-contained, reproducible one-flow, one-seed experiment launcher.

## Commands

```bash
cd /home/code/LLMSTSC
bash recordlocal/run_stage1_efficiency_oneflow.sh --preflight-only
bash recordlocal/run_stage1_efficiency_oneflow.sh
```

The launcher creates a fresh timestamped `recordlocal/runs/stage1_convergence_jinan1_s2024_*/` directory; it refuses to overwrite an existing run. To use different GPUs, set `GPUS=0,1,2,3` with four available GPU IDs.

## Training contracts

- Jinan-1: `data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`
- Single training and evaluation seed: `2024`
- Three arms: `full`, `without_stage1`, `shuffled_alignment`
- Frozen Qwen3-VL-8B backbone; TS2I + 8 prefix tokens; hidden size 256; max length 512; bf16
- Stage I: 15 epochs, four GPUs, per GPU batch 4, gradient accumulation 1; true contrastive global batch 16
- Stage II: 15 epochs, four GPUs, per GPU batch 1, gradient accumulation 8; effective batch 32
- Same Stage II training data, learning rate schedule, loss, initialization seed and validation split across arms
- Best checkpoint selected only by **minimum validation top-1 cost regret**
- Online tests: same training flow, 3600 seconds, seed 2024; checkpoints at epochs 1, 3, 5, 10 and 15
- Shuffled control shuffles only labeled `clip_caption`, leaving image state and Stage II labels unchanged

## Files per run

- `manifest.json` and `status.tsv`: protocol and step-level status
- `derived/shuffle_manifest.json`: source/derived hashes and caption mismatch audit
- `full/stage1`, `shuffled_alignment/stage1`: alignment checkpoints and metrics
- `<arm>/stage2/metrics.jsonl` and `epoch_*.pt`: all offline validation epochs and checkpoints
- `<arm>/online/epoch_<N>/result.json`: five online checkpoint results per arm
- `online_env/`: CityFlow environment logs and snapshots saved under recordlocal
- `convergence.csv`, `online_comparison.csv`, `summary.json`, `REPORT.md`: automatically regenerated aggregates
- `logs/`: bounded, individual stage/training/online logs; launcher log is at `recordlocal/<run_id>.launcher.log`

Monitor a running job:

```bash
tail -n 12 /home/code/LLMSTSC/recordlocal/runs/<run_id>/status.tsv
nvidia-smi
```

**Reporting limits:** One seed demonstrates a pilot effect only. Do not report statistical significance, multi-seed stability, or true training-time savings unless the complete Stage I overhead is included and validated under the matched hardware protocol. The initial run ending in `202516` was intentionally interrupted after detecting an InfoNCE global batch-size mismatch. The corrected run ending in `202803` uses the actual batch size 16 for contrastive alignment.
