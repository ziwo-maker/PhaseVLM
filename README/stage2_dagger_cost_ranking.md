# Stage 2 DAgger And Cost Ranking

## Data Contract

`scripts/build_dagger_stage2_dataset.py` aggregates two data sources:

- `--base-data`: original expert demonstration JSONL files.
- `--rollout-data`: policy-closed-loop trajectories that have been relabelled by SimExpert. Each selected decision must contain `expert_action`, and should include `candidate_action_metrics` from `SimExpertCollector` (`score`, `aql`, `awt`, `att`, `spillback`).

The builder preserves every rollout observation row. It clears `expert_action` for easy decision rows and assigns `sample_weight > 1` only to high-priority relabelled decisions. This preserves the temporal history needed by `GroupedTrafficActionDataset` while focusing the Stage 2 loss on the policy-induced failures.

Priority is the sum of normalized total queue, maximum phase wait, behavior-action regret, and missed max-queue gap. The behavior action is read from `policy_action`, `base_action`, or `action`.

## Build Dataset

```bash
python scripts/build_dagger_stage2_dataset.py \
  --base-data /path/to/expert_traffic_prompt_series.jsonl \
  --rollout-data /path/to/policy_rollout_relabelled.jsonl \
  --output records/dagger/stage2_train.jsonl \
  --min-priority 1.0 \
  --queue-scale 10 --wait-scale 60 --regret-scale 1 --missed-gap-scale 5 \
  --weight-scale 0.5 --max-sample-weight 4
```

The command writes `stage2_train.jsonl.summary.json` with all source paths, thresholds, selected-row count, and label distribution. Inspect it before training; a selected-row count of zero means the thresholds or relabelling input need attention.

## Train Stage 2

The default Stage 2 objective is now:

`hard BC + cost soft-target BC + 0.15 * pairwise cost ranking + conditional 0.02 * KL(policy || base-action prior)`.

The KL term is conditional by default: it applies to rows where `base_action == expert_action`
or `sample_weight <= 1.0`. This keeps conservative imitation on ordinary demonstrations while
avoiding a pull back to the behavior action on weighted DAgger correction rows.

Candidate `score` is the preferred cost because it is the same composite score produced by SimExpert; legacy data falls back to `candidate_action_waits`. Class balancing is disabled by default. Enable it only as an explicit ablation with `--enable_class_balance`.

```bash
torchrun --standalone --nproc_per_node=2 projection_train/train_projection_bc.py \
  --data_path records/dagger/stage2_train.jsonl \
  --llm_path /path/to/qwen-vl-8B \
  --align_ckpt /path/to/stage1/best.pt \
  --output_dir records/dagger/stage2/checkpoints \
  --prompt_mode queue_wait_short \
  --batch_size 4 --grad_accum_steps 8 \
  --best_metric val_top1_cost_regret --best_mode min \
  --target_mode mixed --soft_target_alpha 0.3 --soft_label_temperature 0.35 \
  --ranking_loss_weight 0.15 --ranking_min_cost_gap 0.02 \
  --base_policy_kl_weight 0.02 --base_policy_prior_mass 0.70 \
  --base_policy_kl_mode match_or_low_weight --base_policy_kl_low_weight_threshold 1.0
```

Do not use final online test traffic to choose these loss weights or checkpoints. Select on a separately fixed online validation set, then run one untouched final test.
