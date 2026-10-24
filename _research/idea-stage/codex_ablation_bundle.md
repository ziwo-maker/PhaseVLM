# A-Override Distillation — Ablation Planning Bundle

You are a rigorous ML reviewer. Design ablation studies for the following method and results.

## Method: A-Override Correction Distillation

A simple hand-crafted heuristic ("A-override") corrects a BC-trained traffic signal controller's systematic errors, and the corrections are distilled back into the model via targeted BC fine-tuning.

**Pipeline:**
1. Deploy BC-trained PhaseVLM (frozen Qwen3-VL-8B + trainable prefix/projector) on target network
2. At each decision step, apply A-override rule: if the max-queue phase differs from model's predicted action by queue_gap ≥ 2, override to the max-queue phase (with left-turn wait protection at 60s)
3. Record all states where override fires (model action ≠ override action)
4. Replace `expert_action` labels with the override action for those states ONLY (319/1920 = 16.6% of decisions)
5. Retrain Stage 2 BC on the corrected dataset
6. Evaluate WITHOUT A-override

**Removable/Replaceable Components:**
| Component | Description |
|-----------|-------------|
| A-override correction | The 319 label corrections (gap=2, protect=60) — core component |
| Left-turn wait protection | protect=60 prevents overriding when left-turn wait ≥ 60s |
| Gap threshold | gap=2 (current) vs gap=3, gap=5 |
| Correction ratio | 16.6% of labels corrected (319/1920) |
| Label replacement type | Hard label replacement (current) vs soft target |
| Source Stage 1 | Jinan-trained Stage 1 (current) vs Hangzhou-native Stage 1 |

## Current Results

### M1: Controlled Ablation (Jinan Stage1 → Hangzhou, seed=2024, NO override at test)
| Metric | R1 (Uncorrected BC) | R2 (A-Override Distill) | Improvement |
|--------|:---:|:---:|:---:|
| Queue | 119.3 | 79.0 | -33.8% |
| Wait | 243.5 | 108.1 | -55.6% |
| Travel | 381.0 | 332.1 | -12.8% |
| **Sum** | **743.8** | **519.2** | **-30.2%** |

### Historical (May, Hangzhou-native Stage1, single run, NO override)
| Method | Queue | Wait | Travel | Sum |
|--------|-------|------|--------|-----|
| A-override Distill | 58.9 | 39.9 | 308.2 | 407.0 |
| A-override Runtime (gap=2,p=60) | 57.2 | 42.4 | 306.8 | 406.5 |
| Best non-gap Stage 2 BC | 81.9 | 150.4 | 336.5 | 568.9 |

### Historical (Jinan 2000, 3-seed, NO override)
| Method | Queue | Wait | Travel | Sum |
|--------|-------|------|--------|-----|
| A-override Distill | 111.1 | 40.5 | 272.6 | 424.1 |
| Best non-gap Stage 2 BC | 118.1 | 57.5 | 279.6 | 455.1 |

## Claims We Want to Make

1. **A-override distillation is effective**: 319 targeted label corrections (16.6%) produce 6.8-30.2% Sum improvement
2. **Distilled model matches runtime override**: The model LEARNS the correction, not just memorizes it
3. **Only a small fraction of labels need correction**: We don't need to relabel everything
4. **The method is general**: Works across networks (Jinan, Hangzhou) and demand levels

## Your Task

Design ablation studies that:
1. Isolate contribution of each component
2. Answer questions reviewers will definitely ask
3. Test sensitivity to key hyperparameters (gap, protect, correction ratio)
4. Compare against natural alternative design choices

For each ablation specify: name, what_it_tests, expected_if_component_matters, priority (1-5).

Also provide: coverage_assessment, unnecessary_ablations, suggested_order, estimated_compute.

**Compute budget**: 3× RTX 5090, ~50 GPU-hours available. Each training run ~0.3 GPU-hours (2 GPUs × 15min), each eval ~0.15 GPU-hours.
