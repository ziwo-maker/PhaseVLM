# PhaseVLM Stage 3 — Idea Triage Bundle

You are a critical reviewer evaluating 5 candidate research ideas for a AAAI/NeurIPS submission.

## Context
PhaseVLM: VLM-based traffic signal control. Frozen Qwen3-VL-8B + learnable prefix tokens + SimExpert (counterfactual simulator). Current Stage 1 (alignment) + Stage 2 (BC). Stage 3 needs genuine innovation beyond "collect states → relabel → fine-tune."

## Candidates (with prior_work and so_what annotations)

### Idea 1: Counterfactual Preference Optimization (CPO)
Use SimExpert to generate per-action counterfactual costs, construct pairwise preference labels (action A better than B based on Pareto comparison), then fine-tune prefix tokens with DPO-style preference loss.
- **prior_work**: RLAIF for Multi-Objective TSC (2025) uses LLM preferences with DQN, not counterfactual simulation + VLM prefix. CFLight (2025) uses counterfactual causal models for safe RL, not preference optimization. DPO itself is from LLM alignment (NeurIPS 2023).
- **so_what**: Opens a new paradigm — simulator as preference oracle for policy learning. If it works, applies to any domain with a counterfactual simulator. If it fails, first rigorous negative result for preference optimization in control.
- **risk**: MEDIUM, effort: 2-3 weeks

### Idea 2: Differentiable SimExpert Surrogate
Train GNN dynamics surrogate on SimExpert rollouts → freeze → backprop through it to optimize prefix tokens via gradient descent (no RL).
- **prior_work**: Model-based RL for TSC (PRLight 2025) uses env models for policy reuse, not differentiable gradients. Differentiable agent-based simulation (2026) exists but not for VLM controllers.
- **so_what**: Transforms simulator from black-box evaluator to differentiable training infrastructure. Generalizable beyond TSC.
- **risk**: HIGH, effort: 3-4 weeks

### Idea 3: Verifier-Guided Decoding + Rank-Correlation Diagnostic
Best-of-N decoding with SimExpert as verifier + measure rank correlation between policy logits and SimExpert costs. Answers "is the VLM's knowledge under-utilized?"
- **prior_work**: Best-of-N with verifier used in code generation (AlphaCode) but never TSC. Rank-correlation diagnostic for VLM-TSC failure mode partitioning is entirely new.
- **so_what**: Either reveals free performance (cheap decoding fix) or redirects research to better representations. Disciplines all subsequent PhaseVLM development.
- **risk**: LOW, effort: 3 days

### Idea 4: Active Counterfactual Query
Select states for SimExpert relabeling using acquisition function (entropy + ensemble disagreement + OOD score) instead of random selection.
- **prior_work**: Active learning exists broadly. CuraLight (2026) uses debate-guided data curation for offline data, not active online SimExpert queries.
- **so_what**: Makes SimExpert-based adaptation 2-4× more query-efficient. Reveals which states matter for adaptation.
- **risk**: LOW-MEDIUM, effort: 2 weeks

### Idea 5: Topology-Conditioned Prefix Hypernetwork
GNN encodes road network graph → HyperNetwork generates prefix token + projector weights → zero-shot on new networks.
- **prior_work**: GNN control interface (arXiv:2607.21831) for movement scoring. HyperNetworks for multi-task PEFT (CVPR 2025/2026) but never for TSC.
- **so_what**: If it works, PhaseVLM deploys on ANY road network with zero target data. Genuine breakthrough for practical TSC.
- **risk**: HIGH, effort: 3-4 weeks

## Your Task

For each idea, play devil's advocate:
1. What's the strongest objection a reviewer would raise?
2. What's the most likely failure mode?
3. Is the prior_work note a real novelty problem, or differentiable?
4. Rank the 5 ideas for a top-venue submission
5. Which 1-2 would you actually invest in, and why?
