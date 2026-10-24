# PhaseVLM Stage 3 — Research Idea Generation Bundle

You are a senior ML researcher reviewing a project and brainstorming innovative Stage 3 research directions. Be creative but grounded. Prioritize ideas where the answer matters regardless of direction.

## Research Direction

Design a **Stage 3** for PhaseVLM that substantially improves over the current "cross-network few-shot adaptation with SimExpert relabeling" approach. The goal is to find a genuinely innovative contribution that would strengthen a AAAI/NeurIPS submission.

## Project Background: PhaseVLM

PhaseVLM is a **VLM-based traffic signal control (TSC)** system:

**Architecture:**
- Frozen **Qwen3-VL-8B** backbone
- **TimeSeriesProjector**: Converts 16-step detector history (8-dim per step: queue lengths, wait times per lane) into a TS2I (Time-Series-to-Image) traffic state image (224×224×3)
- **VisionPrefixProjector**: Maps VLM vision encoder pooled features → 8-16 learnable prefix tokens injected into the LLM
- **PrefixActionPolicy**: Outputs logits over 4 traffic signal phases (ETWT, NTST, ELWL, NLSL)
- New: **IntersectionTopologyEncoder** (MLP): Maps static intersection features (grid position, neighbor existence, network scale) to topology embedding for cross-network conditioning

**Training Pipeline:**
- **Stage 1**: Vision-Language Alignment — contrastive learning between TS2I images and text captions describing traffic states
- **Stage 2**: Action Adaptation — supervised BC using SimExpert (snapshot-based counterfactual simulator) labels, with optional cost ranking + soft target distillation
- **Stage 3 (current design)**: Cross-network few-shot adaptation — collect states on target network via canonical evaluator → relabel with SimExpert/RuleOnly → fine-tune BC on relabeled data

**Key Empirical Findings:**
1. Data quality (SimExpert labels) decisively outweighs loss function design — pure BC on high-quality data beats BC+rank+KL
2. Queue-Wait trade-off: cost ranking with balanced weights (awt=2.0, aql=1.0) improves Queue but degrades Wait
3. val_acc ≠ online performance — lower val_acc checkpoints can achieve better online metrics
4. Cross-network transfer (Jinan 3×4 ↔ Hangzhou 4×4) is possible but limited by topology mismatch
5. SimExpert is the gold-standard labeler but is expensive (snapshot-based counterfactual simulation per action)

**Current Stage 3 Limitations (reviewer score: 5/10, conceptual 7/10):**
- Random state selection for SimExpert querying (no active learning)
- SimExpert used only for hard label generation (ignores rich counterfactual information)
- RuleOnly fallback (queue + α*wait) is a weak heuristic
- Topology adapter is a simple MLP, no graph structure
- No adaptation during deployment (offline only)
- K budget infeasible: one episode has ≤120 snapshots, K=128 and K=512 silently cap
- Queue-Wait trade-off handled via fixed scalar weights

## Current Landscape Map

### 1. VLM/LLM-Based Traffic Signal Control (2025-2026)
- **VLMLight** (NeurIPS 2025): Dual-branch VLM (fast RL + slow LLM reasoning), multi-view visual inputs, emergency vehicle prioritization. Uses Qwen2.5-VL-32B. NOT prefix-tuning based.
- **Traffic-R1** (2025): 3B LLM with RL self-exploration + expert guidance, reasoning-based action selection
- **CuraLight** (IJCNN 2026): Multi-LLM debate-guided data curation + diffusion RL exploration, LoRA fine-tuning
- **DGLight** (2026): DQN-guided GRPO fine-tuning of LLMs for TSC
- **TSC-LLM** (Transportation Research Part C, 2026): Label-free PPO-based RLFT with preference-weight sampling for multi-objective reward
- **ReasonLight** (2025): Multimodal foundation model for zero-shot TSC

**Key gap**: NONE of these use prefix tuning on a frozen VLM. PhaseVLM's architecture (frozen VLM + learnable prefix + TS2I projector) is unique. No one has combined VLM prefix tuning with counterfactual preference optimization.

### 2. Cross-Network Generalization & Adaptation
- **TransferLight** (2024/2025): Zero-shot TSC on any road network
- **GNN-based control interface** (arXiv:2607.21831, Jul 2026): Movement-level GNN transfers across unseen grid geometries, but sensitive to coverage distribution shifts
- **MetaLight** (AAAI 2020): Value-based meta-RL, adapts in 1-5 steps but fails under large distribution shifts (>22% error)
- **PRLight** (2025): Model-based RL with policy reuse via source-target similarity
- **GESA** (IEEE T-ITS, 2024): Scenario-agnostic co-training across 7 cities, 13.27% higher reward

**Key gap**: No one conditions VLM prefix tokens on road network topology. Meta-learning for TSC exists (MetaLight) but not in combination with VLM architectures.

### 3. Counterfactual Methods in TSC
- **CFLight** (2025): Counterfactual causal model for safe RL, validates on real-world data (Cologne)
- **UCATSC** (2026): Uncertainty-aware counterfactual MPC with hard safety constraints, belief-space rollouts
- **SimExpert** (PhaseVLM's own): Snapshot-based counterfactual simulation — saves/restores CityFlow state, simulates each action for K seconds, collects metrics

**Key gap**: Counterfactual simulators are used only for evaluation or hard labeling. No one uses counterfactual simulation to construct **preference pairs** for preference optimization (DPO-style). No one builds a **differentiable surrogate** from counterfactual rollouts.

### 4. Preference Optimization (DPO/RLHF) for TSC
- **RLAIF for Multi-Objective TSC** (arXiv:2602.20728): First to use LLMs as preference annotators in TSC, DQN is the underlying policy
- **Multi-PrefDrive** (IEEE, 2025): Plackett-Luce preference model for action selection (autonomous driving, not TSC)
- **DPO** (NeurIPS 2023): Closed-form optimal policy from preference data via classification loss

**Key gap**: Direct Preference Optimization (DPO) has NOT been applied to TSC. The combination of counterfactual simulation + DPO-style preference learning for TSC is completely unexplored.

### 5. Test-Time & Online Adaptation
- **AdaTent++** (2026): Reliability-gated test-time adaptation for traffic sign detection
- **Online adaptive TP-RL-TSC** (ML4CPS 2026): Online clustering + cluster-specific RL agents (conceptual, no empirical results yet)
- **Distribution shift in TSC**: 0.1 KL divergence increase → 58% delay increase; GEH distance predicts performance degradation

**Key gap**: Test-time adaptation for traffic signal control policies (not perception) is essentially unexplored.

### 6. Parameter-Efficient VLM Adaptation
- **Dynamic Prefix Weighting** (CVPR 2026): Token-level prefix weighting for continual VLM learning
- **Skip Tuning** (CVPR 2025): Layer/class skipping outperforms prompt tuning without extra parameters
- **Hypernetwork-based PEFT**: Single hypernetwork generates prefix + adapter weights across modalities, <1% new params

**Key gap**: Hypernetwork-generated prefix tokens conditioned on external factors (like road topology) is a natural extension not yet explored for TSC.

## Key Structural Gaps (Opportunities)

### Gap 1: Counterfactual Preference Optimization
**Angle**: `method-transfer` — DPO/RLHF has transformed LLM alignment but is untried in TSC
PhaseVLM uniquely has SimExpert (counterfactual simulator) + prefix-tuning architecture. This is the perfect setup for DPO-style preference optimization: SimExpert provides pairwise preferences between candidate actions, and prefix tokens are optimized via a preference loss.

### Gap 2: Differentiable Surrogate from Counterfactual Rollouts
**Angle**: `scaling-regime` — gradient-based policy optimization via surrogate models
SimExpert generates rich counterfactual trajectories. Train a lightweight GNN surrogate on this data → differentiable → backprop through it to optimize prefix tokens. Avoids RL instability entirely.

### Gap 3: Active Counterfactual Query
**Angle**: `diagnostic` — which states are worth simulating?
SimExpert queries are expensive (~seconds per action). Current approach selects states randomly. Use ensemble disagreement, predictive entropy, or OOD detection to select the most informative states for counterfactual simulation.

### Gap 4: HyperNetwork-Generated Topology-Conditioned Prefix Tokens
**Angle**: `method-transfer` — HyperNetworks (successful in multi-task learning) + GNN topology encoding → zero-shot generalization
Instead of fixed learnable prefix tokens, generate them via a HyperNetwork conditioned on GNN-encoded road network topology. Enables zero-shot transfer to arbitrary grid sizes.

### Gap 5: Test-Time Simulator-Guided Self-Adaptation
**Angle**: `untested-assumption` — policies must be frozen at deployment
Deploy the policy, use lightweight simulator queries at test time to verify/improve actions, update prefix tokens online. Bridges the offline-online gap in real time.

### Gap 6: Pareto-Constrained Preference Learning
**Angle**: `contradiction` — Queue-Wait trade-off is a constraint problem, not a weighting problem
Current approaches use fixed scalar weights. Instead: learn a policy that maximizes ATT/AQL subject to AWT/spillback constraints, using Lagrangian duality or CPO-style constrained optimization on counterfactual preferences.

## Your Task

Generate **8-12 concrete research ideas** for Stage 3. For each idea, provide:
1. One-sentence summary
2. Core hypothesis (what you expect to find and why)
3. Minimum viable experiment (cheapest way to test)
4. Expected contribution type: empirical finding / new method / theoretical result / diagnostic
5. Risk level: LOW (likely works) / MEDIUM (50-50) / HIGH (speculative)
6. Estimated effort: days / weeks / months

Prioritize ideas that are:
- Testable with moderate compute (2× RTX 5090 or less)
- Likely to produce a clear positive OR negative result (both publishable)
- Not "apply X to Y" unless the application reveals genuinely surprising insights
- Differentiated from all papers mentioned above
- Leveraging PhaseVLM's unique architecture (frozen VLM + prefix tokens + SimExpert)
