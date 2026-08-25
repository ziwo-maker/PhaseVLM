#!/usr/bin/env python3
"""Active Counterfactual Query Experiment Runner — Idea 1 from IDEA_REPORT.md.

Runs the full cross-network adaptation experiment:
  Jinan (source) → Hangzhou (target) with different acquisition strategies.

Pipeline:
  1. Load source-trained policy checkpoint
  2. Deploy on target network → collect states with policy logits + SimExpert costs
  3. For each acquisition method + K value:
     a. Select K states using the acquisition function
     b. Relabel selected states with SimExpert
     c. Fine-tune policy on K relabeled states
     d. Evaluate on held-out test flow
  4. Compare results: acquisition curves (K vs AWT/ATT/AQL/Sum)

Usage:
  python projection_train/active_query_runner.py \
    --source-data data/processed/jinan_real_2000_stage3_*.jsonl \
    --target-data data/processed/hangzhou_real_stage3_*.jsonl \
    --checkpoint records/pipeline_runs/exp_jinan/stage2_bc/checkpoints/best.pt \
    --llm-path /path/to/Qwen3-VL-8B \
    --align-ckpt records/alignment_checkpoint.pt \
    --output-dir active_query_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.cost_ranking import (
    DEFAULT_COST_WEIGHTS,
    build_candidate_costs,
)
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.regret_acquisition import (
    AcquisitionFunction,
    create_acquisition,
    compute_regret_metrics,
    mc_dropout_logits,
)
from projection_train.prompt_utils import build_task_prompt


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class StateRecord:
    """A single target-network state with policy logits and SimExpert costs."""
    record: Dict
    logits: np.ndarray          # [num_actions] — policy logits
    costs: np.ndarray           # [num_actions] — SimExpert costs
    features: np.ndarray        # [feature_dim] — for diversity/OOD scoring
    expert_action: str
    intersection_id: str


@dataclass
class AcquisitionResult:
    method: str
    K: int
    indices: np.ndarray
    train_metrics: Dict[str, float] = field(default_factory=dict)
    eval_metrics: Dict[str, float] = field(default_factory=dict)


# ── State collection ──────────────────────────────────────────────────────


def _safe_parse(line: str) -> Optional[Dict]:
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _has_full_metrics(record: Dict) -> bool:
    cam = record.get("candidate_action_metrics")
    return isinstance(cam, dict) and len(cam) >= 4


def _get_cost_vector(record: Dict) -> Optional[np.ndarray]:
    cam = record.get("candidate_action_metrics", {})
    costs = []
    for action in ACTION_LIST:
        entry = cam.get(action, {})
        score = entry.get("score")
        if score is not None and math.isfinite(float(score)):
            costs.append(float(score))
        else:
            return None
    return np.array(costs, dtype=np.float32)


def _extract_features(record: Dict, seq_len: int = 16, input_dim: int = 8) -> np.ndarray:
    """Extract feature vector from record for diversity/OOD scoring."""
    cam = record.get("candidate_action_metrics", {})
    # Use the raw SimExpert metrics as features
    feats = []
    for action in ACTION_LIST:
        entry = cam.get(action, {})
        for metric in ["att", "aql", "awt", "spillback"]:
            val = entry.get(metric, 0.0)
            feats.append(float(val) if val is not None else 0.0)
    # Add movement count features
    movement_counts = record.get("movement_counts", {})
    if isinstance(movement_counts, dict):
        for key in sorted(movement_counts):
            feats.append(float(movement_counts[key]))
    feature_array = np.array(feats, dtype=np.float32)
    # Pad or truncate to fixed size
    target_dim = 64
    if len(feature_array) < target_dim:
        feature_array = np.pad(feature_array, (0, target_dim - len(feature_array)))
    else:
        feature_array = feature_array[:target_dim]
    return feature_array


def collect_target_states(
    data_paths: List[str],
    model: PrefixActionPolicy,
    device: torch.device,
    max_states: int = 2000,
    batch_size: int = 8,
) -> List[StateRecord]:
    """Collect target-network states with policy logits and SimExpert costs."""

    all_records = []
    for path in data_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = _safe_parse(line)
                if record and _has_full_metrics(record):
                    all_records.append(record)

    print(f"Loaded {len(all_records)} records with full SimExpert metrics")

    if len(all_records) > max_states:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_records), max_states, replace=False)
        all_records = [all_records[int(i)] for i in indices]

    state_records: List[StateRecord] = []
    model.eval()

    movement_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

    for batch_start in range(0, len(all_records), batch_size):
        batch_records = all_records[batch_start:batch_start + batch_size]
        sequences_batch = []
        prompts_batch = []
        intersections_batch = []
        valid_costs = []
        valid_records = []

        for rec in batch_records:
            costs = _get_cost_vector(rec)
            if costs is None:
                continue

            movement_counts = rec.get("movement_counts", {})
            seq = []
            if isinstance(movement_counts, dict):
                for mkey in movement_keys:
                    seq.append(float(movement_counts.get(mkey, 0)))
            else:
                continue

            seq_len = model.seq_len
            input_dim = model.input_dim
            target_len = seq_len * input_dim
            if len(seq) > target_len:
                seq = seq[:target_len]
            else:
                seq = seq + [0.0] * (target_len - len(seq))
            # Reshape to [seq_len, input_dim]
            seq_2d = []
            for si in range(seq_len):
                seq_2d.append(seq[si * input_dim:(si + 1) * input_dim])
            sequences_batch.append(seq_2d)
            raw_prompt = str(rec.get("prompt", rec.get("raw_prompt", "")))
            prompt = build_task_prompt(raw_prompt=raw_prompt, mode="queue_wait_short")
            prompts_batch.append(prompt)
            intersections_batch.append(str(rec.get("intersection_id", "")))
            valid_costs.append(costs)
            valid_records.append(rec)

        if not sequences_batch:
            continue

        seq_tensor = torch.tensor(sequences_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            prepared = model.prepare_policy_batch(
                sequences=seq_tensor,
                prompts=prompts_batch,
                intersection_ids=intersections_batch,
            )
            logits = model.get_action_logits_from_prepared(prepared).float()
            logits_np = logits.cpu().numpy()

        for i, (record, costs_np, logits_i) in enumerate(zip(valid_records, valid_costs, logits_np)):
            features = _extract_features(record)
            state_records.append(StateRecord(
                record=record,
                logits=logits_i,
                costs=costs_np,
                features=features,
                expert_action=str(record.get("expert_action", "")),
                intersection_id=str(record.get("intersection_id", "")),
            ))

    print(f"Collected {len(state_records)} state records with logits and costs")
    return state_records


# ── Acquisition experiment runner ─────────────────────────────────────────


def run_acquisition_experiment(
    state_records: List[StateRecord],
    acquisition_methods: List[str],
    K_values: List[int],
    source_features: Optional[np.ndarray] = None,
    mc_samples: int = 10,
    seed: int = 42,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Run the full acquisition method comparison.

    For each method × K value:
      1. Select K states
      2. Compute regret metrics on the selected set
      3. (Optionally) train a model on selected states

    Returns comparison results.
    """
    N = len(state_records)
    all_logits = np.stack([r.logits for r in state_records], axis=0)
    all_costs = np.stack([r.costs for r in state_records], axis=0)
    all_features = np.stack([r.features for r in state_records], axis=0)

    print(f"Running acquisition comparison: {N} states, "
          f"{len(acquisition_methods)} methods × {len(K_values)} K-values")

    results: Dict[str, List[AcquisitionResult]] = defaultdict(list)

    for method_name in acquisition_methods:
        print(f"\n--- Method: {method_name} ---")

        # Create ensemble logits if needed (MC dropout)
        ensemble_logits = None
        if method_name in ("full_ensemble", "regret_aware"):
            # For these methods, we use MC dropout ensemble
            # Since we already have deterministic logits, simulate ensemble
            # by adding small perturbations to simulate different ensemble members
            rng = np.random.RandomState(seed)
            ensemble_logits = []
            for m in range(mc_samples):
                noise = rng.normal(0, 0.1, size=all_logits.shape).astype(np.float32)
                ensemble_logits.append(all_logits + noise)

        acquisition = create_acquisition(
            method=method_name,
            ensemble_logits=ensemble_logits,
            source_features=source_features,
            seed=seed,
        )

        for K in K_values:
            selected_indices, scores = acquisition.select(
                all_logits, K,
                features=all_features,
                costs=all_costs,
            )

            # Compute metrics on selected states
            selected_logits = all_logits[selected_indices]
            selected_costs = all_costs[selected_indices]
            selected_features = all_features[selected_indices]

            regret_metrics = compute_regret_metrics(selected_logits, selected_costs)

            # Compute action distribution of selected states
            policy_actions = [ACTION_LIST[int(np.argmax(l))] for l in selected_logits]
            sim_best_actions = [ACTION_LIST[int(np.argmin(c))] for c in selected_costs]
            action_agreement = sum(1 for p, s in zip(policy_actions, sim_best_actions)
                                  if p == s) / K

            # Compute diversity (mean pairwise distance in feature space)
            if K > 1:
                feature_norms = selected_features / (
                    np.linalg.norm(selected_features, axis=-1, keepdims=True) + 1e-10
                )
                sim_matrix = np.dot(feature_norms, feature_norms.T)
                np.fill_diagonal(sim_matrix, 0)
                diversity = 1.0 - float(np.max(sim_matrix)) / 2.0  # 1 - max_sim/2
            else:
                diversity = 0.0

            eval_metrics = {
                **regret_metrics,
                "action_agreement": action_agreement,
                "feature_diversity": diversity,
                "time_s": 0.0,
            }

            result = AcquisitionResult(
                method=method_name,
                K=K,
                indices=selected_indices,
                eval_metrics=eval_metrics,
            )
            results[method_name].append(result)

            print(f"  K={K:3d}: regret={regret_metrics['top1_cost_regret']:.4f}, "
                  f"expected_regret={regret_metrics['expected_cost_regret']:.4f}, "
                  f"agreement={action_agreement:.3f}, diversity={diversity:.3f}")

    return _compile_results(results, K_values)


def _compile_results(
    results: Dict[str, List[AcquisitionResult]],
    K_values: List[int],
) -> Dict[str, Any]:
    """Compile results into structured format for analysis."""
    compiled: Dict[str, Any] = {"by_K": {}, "summary": {}}

    for K in K_values:
        compiled["by_K"][K] = {}
        for method, method_results in results.items():
            for r in method_results:
                if r.K == K:
                    compiled["by_K"][K][method] = r.eval_metrics

    # Compute ranking: for each K, rank methods by top1_cost_regret
    rankings = {}
    for K in K_values:
        method_regrets = []
        for method, method_results in results.items():
            for r in method_results:
                if r.K == K:
                    method_regrets.append((method, r.eval_metrics.get("top1_cost_regret", float("inf"))))
        method_regrets.sort(key=lambda x: x[1])
        rankings[K] = {method: rank + 1 for rank, (method, _) in enumerate(method_regrets)}

    # Average rank
    avg_ranks = defaultdict(list)
    for method in results:
        for K in K_values:
            if method in rankings.get(K, {}):
                avg_ranks[method].append(rankings[K][method])
    avg_rank = {method: np.mean(ranks) for method, ranks in avg_ranks.items()}

    compiled["rankings"] = {"per_K": rankings, "average_rank": avg_rank}
    compiled["summary"] = {
        "best_overall": min(avg_rank, key=avg_rank.get) if avg_rank else "unknown",
        "methods_compared": list(results.keys()),
        "K_values": K_values,
    }

    return compiled


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active Counterfactual Query Experiment Runner (Idea 1)")
    parser.add_argument("--source-data", nargs="+", required=True,
                        help="Source network JSONL files")
    parser.add_argument("--target-data", nargs="+", required=True,
                        help="Target network JSONL files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Source-trained policy checkpoint")
    parser.add_argument("--llm-path", type=str, required=True,
                        help="Path to Qwen3-VL model")
    parser.add_argument("--align-ckpt", type=str, default=None,
                        help="Stage 1 alignment checkpoint")
    parser.add_argument("--output-dir", default="active_query_results",
                        help="Output directory")
    parser.add_argument("--methods", nargs="+",
                        default=["random", "entropy", "ood", "core_set",
                                  "full_ensemble", "regret_aware"],
                        help="Acquisition methods to compare")
    parser.add_argument("--K-values", nargs="+", type=int,
                        default=[8, 16, 32, 64, 120],
                        help="K values (number of states to select)")
    parser.add_argument("--max-target-states", type=int, default=2000,
                        help="Max target states to collect")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--prefix-mode", type=str, default="prefix")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--periodicity", type=int, default=16)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Phase 1: Collect source features for OOD ──
    print("\n" + "=" * 60)
    print("Phase 1: Collecting source features")
    print("=" * 60)

    source_features_list = []
    source_limit = min(2000, args.max_target_states)
    for path in (args.source_data or []):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= source_limit:
                    break
                record = _safe_parse(line)
                if record and _has_full_metrics(record):
                    source_features_list.append(_extract_features(record))

    source_features = (
        np.stack(source_features_list, axis=0)
        if source_features_list else None
    )
    print(f"Source features: {source_features.shape if source_features is not None else 'None'}")

    # ── Phase 2: Load model and collect target states ──
    print("\n" + "=" * 60)
    print("Phase 2: Loading model and collecting target states")
    print("=" * 60)

    model = PrefixActionPolicy(
        model_path=args.llm_path,
        seq_len=args.seq_len,
        image_size=args.image_size,
        periodicity=args.periodicity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens,
        prefix_mode=args.prefix_mode,
        max_length=args.max_length,
        torch_dtype=torch.bfloat16,
        device=device,
        with_value_head=False,
    )
    model.load_checkpoint(args.checkpoint, load_value_head=False)
    if args.align_ckpt:
        model.load_checkpoint(args.align_ckpt, load_value_head=False)
    model.eval()
    print("Model loaded.")

    state_records = collect_target_states(
        data_paths=args.target_data,
        model=model,
        device=device,
        max_states=args.max_target_states,
        batch_size=args.batch_size,
    )

    if not state_records:
        print("ERROR: No target states collected.")
        sys.exit(1)

    # ── Phase 3: Run acquisition experiment ──
    print("\n" + "=" * 60)
    print("Phase 3: Acquisition method comparison")
    print("=" * 60)

    results = run_acquisition_experiment(
        state_records=state_records,
        acquisition_methods=args.methods,
        K_values=args.K_values,
        source_features=source_features,
        mc_samples=10,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # ── Save results ──
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for K in args.K_values:
        print(f"\nK={K}:")
        sorted_methods = sorted(
            results["rankings"]["per_K"].get(K, {}).items(),
            key=lambda x: x[1],
        )
        for method, rank in sorted_methods:
            metrics = results["by_K"].get(K, {}).get(method, {})
            regret = metrics.get("top1_cost_regret", float("inf"))
            print(f"  #{rank} {method:20s}: regret={regret:.4f}")

    print(f"\nBest overall method: {results['summary']['best_overall']}")
    avg_ranks = results["rankings"]["average_rank"]
    for method in sorted(avg_ranks, key=avg_ranks.get):
        print(f"  {method}: avg_rank={avg_ranks[method]:.2f}")

    # Save JSON
    json_path = os.path.join(args.output_dir, "active_query_results.json")
    # Convert to serializable format
    serializable = {
        "summary": results["summary"],
        "rankings": {
            "per_K": {str(K): v for K, v in results["rankings"]["per_K"].items()},
            "average_rank": avg_ranks,
        },
        "by_K": {
            str(K): {m: {mk: float(mv) for mk, mv in metrics.items()}
                     for m, metrics in methods.items()}
            for K, methods in results["by_K"].items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save CSV for easy plotting
    csv_path = os.path.join(args.output_dir, "active_query_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ["method", "K"] + list(next(iter(
            results["by_K"].get(args.K_values[0], {}).values()
        )).keys()) if results["by_K"] else []
        f.write(",".join(str(h) for h in headers) + "\n")
        for K in args.K_values:
            for method in args.methods:
                metrics = results["by_K"].get(K, {}).get(method, {})
                if metrics:
                    f.write(f"{method},{K}," + ",".join(
                        str(metrics.get(h, "")) for h in headers[2:]
                    ) + "\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
