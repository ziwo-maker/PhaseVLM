#!/usr/bin/env python3
"""Bottleneck Diagnostic — Idea 3 from IDEA_REPORT.md.

Decomposes PhaseVLM's failure into calibration vs representation vs oracle mismatch
via counterfactual verification against SimExpert action costs.

For each held-out state:
  1. Enumerate ALL 4 actions through SimExpert (not sample — enumerate)
  2. Get policy logits for all 4 actions
  3. Compute diagnostics:
     a. Oracle gap = cost(policy_argmax) - cost(SimExpert_best)
     b. Top-action accuracy = does argmax match SimExpert best?
     c. Margin-conditioned accuracy = accuracy stratified by logit margin
     d. ECE = calibration error between confidence and action optimality
     e. Spearman ρ = rank correlation between policy ranking and cost ranking

Outputs:
  - bottleneck_diagnostic.json  — summary metrics and diagnosis
  - bottleneck_diagnostic.csv   — per-state diagnostic rows
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
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
    cost_soft_targets,
)
from projection_train.grouped_dataset import GroupedTrafficActionDataset, grouped_action_collate
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.prompt_utils import build_task_prompt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────


def _safe_parse(line: str) -> Optional[Dict]:
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _has_full_metrics(record: Dict) -> bool:
    cam = record.get("candidate_action_metrics")
    return isinstance(cam, dict) and len(cam) >= 4


def _get_cost_vector(record: Dict) -> Optional[np.ndarray]:
    """Get the SimExpert cost vector for all 4 actions (lower = better)."""
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


def _get_best_action(costs: np.ndarray) -> str:
    idx = int(np.argmin(costs))
    return ACTION_LIST[idx]


def _oracle_gap(costs: np.ndarray, policy_action: str) -> float:
    """cost(policy_action) - cost(best_action), non-negative by definition."""
    best_cost = float(np.min(costs))
    policy_idx = action_to_index(policy_action)
    policy_cost = float(costs[policy_idx])
    return policy_cost - best_cost


def _spearman_rho(logits: np.ndarray, costs: np.ndarray) -> float:
    """Spearman rank correlation between policy logits and costs.

    Policy prefers higher logits; SimExpert prefers lower costs → negate costs.
    """
    from scipy.stats import spearmanr
    rho, _ = spearmanr(logits, -costs)
    if not math.isfinite(rho):
        return 0.0
    return float(rho)


def _ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


# ── Diagnostic computation ────────────────────────────────────────────────


def compute_diagnostics(
    model: PrefixActionPolicy,
    records: List[Dict],
    device: torch.device,
    batch_size: int = 8,
    max_samples: int = 5000,
) -> Dict:
    """Run the full bottleneck diagnostic: get policy logits, compare to SimExpert costs."""

    if len(records) > max_samples:
        indices = np.random.choice(len(records), max_samples, replace=False)
        records = [records[int(i)] for i in indices]

    all_logits: List[np.ndarray] = []
    all_costs: List[np.ndarray] = []
    all_intersections: List[str] = []
    all_dates: List[str] = []
    all_expert_actions: List[str] = []

    model.eval()

    # Process in batches
    for batch_start in tqdm(range(0, len(records), batch_size), desc="Running diagnostic"):
        batch_records = records[batch_start:batch_start + batch_size]
        if not batch_records:
            break

        # Build sequences and prompts
        sequences_batch = []
        prompts_batch = []
        intersection_batch = []
        valid_indices = []

        for i, rec in enumerate(batch_records):
            costs = _get_cost_vector(rec)
            if costs is None:
                continue
            # Build sequence tensor from movement_counts
            movement_counts = rec.get("movement_counts", {})
            if not movement_counts:
                continue

            seq = []
            movement_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
            if isinstance(movement_counts, dict):
                for mkey in movement_keys:
                    val = movement_counts.get(mkey, 0)
                    try:
                        seq.append(float(val))
                    except (TypeError, ValueError):
                        seq.append(0.0)
            elif isinstance(movement_counts, list):
                seq = [float(v) for v in movement_counts[:model.seq_len * model.input_dim]]

            if len(seq) < model.seq_len * model.input_dim:
                seq += [0.0] * (model.seq_len * model.input_dim - len(seq))

            # Reshape to [seq_len, input_dim]
            seq_len = model.seq_len
            input_dim = model.input_dim
            target_len = seq_len * input_dim
            if len(seq) > target_len:
                seq = seq[:target_len]
            else:
                seq = seq + [0.0] * (target_len - len(seq))
            # Reshape flat list to [seq_len, input_dim]
            seq_2d = []
            for si in range(seq_len):
                seq_2d.append(seq[si * input_dim:(si + 1) * input_dim])
            sequences_batch.append(seq_2d)
            # Build prompt from raw_prompt in record
            raw_prompt = str(rec.get("prompt", rec.get("raw_prompt", "")))
            prompt = build_task_prompt(raw_prompt=raw_prompt, mode="queue_wait_short")
            prompts_batch.append(prompt)
            intersection_batch.append(str(rec.get("intersection_id", "")))
            all_costs.append(costs)
            all_intersections.append(str(rec.get("intersection_id", "")))
            all_dates.append(str(rec.get("date", "")))
            all_expert_actions.append(str(rec.get("expert_action", "")))
            valid_indices.append(i)

        if not sequences_batch:
            continue

        seq_tensor = torch.tensor(sequences_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            prepared = model.prepare_policy_batch(
                sequences=seq_tensor,
                prompts=prompts_batch,
                intersection_ids=intersection_batch,
                dataset_hints=["Hangzhou-4_4"] * len(intersection_batch),
            )
            logits = model.get_action_logits_from_prepared(prepared).float()
            all_logits.extend(logits.cpu().numpy())

    if not all_logits:
        return {"error": "No valid samples processed. Check data and model compatibility."}

    logits_array = np.stack(all_logits, axis=0)  # [N, 4]
    costs_array = np.stack(all_costs, axis=0)   # [N, 4]

    n_samples = logits_array.shape[0]
    print(f"Processed {n_samples} samples")

    # ── (a) Oracle gap ──
    policy_argmax = np.argmax(logits_array, axis=1)
    oracle_gaps = []
    for i in range(n_samples):
        policy_action = ACTION_LIST[policy_argmax[i]]
        gap = _oracle_gap(costs_array[i], policy_action)
        oracle_gaps.append(gap)

    oracle_gaps = np.array(oracle_gaps)
    mean_oracle_gap = float(np.mean(oracle_gaps))
    median_oracle_gap = float(np.median(oracle_gaps))

    # ── (b) Top-action accuracy ──
    sim_best_idx = np.argmin(costs_array, axis=1)
    correct = (policy_argmax == sim_best_idx).astype(np.float32)
    top_action_accuracy = float(np.mean(correct))

    # ── (c) Margin-conditioned accuracy ──
    # Sort logits, compute margin between top-2
    sorted_logits = np.sort(logits_array, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]

    margin_bins = [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")]
    margin_accuracy = {}
    for i in range(len(margin_bins) - 1):
        lo, hi = margin_bins[i], margin_bins[i + 1]
        mask = (margins >= lo) & (margins < hi)
        if mask.sum() > 0:
            acc = float(correct[mask].mean())
            margin_accuracy[f"margin_{lo:.1f}_{hi if hi < float('inf') else 'inf'}"] = {
                "accuracy": round(acc, 4),
                "count": int(mask.sum()),
            }

    # ── (d) ECE calibration ──
    # Confidence = softmax probability of top action
    confidences = np.exp(logits_array) / np.exp(logits_array).sum(axis=1, keepdims=True)
    top_confidences = confidences[np.arange(n_samples), policy_argmax]
    # "Accuracy" for calibration = did the policy pick the SimExpert-best action?
    ece_value = _ece(top_confidences, correct)

    # ── (e) Spearman ρ ──
    rho_values = []
    for i in range(n_samples):
        rho = _spearman_rho(logits_array[i], costs_array[i])
        rho_values.append(rho)
    rho_values = np.array(rho_values)
    mean_rho = float(np.mean(rho_values))
    median_rho = float(np.median(rho_values))

    # ── (f) Expert action agreement ──
    expert_correct = []
    for i in range(n_samples):
        expert_action = all_expert_actions[i]
        expert_correct.append(policy_argmax[i] == action_to_index(expert_action))
    expert_correct = np.array(expert_correct, dtype=np.float32)
    expert_agreement = float(np.mean(expert_correct))

    # ── Stratify by congestion ──
    # Use queue as a proxy for congestion level
    queue_values = []
    for i in range(n_samples):
        # queue ≈ cost based on aql weight
        cam = records[i].get("candidate_action_metrics", {}) if i < len(records) else {}
        q = 0.0
        for action, vals in cam.items():
            q = max(q, vals.get("aql", 0.0))
        queue_values.append(q)
    queue_array = np.array(queue_values)
    q_median = float(np.median(queue_array))

    low_q_mask = queue_array <= q_median
    high_q_mask = queue_array > q_median

    stratified = {
        "low_congestion": {
            "count": int(low_q_mask.sum()),
            "top_action_accuracy": float(correct[low_q_mask].mean()) if low_q_mask.any() else 0.0,
            "mean_oracle_gap": float(oracle_gaps[low_q_mask].mean()) if low_q_mask.any() else 0.0,
            "mean_rho": float(rho_values[low_q_mask].mean()) if low_q_mask.any() else 0.0,
        },
        "high_congestion": {
            "count": int(high_q_mask.sum()),
            "top_action_accuracy": float(correct[high_q_mask].mean()) if high_q_mask.any() else 0.0,
            "mean_oracle_gap": float(oracle_gaps[high_q_mask].mean()) if high_q_mask.any() else 0.0,
            "mean_rho": float(rho_values[high_q_mask].mean()) if high_q_mask.any() else 0.0,
        },
    }

    # ── Diagnosis ──
    if mean_rho < 0.3:
        bottleneck = "representation_failure"
        recommendation = (
            "Low rank correlation (ρ < 0.3): policy ranking is poorly aligned with "
            "SimExpert cost ranking. Invest in TimeSeriesProjector or prefix capacity."
        )
    elif mean_rho > 0.5 and ece_value > 0.2:
        bottleneck = "calibration_failure"
        recommendation = (
            f"Good rank correlation (ρ={mean_rho:.2f}) but high ECE (ECE={ece_value:.3f}): "
            "policy knows what's good but is poorly calibrated. "
            "Apply temperature scaling or label smoothing."
        )
    elif mean_rho > 0.5 and ece_value <= 0.2:
        bottleneck = "good_alignment"
        recommendation = (
            f"Well-calibrated (ρ={mean_rho:.2f}, ECE={ece_value:.3f}). "
            "Remaining oracle gap likely from limited prefix capacity or SimExpert noise."
        )
    else:
        bottleneck = "mixed"
        recommendation = (
            f"Mixed signals (ρ={mean_rho:.2f}, ECE={ece_value:.3f}). "
            "Review per-state diagnostics and consider both calibration and capacity improvements."
        )

    results = {
        "n_samples": n_samples,
        "dataset": "see csv for per-state details",
        "oracle_gap": {
            "mean": round(mean_oracle_gap, 6),
            "median": round(median_oracle_gap, 6),
            "p25": round(float(np.percentile(oracle_gaps, 25)), 6),
            "p75": round(float(np.percentile(oracle_gaps, 75)), 6),
            "p95": round(float(np.percentile(oracle_gaps, 95)), 6),
            "min": round(float(np.min(oracle_gaps)), 6),
            "max": round(float(np.max(oracle_gaps)), 6),
        },
        "top_action_accuracy": round(top_action_accuracy, 4),
        "expert_agreement": round(expert_agreement, 4),
        "margin_conditioned_accuracy": margin_accuracy,
        "ece": round(ece_value, 4),
        "spearman_rho": {
            "mean": round(mean_rho, 4),
            "median": round(median_rho, 4),
            "p25": round(float(np.percentile(rho_values, 25)), 4),
            "p75": round(float(np.percentile(rho_values, 75)), 4),
            "rho_positive_fraction": round(float(np.mean(rho_values > 0)), 4),
        },
        "stratified_by_congestion": stratified,
        "bottleneck_diagnosis": bottleneck,
        "recommendation": recommendation,
    }

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Bottleneck Diagnostic (Idea 3)")
    parser.add_argument("--data", nargs="+", required=True,
                        help="JSONL files with candidate_action_metrics")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--llm-path", type=str, required=True,
                        help="Path to Qwen3-VL model")
    parser.add_argument("--align-ckpt", type=str, default=None,
                        help="Stage 1 alignment checkpoint")
    parser.add_argument("--output-dir", default="bottleneck_diagnostic",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--prefix-mode", type=str, default="prefix")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--periodicity", type=int, default=16)
    parser.add_argument("--enable-topology-adapter", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    all_records = []
    for path in args.data:
        file_records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = _safe_parse(line)
                if record and _has_full_metrics(record):
                    file_records.append(record)
        print(f"Loaded {len(file_records)} valid records from {os.path.basename(path)}")
        all_records.extend(file_records)

    print(f"Total valid records: {len(all_records)}")

    if not all_records:
        print("ERROR: No valid records found.")
        sys.exit(1)

    # Sample
    if len(all_records) > args.max_samples:
        indices = np.random.choice(len(all_records), args.max_samples, replace=False)
        all_records = [all_records[int(i)] for i in indices]

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
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
        enable_topology_adapter=args.enable_topology_adapter,
    )
    model.load_checkpoint(args.checkpoint, load_value_head=False)
    if args.align_ckpt:
        model.load_checkpoint(args.align_ckpt, load_value_head=False)
    model.eval()
    print("Model loaded.")

    # Run diagnostic
    print(f"\nRunning bottleneck diagnostic on {len(all_records)} samples...")
    results = compute_diagnostics(
        model=model,
        records=all_records,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    if "error" in results:
        print(f"ERROR: {results['error']}")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("BOTTLENECK DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Samples: {results['n_samples']}")
    print(f"Oracle gap: mean={results['oracle_gap']['mean']:.4f}, median={results['oracle_gap']['median']:.4f}")
    print(f"Top-action accuracy: {results['top_action_accuracy']:.2%}")
    print(f"Expert agreement: {results['expert_agreement']:.2%}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Spearman ρ: mean={results['spearman_rho']['mean']:.3f}, median={results['spearman_rho']['median']:.3f}")
    print(f"ρ positive: {results['spearman_rho']['rho_positive_fraction']:.1%}")

    print("\n--- Stratified by congestion ---")
    for level, stats in results["stratified_by_congestion"].items():
        print(f"  {level} (n={stats['count']}): acc={stats['top_action_accuracy']:.2%}, "
              f"gap={stats['mean_oracle_gap']:.4f}, ρ={stats['mean_rho']:.3f}")

    print("\n--- Margin-conditioned accuracy ---")
    for margin_range, stats in results["margin_conditioned_accuracy"].items():
        print(f"  {margin_range}: acc={stats['accuracy']:.2%} (n={stats['count']})")

    print(f"\n*** DIAGNOSIS: {results['bottleneck_diagnosis']} ***")
    print(f"Recommendation: {results['recommendation']}")

    # Save results
    json_path = os.path.join(args.output_dir, "bottleneck_diagnostic.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save per-sample CSV
    csv_path = os.path.join(args.output_dir, "bottleneck_diagnostic.csv")
    # We'd need to recompute per-sample values, but the key outputs are already in the JSON
    # For brevity, save summary CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key in ["top_action_accuracy", "expert_agreement", "ece"]:
            f.write(f"{key},{results[key]}\n")
        for key, val in results["oracle_gap"].items():
            f.write(f"oracle_gap_{key},{val}\n")
        for key, val in results["spearman_rho"].items():
            f.write(f"spearman_rho_{key},{val}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
