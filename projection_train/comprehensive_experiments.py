#!/usr/bin/env python3
"""Comprehensive Stage 3 Experiment Suite.

Runs all experiments from IDEA_REPORT.md at scale:

Phase 1: Bottleneck Diagnostic — 4 configurations × 2000 samples
  - Jinan model on Jinan data (in-network)
  - Jinan model on Hangzhou data (cross-network, real scenario)
  - Hangzhou model on Hangzhou data (in-network)
  - Hangzhou model on Jinan data (cross-network)

Phase 2: Full 6-Method Active Query — 6 methods × 5 K-values × 2000 states
  - random, entropy, ood, core_set, full_ensemble, regret_aware
  - K ∈ {8, 16, 32, 64, 120}
  - Multi-seed (42, 2024, 2025)

Phase 3: Per-intersection & congestion-stratified analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.prompt_utils import build_task_prompt, PROMPT_MODES
from projection_train.regret_acquisition import (
    AcquisitionFunction,
    create_acquisition,
    compute_regret_metrics,
    mc_dropout_logits,
)
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────

MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]


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


def _extract_sequence(record: Dict, seq_len: int, input_dim: int) -> Optional[List[List[float]]]:
    """Build [seq_len, input_dim] sequence from movement_counts."""
    mc = record.get("movement_counts", {})
    if not isinstance(mc, dict) or not mc:
        return None
    seq = []
    for mkey in MOVEMENT_KEYS:
        try:
            seq.append(float(mc.get(mkey, 0)))
        except (TypeError, ValueError):
            seq.append(0.0)
    target_len = seq_len * input_dim
    if len(seq) < target_len:
        seq += [0.0] * (target_len - len(seq))
    else:
        seq = seq[:target_len]
    result = []
    for si in range(seq_len):
        result.append(seq[si * input_dim:(si + 1) * input_dim])
    return result


def _extract_features(record: Dict) -> np.ndarray:
    cam = record.get("candidate_action_metrics", {})
    feats = []
    for action in ACTION_LIST:
        entry = cam.get(action, {})
        for metric in ["att", "aql", "awt", "spillback"]:
            val = entry.get(metric, 0.0)
            feats.append(float(val) if val is not None else 0.0)
    mc = record.get("movement_counts", {})
    if isinstance(mc, dict):
        for key in sorted(mc):
            try:
                feats.append(float(mc[key]))
            except (TypeError, ValueError):
                feats.append(0.0)
    arr = np.array(feats, dtype=np.float32)
    target_dim = 64
    if len(arr) < target_dim:
        arr = np.pad(arr, (0, target_dim - len(arr)))
    else:
        arr = arr[:target_dim]
    return arr


def _oracle_gap(costs: np.ndarray, policy_action: str) -> float:
    best_cost = float(np.min(costs))
    policy_idx = action_to_index(policy_action)
    return float(costs[policy_idx]) - best_cost


def _spearman_rho(logits: np.ndarray, costs: np.ndarray) -> float:
    from scipy.stats import spearmanr
    try:
        rho, _ = spearmanr(logits, -costs)
        if not math.isfinite(rho):
            return 0.0
        return float(rho)
    except Exception:
        return 0.0


def _ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confidences)) * abs(
            accuracies[mask].mean() - confidences[mask].mean()
        )
    return float(ece)


def _grid_pos(intersection_id: str) -> Optional[Tuple[int, int]]:
    parts = str(intersection_id).split("_")
    if len(parts) >= 3:
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return None


def _is_border(pos: Optional[Tuple[int, int]], rows: int, cols: int) -> bool:
    if pos is None:
        return False
    r, c = pos
    return r == 1 or r == rows or c == 1 or c == cols


# ── Phase 1: Bottleneck Diagnostic ────────────────────────────────────────


@dataclass
class DiagnosticResult:
    config_name: str
    n_samples: int
    top_action_accuracy: float
    expert_agreement: float
    oracle_gap_mean: float
    oracle_gap_median: float
    ece: float
    spearman_rho_mean: float
    spearman_rho_median: float
    rho_positive_fraction: float
    stratified: Dict[str, Dict]
    margin_accuracy: Dict[str, Dict]
    per_intersection: Dict[str, Dict] = field(default_factory=dict)
    bottleneck: str = ""
    recommendation: str = ""


def run_bottleneck_diagnostic(
    model: PrefixActionPolicy,
    records: List[Dict],
    device: torch.device,
    config_name: str,
    batch_size: int = 8,
    max_samples: int = 2000,
    rows: int = 4,
    cols: int = 4,
) -> DiagnosticResult:
    """Full bottleneck diagnostic on a model-data pair."""

    if len(records) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), max_samples, replace=False)
        records = [records[int(i)] for i in indices]

    n = len(records)
    print(f"  [{config_name}] Running diagnostic on {n} samples...")

    all_logits: List[np.ndarray] = []
    all_costs: List[np.ndarray] = []
    all_intersections: List[str] = []
    all_expert_actions: List[str] = []
    all_congestion: List[float] = []

    model.eval()
    for batch_start in tqdm(range(0, n, batch_size), desc=f"  {config_name}", leave=False):
        batch_records = records[batch_start:batch_start + batch_size]
        seqs, prompts, iids, costs_l, recs_l = [], [], [], [], []

        for rec in batch_records:
            cost_vec = _get_cost_vector(rec)
            if cost_vec is None:
                continue
            seq = _extract_sequence(rec, model.seq_len, model.input_dim)
            if seq is None:
                continue
            raw = str(rec.get("prompt", rec.get("raw_prompt", "")))
            prompt = build_task_prompt(raw_prompt=raw, mode="queue_wait_short")
            seqs.append(seq)
            prompts.append(prompt)
            iids.append(str(rec.get("intersection_id", "")))
            costs_l.append(cost_vec)
            recs_l.append(rec)

        if not seqs:
            continue

        seq_tensor = torch.tensor(seqs, dtype=torch.float32, device=device)
        with torch.no_grad():
            prepared = model.prepare_policy_batch(
                sequences=seq_tensor, prompts=prompts,
                intersection_ids=iids, dataset_hints=[""] * len(iids),
            )
            logits = model.get_action_logits_from_prepared(prepared).float()

        for i, (rec, cost, log) in enumerate(zip(recs_l, costs_l, logits.cpu().numpy())):
            all_logits.append(log)
            all_costs.append(cost)
            all_intersections.append(str(rec.get("intersection_id", "")))
            all_expert_actions.append(str(rec.get("expert_action", "")))
            cam = rec.get("candidate_action_metrics", {})
            max_q = max((v.get("aql", 0.0) or 0.0 for v in cam.values()), default=0.0)
            all_congestion.append(max_q)

    if not all_logits:
        return DiagnosticResult(config_name=config_name, n_samples=0, top_action_accuracy=0.0,
                                expert_agreement=0.0, oracle_gap_mean=0.0, oracle_gap_median=0.0,
                                ece=0.0, spearman_rho_mean=0.0, spearman_rho_median=0.0,
                                rho_positive_fraction=0.0, stratified={}, margin_accuracy={},
                                bottleneck="no_data", recommendation="No valid samples.")

    logits_arr = np.stack(all_logits, axis=0)
    costs_arr = np.stack(all_costs, axis=0)
    congestion_arr = np.array(all_congestion)
    n_actual = len(all_logits)

    # Basic metrics
    policy_argmax = np.argmax(logits_arr, axis=1)
    sim_best = np.argmin(costs_arr, axis=1)
    correct = (policy_argmax == sim_best).astype(np.float32)
    top_acc = float(np.mean(correct))

    expert_correct = np.array([
        policy_argmax[i] == action_to_index(all_expert_actions[i])
        for i in range(n_actual)
    ], dtype=np.float32)
    expert_ag = float(np.mean(expert_correct))

    oracle_gaps = np.array([
        _oracle_gap(costs_arr[i], ACTION_LIST[policy_argmax[i]])
        for i in range(n_actual)
    ])
    gap_mean = float(np.mean(oracle_gaps))
    gap_median = float(np.median(oracle_gaps))

    # ECE
    probs = np.exp(logits_arr) / np.exp(logits_arr).sum(axis=1, keepdims=True)
    confidences = probs[np.arange(n_actual), policy_argmax]
    ece_val = _ece(confidences, correct)

    # Spearman ρ
    rho_vals = np.array([_spearman_rho(logits_arr[i], costs_arr[i]) for i in range(n_actual)])
    rho_mean = float(np.mean(rho_vals))
    rho_median = float(np.median(rho_vals))
    rho_pos = float(np.mean(rho_vals > 0))

    # Margin-conditioned accuracy
    sorted_l = np.sort(logits_arr, axis=1)
    margins = sorted_l[:, -1] - sorted_l[:, -2]
    margin_bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float("inf"))]
    margin_acc = {}
    for lo, hi in margin_bins:
        mask = (margins >= lo) & (margins < hi)
        if mask.sum() > 0:
            margin_acc[f"margin_{lo}_{hi}"] = {
                "accuracy": round(float(correct[mask].mean()), 4),
                "count": int(mask.sum()),
            }

    # Stratified by congestion
    q_med = float(np.median(congestion_arr))
    lo_mask = congestion_arr <= q_med
    hi_mask = congestion_arr > q_med
    stratified = {
        "low_congestion": {
            "count": int(lo_mask.sum()), "accuracy": float(correct[lo_mask].mean()) if lo_mask.any() else 0.0,
            "oracle_gap": float(oracle_gaps[lo_mask].mean()) if lo_mask.any() else 0.0,
            "rho": float(rho_vals[lo_mask].mean()) if lo_mask.any() else 0.0,
        },
        "high_congestion": {
            "count": int(hi_mask.sum()), "accuracy": float(correct[hi_mask].mean()) if hi_mask.any() else 0.0,
            "oracle_gap": float(oracle_gaps[hi_mask].mean()) if hi_mask.any() else 0.0,
            "rho": float(rho_vals[hi_mask].mean()) if hi_mask.any() else 0.0,
        },
    }

    # Per-intersection analysis
    per_intersection = {}
    by_iid = defaultdict(list)
    for i, iid in enumerate(all_intersections):
        by_iid[iid].append(i)
    for iid, indices in sorted(by_iid.items()):
        idx_arr = np.array(indices)
        pos = _grid_pos(iid)
        per_intersection[iid] = {
            "count": len(indices),
            "accuracy": float(correct[idx_arr].mean()),
            "oracle_gap": float(oracle_gaps[idx_arr].mean()),
            "rho": float(rho_vals[idx_arr].mean()),
            "is_border": _is_border(pos, rows, cols),
            "position": f"({pos[0]},{pos[1]})" if pos else "unknown",
        }

    # Diagnosis
    if rho_mean < 0.3:
        bottleneck = "representation_failure"
        rec = ("Low rank correlation (ρ<0.3): policy ranking poorly aligned with SimExpert. "
               "Invest in TimeSeriesProjector capacity or prefix token count.")
    elif rho_mean > 0.5 and ece_val > 0.2:
        bottleneck = "calibration_failure"
        rec = (f"Good rank correlation (ρ={rho_mean:.2f}) but high ECE ({ece_val:.3f}). "
               "Apply temperature scaling or label smoothing.")
    elif rho_mean > 0.5 and ece_val <= 0.2:
        bottleneck = "good_alignment"
        rec = "Well-calibrated. Remaining gap from prefix capacity or SimExpert noise."
    else:
        bottleneck = "mixed"
        rec = f"Mixed signals (ρ={rho_mean:.2f}, ECE={ece_val:.3f}). Review per-state diagnostics."

    return DiagnosticResult(
        config_name=config_name, n_samples=n_actual,
        top_action_accuracy=round(top_acc, 4), expert_agreement=round(expert_ag, 4),
        oracle_gap_mean=round(gap_mean, 4), oracle_gap_median=round(gap_median, 4),
        ece=round(ece_val, 4), spearman_rho_mean=round(rho_mean, 4),
        spearman_rho_median=round(rho_median, 4), rho_positive_fraction=round(rho_pos, 4),
        stratified=stratified, margin_accuracy=margin_acc,
        per_intersection=per_intersection, bottleneck=bottleneck, recommendation=rec,
    )


# ── Phase 2: Full Active Query ────────────────────────────────────────────


@dataclass
class ActiveQueryResult:
    method: str
    K: int
    seed: int
    top1_cost_regret: float
    expected_cost_regret: float
    action_agreement: float
    feature_diversity: float
    action_distribution: Dict[str, float] = field(default_factory=dict)


def run_full_active_query(
    model: PrefixActionPolicy,
    target_records: List[Dict],
    source_records: List[Dict],
    device: torch.device,
    methods: List[str],
    K_values: List[int],
    seeds: List[int],
    batch_size: int = 8,
    max_states: int = 2000,
) -> List[ActiveQueryResult]:
    """Full 6-method × K × seed active query comparison."""

    # Collect target states with policy logits
    print(f"  Collecting target states (max {max_states})...")
    state_logits, state_costs, state_features = [], [], []
    model.eval()

    if len(target_records) > max_states:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(target_records), max_states, replace=False)
        target_records = [target_records[int(i)] for i in indices]

    valid_records = []
    for batch_start in tqdm(range(0, len(target_records), batch_size), desc="  Collecting"):
        batch = target_records[batch_start:batch_start + batch_size]
        seqs, prompts, iids, costs_l, recs_l = [], [], [], [], []
        for rec in batch:
            cost_vec = _get_cost_vector(rec)
            if cost_vec is None:
                continue
            seq = _extract_sequence(rec, model.seq_len, model.input_dim)
            if seq is None:
                continue
            raw = str(rec.get("prompt", rec.get("raw_prompt", "")))
            prompt = build_task_prompt(raw_prompt=raw, mode="queue_wait_short")
            seqs.append(seq)
            prompts.append(prompt)
            iids.append(str(rec.get("intersection_id", "")))
            costs_l.append(cost_vec)
            recs_l.append(rec)
        if not seqs:
            continue
        seq_tensor = torch.tensor(seqs, dtype=torch.float32, device=device)
        with torch.no_grad():
            prepared = model.prepare_policy_batch(sequences=seq_tensor, prompts=prompts, intersection_ids=iids)
            logits = model.get_action_logits_from_prepared(prepared).float()
        for i, (rec, cost, log) in enumerate(zip(recs_l, costs_l, logits.cpu().numpy())):
            state_logits.append(log)
            state_costs.append(cost)
            state_features.append(_extract_features(rec))
            valid_records.append(rec)

    N = len(state_logits)
    print(f"  Collected {N} valid target states")
    if N < min(K_values):
        print(f"  WARNING: Not enough states for smallest K={min(K_values)}")
        return []

    all_logits = np.stack(state_logits, axis=0)
    all_costs = np.stack(state_costs, axis=0)
    all_features = np.stack(state_features, axis=0)

    # Source features for OOD
    source_feats = []
    src_limit = min(len(source_records), 1000)
    for i, rec in enumerate(source_records[:src_limit]):
        if _has_full_metrics(rec):
            source_feats.append(_extract_features(rec))
    source_features_arr = np.stack(source_feats, axis=0) if source_feats else None
    print(f"  Source features: {source_features_arr.shape if source_features_arr is not None else 'None'}")

    results: List[ActiveQueryResult] = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        rng = np.random.RandomState(seed)

        for method_name in methods:
            # Create ensemble logits (MC dropout simulation for now)
            ensemble_logits = None
            if method_name in ("full_ensemble", "regret_aware"):
                ensemble_logits = []
                for m in range(10):
                    noise = rng.normal(0, 0.15, size=all_logits.shape).astype(np.float32)
                    ensemble_logits.append(all_logits + noise)

            acquisition = create_acquisition(
                method=method_name, ensemble_logits=ensemble_logits,
                source_features=source_features_arr, seed=seed,
            )

            for K in K_values:
                selected_idx, scores = acquisition.select(all_logits, K, features=all_features, costs=all_costs)

                sel_logits = all_logits[selected_idx]
                sel_costs = all_costs[selected_idx]
                sel_features = all_features[selected_idx]

                regret_metrics = compute_regret_metrics(sel_logits, sel_costs)

                policy_actions = [ACTION_LIST[int(np.argmax(l))] for l in sel_logits]
                sim_best_actions = [ACTION_LIST[int(np.argmin(c))] for c in sel_costs]
                agreement = sum(1 for p, s in zip(policy_actions, sim_best_actions) if p == s) / K

                if K > 1:
                    norms = sel_features / (np.linalg.norm(sel_features, axis=-1, keepdims=True) + 1e-10)
                    sim_mat = np.dot(norms, norms.T)
                    np.fill_diagonal(sim_mat, 0)
                    diversity = 1.0 - float(np.max(sim_mat)) / 2.0
                else:
                    diversity = 0.0

                action_dist = {}
                for a in policy_actions:
                    action_dist[a] = action_dist.get(a, 0) + 1
                action_dist = {k: v / K for k, v in action_dist.items()}

                results.append(ActiveQueryResult(
                    method=method_name, K=K, seed=seed,
                    top1_cost_regret=round(regret_metrics["top1_cost_regret"], 4),
                    expected_cost_regret=round(regret_metrics["expected_cost_regret"], 4),
                    action_agreement=round(agreement, 4),
                    feature_diversity=round(diversity, 4),
                    action_distribution=action_dist,
                ))

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive Stage 3 Experiment Suite")
    parser.add_argument("--jinan-data", nargs="+", required=True, help="Jinan JSONL data files")
    parser.add_argument("--hangzhou-data", nargs="+", required=True, help="Hangzhou JSONL data files")
    parser.add_argument("--jinan-ckpt", type=str, required=True, help="Jinan-trained model checkpoint")
    parser.add_argument("--hangzhou-ckpt", type=str, required=True, help="Hangzhou-trained model checkpoint")
    parser.add_argument("--llm-path", type=str, required=True, help="Qwen3-VL model path")
    parser.add_argument("--output-dir", default="comprehensive_results", help="Output directory")
    parser.add_argument("--max-diagnostic-samples", type=int, default=2000)
    parser.add_argument("--max-active-states", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--K-values", nargs="+", type=int, default=[8, 16, 32, 64, 120])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025])
    parser.add_argument("--methods", nargs="+",
                        default=["random", "entropy", "ood", "core_set", "full_ensemble", "regret_aware"])
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load all data ──
    def load_records(paths: List[str]) -> List[Dict]:
        records = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = _safe_parse(line)
                    if rec and _has_full_metrics(rec):
                        records.append(rec)
        return records

    print("\nLoading data...")
    jinan_records = load_records(args.jinan_data)
    hangzhou_records = load_records(args.hangzhou_data)
    print(f"  Jinan: {len(jinan_records)} valid records")
    print(f"  Hangzhou: {len(hangzhou_records)} valid records")

    # ── Phase 1: Bottleneck Diagnostic ──
    print("\n" + "=" * 70)
    print("PHASE 1: BOTTLENECK DIAGNOSTIC (4 configurations × {} samples)".format(
        args.max_diagnostic_samples))
    print("=" * 70)

    diagnostic_results: List[DiagnosticResult] = []

    # Config 1: Jinan model on Jinan data
    print("\n[Config 1/4] Jinan model → Jinan data (in-network baseline)")
    model_jinan = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=args.seq_len, image_size=224,
        periodicity=16, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens, prefix_mode="prefix",
        max_length=args.max_length, torch_dtype=torch.bfloat16, device=device,
        with_value_head=False,
    )
    model_jinan.load_checkpoint(args.jinan_ckpt, load_value_head=False)
    model_jinan.eval()
    r1 = run_bottleneck_diagnostic(model_jinan, jinan_records, device,
                                    "Jinan→Jinan", args.batch_size,
                                    args.max_diagnostic_samples, rows=3, cols=4)
    diagnostic_results.append(r1)
    del model_jinan
    torch.cuda.empty_cache()

    # Config 2: Jinan model on Hangzhou data
    print("\n[Config 2/4] Jinan model → Hangzhou data (cross-network)")
    model_jinan2 = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=args.seq_len, image_size=224,
        periodicity=16, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens, prefix_mode="prefix",
        max_length=args.max_length, torch_dtype=torch.bfloat16, device=device,
        with_value_head=False,
    )
    model_jinan2.load_checkpoint(args.jinan_ckpt, load_value_head=False)
    model_jinan2.eval()
    r2 = run_bottleneck_diagnostic(model_jinan2, hangzhou_records, device,
                                    "Jinan→Hangzhou", args.batch_size,
                                    args.max_diagnostic_samples, rows=4, cols=4)
    diagnostic_results.append(r2)
    del model_jinan2
    torch.cuda.empty_cache()

    # Config 3: Hangzhou model on Hangzhou data
    print("\n[Config 3/4] Hangzhou model → Hangzhou data (in-network baseline)")
    model_hz = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=args.seq_len, image_size=224,
        periodicity=16, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens, prefix_mode="prefix",
        max_length=args.max_length, torch_dtype=torch.bfloat16, device=device,
        with_value_head=False,
    )
    model_hz.load_checkpoint(args.hangzhou_ckpt, load_value_head=False)
    model_hz.eval()
    r3 = run_bottleneck_diagnostic(model_hz, hangzhou_records, device,
                                    "Hangzhou→Hangzhou", args.batch_size,
                                    args.max_diagnostic_samples, rows=4, cols=4)
    diagnostic_results.append(r3)

    # Config 4: Hangzhou model on Jinan data
    print("\n[Config 4/4] Hangzhou model → Jinan data (cross-network)")
    r4 = run_bottleneck_diagnostic(model_hz, jinan_records, device,
                                    "Hangzhou→Jinan", args.batch_size,
                                    args.max_diagnostic_samples, rows=3, cols=4)
    diagnostic_results.append(r4)
    del model_hz
    torch.cuda.empty_cache()

    # Print diagnostic summary
    print("\n" + "-" * 70)
    print("BOTTLENECK DIAGNOSTIC SUMMARY")
    print("-" * 70)
    print(f"{'Config':<25s} {'N':>5s} {'Acc':>7s} {'ρ':>7s} {'ECE':>6s} {'Gap':>7s} {'Diagnosis':>25s}")
    print("-" * 70)
    for r in diagnostic_results:
        print(f"{r.config_name:<25s} {r.n_samples:>5d} {r.top_action_accuracy:>6.1%} "
              f"{r.spearman_rho_mean:>7.3f} {r.ece:>6.3f} {r.oracle_gap_median:>7.3f} "
              f"{r.bottleneck:>25s}")

    # ── Phase 2: Full Active Query ──
    print("\n" + "=" * 70)
    print("PHASE 2: FULL 6-METHOD ACTIVE QUERY ({} methods × {} K-values × {} seeds)".format(
        len(args.methods), len(args.K_values), len(args.seeds)))
    print("=" * 70)

    # Use Jinan→Hangzhou as the real cross-network scenario
    model_aq = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=args.seq_len, image_size=224,
        periodicity=16, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens, prefix_mode="prefix",
        max_length=args.max_length, torch_dtype=torch.bfloat16, device=device,
        with_value_head=False,
    )
    model_aq.load_checkpoint(args.jinan_ckpt, load_value_head=False)
    model_aq.eval()

    aq_results = run_full_active_query(
        model=model_aq,
        target_records=hangzhou_records,
        source_records=jinan_records,
        device=device,
        methods=args.methods,
        K_values=args.K_values,
        seeds=args.seeds,
        batch_size=args.batch_size,
        max_states=args.max_active_states,
    )
    del model_aq
    torch.cuda.empty_cache()

    # Print active query summary
    if aq_results:
        print("\n" + "-" * 70)
        print("ACTIVE QUERY SUMMARY (averaged over seeds)")
        print("-" * 70)
        # Aggregate by method × K
        agg: Dict[Tuple[str, int], List[ActiveQueryResult]] = defaultdict(list)
        for r in aq_results:
            agg[(r.method, r.K)].append(r)

        print(f"{'Method':<20s} {'K':>4s} {'Regret':>8s} {'Agree':>7s} {'Div':>6s}")
        print("-" * 50)
        for (method, K), items in sorted(agg.items()):
            avg_regret = np.mean([it.top1_cost_regret for it in items])
            avg_agree = np.mean([it.action_agreement for it in items])
            avg_div = np.mean([it.feature_diversity for it in items])
            print(f"{method:<20s} {K:>4d} {avg_regret:>8.4f} {avg_agree:>6.1%} {avg_div:>6.3f}")

    # ── Save all results ──
    print("\nSaving results...")

    # Diagnostic results
    diag_data = []
    for r in diagnostic_results:
        diag_data.append({
            "config": r.config_name, "n_samples": r.n_samples,
            "top_action_accuracy": r.top_action_accuracy,
            "expert_agreement": r.expert_agreement,
            "oracle_gap_mean": r.oracle_gap_mean, "oracle_gap_median": r.oracle_gap_median,
            "ece": r.ece, "spearman_rho_mean": r.spearman_rho_mean,
            "spearman_rho_median": r.spearman_rho_median,
            "rho_positive_fraction": r.rho_positive_fraction,
            "bottleneck": r.bottleneck, "recommendation": r.recommendation,
            "stratified": r.stratified, "margin_accuracy": r.margin_accuracy,
            "per_intersection": r.per_intersection,
        })

    with open(os.path.join(args.output_dir, "bottleneck_diagnostic.json"), "w") as f:
        json.dump(diag_data, f, indent=2)

    # Active query results
    aq_data = []
    for r in aq_results:
        aq_data.append({
            "method": r.method, "K": r.K, "seed": r.seed,
            "top1_cost_regret": r.top1_cost_regret,
            "expected_cost_regret": r.expected_cost_regret,
            "action_agreement": r.action_agreement,
            "feature_diversity": r.feature_diversity,
            "action_distribution": r.action_distribution,
        })

    with open(os.path.join(args.output_dir, "active_query_full.json"), "w") as f:
        json.dump(aq_data, f, indent=2)

    # Summary CSV
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,config,metric,value\n")
        for r in diag_data:
            for key in ["top_action_accuracy", "spearman_rho_mean", "ece",
                         "oracle_gap_mean", "oracle_gap_median", "expert_agreement"]:
                f.write(f"diagnostic,{r['config']},{key},{r[key]}\n")
        for r in aq_data:
            for key in ["top1_cost_regret", "expected_cost_regret", "action_agreement", "feature_diversity"]:
                val = getattr(r, key) if hasattr(r, key) else r.get(key, "")
                f.write(f"active_query,{r['method']}_K{r['K']}_seed{r['seed']},{key},{val}\n")

    print(f"\nSaved all results to {args.output_dir}/")
    print(f"  - bottleneck_diagnostic.json")
    print(f"  - active_query_full.json")
    print(f"  - summary.csv")
    print("\n✅ Comprehensive experiment suite complete!")


if __name__ == "__main__":
    main()
