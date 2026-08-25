#!/usr/bin/env python3
"""Temperature scaling and mode-collapse analysis.

Key finding from comprehensive experiments: ALL models exhibit ETWT mode collapse
(100% of predictions are ETWT), leading to representation_failure (ρ < 0.3).

This script:
  1. Quantifies the ETWT bias (per-class prediction frequency)
  2. Applies temperature scaling to logits and measures ρ improvement
  3. Finds the optimal temperature per model-config pair
  4. Tests class-balanced prediction via logit bias correction
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.prompt_utils import build_task_prompt

MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]


def _safe_parse(line):
    try: return json.loads(line.strip())
    except: return None


def _has_full_metrics(r):
    cam = r.get("candidate_action_metrics")
    return isinstance(cam, dict) and len(cam) >= 4


def _get_cost_vector(r):
    cam = r.get("candidate_action_metrics", {})
    costs = []
    for a in ACTION_LIST:
        s = (cam.get(a) or {}).get("score")
        if s is not None and math.isfinite(float(s)):
            costs.append(float(s))
        else:
            return None
    return np.array(costs, dtype=np.float32)


def _extract_sequence(r, seq_len, input_dim):
    mc = r.get("movement_counts", {})
    if not isinstance(mc, dict) or not mc:
        return None
    seq = []
    for k in MOVEMENT_KEYS:
        try: seq.append(float(mc.get(k, 0)))
        except: seq.append(0.0)
    target = seq_len * input_dim
    if len(seq) < target:
        seq += [0.0] * (target - len(seq))
    else:
        seq = seq[:target]
    return [[seq[si * input_dim + di] for di in range(input_dim)] for si in range(seq_len)]


def _spearman_rho(logits, costs):
    from scipy.stats import spearmanr
    try:
        rho, _ = spearmanr(logits, -costs)
        return float(rho) if math.isfinite(rho) else 0.0
    except:
        return 0.0


def _ece(confidences, accuracies, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0: continue
        ece += (mask.sum() / len(confidences)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def collect_logits_costs(model, records, device, batch_size=8, max_samples=2000):
    if len(records) > max_samples:
        indices = np.random.RandomState(42).choice(len(records), max_samples, replace=False)
        records = [records[int(i)] for i in indices]

    all_logits, all_costs, all_experts = [], [], []
    model.eval()
    for bs in range(0, len(records), batch_size):
        batch = records[bs:bs + batch_size]
        seqs, prompts, iids, costs_l, recs_l = [], [], [], [], []
        for rec in batch:
            cv = _get_cost_vector(rec)
            if cv is None: continue
            seq = _extract_sequence(rec, model.seq_len, model.input_dim)
            if seq is None: continue
            raw = str(rec.get("prompt", rec.get("raw_prompt", "")))
            prompt = build_task_prompt(raw_prompt=raw, mode="queue_wait_short")
            seqs.append(seq); prompts.append(prompt)
            iids.append(str(rec.get("intersection_id", "")))
            costs_l.append(cv); recs_l.append(rec)
        if not seqs: continue
        seq_t = torch.tensor(seqs, dtype=torch.float32, device=device)
        with torch.no_grad():
            prepared = model.prepare_policy_batch(sequences=seq_t, prompts=prompts, intersection_ids=iids)
            logits = model.get_action_logits_from_prepared(prepared).float()
        for rec, cv, log in zip(recs_l, costs_l, logits.cpu().numpy()):
            all_logits.append(log); all_costs.append(cv)
            all_experts.append(str(rec.get("expert_action", "")))
    return np.stack(all_logits), np.stack(all_costs), all_experts


def evaluate_with_temperature(logits, costs, experts, temperature):
    """Apply temperature scaling and compute metrics."""
    scaled = logits / max(temperature, 0.01)
    probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
    preds = np.argmax(scaled, axis=1)
    sim_best = np.argmin(costs, axis=1)

    correct = (preds == sim_best).astype(np.float32)
    acc = float(np.mean(correct))

    expert_correct = np.array([preds[i] == action_to_index(experts[i]) for i in range(len(preds))], dtype=np.float32)

    oracle_gaps = np.array([float(costs[i, preds[i]] - costs[i].min()) for i in range(len(preds))])

    confidences = probs[np.arange(len(preds)), preds]
    ece_v = _ece(confidences, correct)

    rho_vals = np.array([_spearman_rho(scaled[i], costs[i]) for i in range(len(preds))])
    rho_mean = float(np.mean(rho_vals))

    # Per-class distribution
    class_dist = {}
    for i in range(len(preds)):
        a = ACTION_LIST[preds[i]]
        class_dist[a] = class_dist.get(a, 0) + 1
    class_dist = {k: v / len(preds) for k, v in class_dist.items()}

    return {
        "temperature": temperature,
        "accuracy": round(acc, 4),
        "expert_agreement": round(float(np.mean(expert_correct)), 4),
        "oracle_gap_mean": round(float(np.mean(oracle_gaps)), 4),
        "oracle_gap_median": round(float(np.median(oracle_gaps)), 4),
        "ece": round(ece_v, 4),
        "spearman_rho_mean": round(rho_mean, 4),
        "spearman_rho_median": round(float(np.median(rho_vals)), 4),
        "class_distribution": class_dist,
        "n_samples": len(preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Temperature Scaling Analysis")
    parser.add_argument("--jinan-data", nargs="+", required=True)
    parser.add_argument("--hangzhou-data", nargs="+", required=True)
    parser.add_argument("--jinan-ckpt", type=str, required=True)
    parser.add_argument("--hangzhou-ckpt", type=str, required=True)
    parser.add_argument("--llm-path", type=str, required=True)
    parser.add_argument("--output-dir", default="temperature_analysis")
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    def load_data(paths):
        records = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    r = _safe_parse(line)
                    if r and _has_full_metrics(r): records.append(r)
        return records

    jinan_records = load_data(args.jinan_data)
    hangzhou_records = load_data(args.hangzhou_data)
    print(f"Data: Jinan={len(jinan_records)}, Hangzhou={len(hangzhou_records)}")

    # Temperature sweep for Jinan model on Jinan data
    print("\n" + "=" * 60)
    print("Sweep 1: Jinan model → Jinan data (temperature scaling)")
    print("=" * 60)

    model = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=16, image_size=224, periodicity=16,
        input_dim=8, hidden_dim=64, prefix_tokens=args.prefix_tokens,
        prefix_mode="prefix", max_length=args.max_length,
        torch_dtype=torch.bfloat16, device=device, with_value_head=False,
    )
    model.load_checkpoint(args.jinan_ckpt, load_value_head=False)
    model.eval()

    logits_jj, costs_jj, experts_jj = collect_logits_costs(
        model, jinan_records, device, args.batch_size, args.max_samples)
    print(f"  Collected {len(logits_jj)} samples")

    # Baseline (T=1.0)
    print("\n--- Temperature sweep ---")
    print(f"{'T':>6s} {'Acc':>7s} {'ρ':>7s} {'ECE':>6s} {'Gap(med)':>9s} {'ETWT%':>7s}")
    print("-" * 50)

    all_results = []
    temperatures = [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    for T in temperatures:
        result = evaluate_with_temperature(logits_jj, costs_jj, experts_jj, T)
        result["config"] = "Jinan→Jinan"
        all_results.append(result)
        etwt_pct = result["class_distribution"].get("ETWT", 0) * 100
        print(f"{T:>6.1f} {result['accuracy']:>7.1%} {result['spearman_rho_mean']:>7.3f} "
              f"{result['ece']:>6.3f} {result['oracle_gap_median']:>9.4f} {etwt_pct:>6.1f}%")

    # Also test logit bias correction (penalize ETWT)
    print("\n--- Bias correction (penalize ETWT logit) ---")
    bias_values = [0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    etwt_idx = action_to_index("ETWT")
    for bias in bias_values:
        biased_logits = logits_jj.copy()
        biased_logits[:, etwt_idx] += bias  # negative bias = penalize ETWT
        result = evaluate_with_temperature(biased_logits, costs_jj, experts_jj, 1.0)
        result["config"] = f"Jinan→Jinan_bias{bias}"
        all_results.append(result)
        etwt_pct = result["class_distribution"].get("ETWT", 0) * 100
        print(f"bias={bias:>4.1f} {result['accuracy']:>7.1%} {result['spearman_rho_mean']:>7.3f} "
              f"{result['ece']:>6.3f} {result['oracle_gap_median']:>9.4f} {etwt_pct:>6.1f}%")

    del model; torch.cuda.empty_cache()

    # Temperature sweep for Hangzhou model on Hangzhou data
    print("\n" + "=" * 60)
    print("Sweep 2: Hangzhou model → Hangzhou data (temperature scaling)")
    print("=" * 60)

    model2 = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=16, image_size=224, periodicity=16,
        input_dim=8, hidden_dim=64, prefix_tokens=args.prefix_tokens,
        prefix_mode="prefix", max_length=args.max_length,
        torch_dtype=torch.bfloat16, device=device, with_value_head=False,
    )
    model2.load_checkpoint(args.hangzhou_ckpt, load_value_head=False)
    model2.eval()

    logits_hh, costs_hh, experts_hh = collect_logits_costs(
        model2, hangzhou_records, device, args.batch_size, args.max_samples)
    print(f"  Collected {len(logits_hh)} samples")

    print(f"\n{'T':>6s} {'Acc':>7s} {'ρ':>7s} {'ECE':>6s} {'Gap(med)':>9s} {'ETWT%':>7s}")
    print("-" * 50)
    for T in temperatures:
        result = evaluate_with_temperature(logits_hh, costs_hh, experts_hh, T)
        result["config"] = "Hangzhou→Hangzhou"
        all_results.append(result)
        etwt_pct = result["class_distribution"].get("ETWT", 0) * 100
        print(f"{T:>6.1f} {result['accuracy']:>7.1%} {result['spearman_rho_mean']:>7.3f} "
              f"{result['ece']:>6.3f} {result['oracle_gap_median']:>9.4f} {etwt_pct:>6.1f}%")

    # Bias correction for Hangzhou
    print("\n--- Bias correction ---")
    for bias in bias_values:
        biased_logits = logits_hh.copy()
        biased_logits[:, etwt_idx] += bias
        result = evaluate_with_temperature(biased_logits, costs_hh, experts_hh, 1.0)
        result["config"] = f"Hangzhou→Hangzhou_bias{bias}"
        all_results.append(result)
        etwt_pct = result["class_distribution"].get("ETWT", 0) * 100
        print(f"bias={bias:>4.1f} {result['accuracy']:>7.1%} {result['spearman_rho_mean']:>7.3f} "
              f"{result['ece']:>6.3f} {result['oracle_gap_median']:>9.4f} {etwt_pct:>6.1f}%")

    del model2; torch.cuda.empty_cache()

    # Save
    with open(os.path.join(args.output_dir, "temperature_sweep.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Find best temperature per config
    best_by_config = {}
    for r in all_results:
        cfg = r["config"]
        if cfg not in best_by_config or r["spearman_rho_mean"] > best_by_config[cfg]["spearman_rho_mean"]:
            best_by_config[cfg] = r

    print("\n" + "=" * 60)
    print("BEST RESULTS PER CONFIG")
    print("=" * 60)
    for cfg, r in sorted(best_by_config.items()):
        T = r.get("temperature", r.get("config", "").split("_bias")[-1] if "_bias" in r.get("config", "") else "N/A")
        print(f"  {cfg}: T={T}, acc={r['accuracy']:.1%}, ρ={r['spearman_rho_mean']:.3f}, "
              f"ECE={r['ece']:.3f}, ETWT={r['class_distribution'].get('ETWT', 0):.1%}")

    print(f"\nSaved: {args.output_dir}/temperature_sweep.json")


if __name__ == "__main__":
    main()
