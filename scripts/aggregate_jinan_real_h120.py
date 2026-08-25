#!/usr/bin/env python3
"""Aggregate responsive H120 jinan results across the 3 parallel GPU jobs.

Reads per-training-seed eval_summary.json files from train_h120_gpu{0,1,2},
combines into n=9 means for K=32 and K=118, and reports delta vs zero-shot.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

METRICS = ["ATT", "AQL", "AWT"]


def load_eval_summary(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    return payload.get("mean", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--zero_shot_dir", default=None)
    args = parser.parse_args()

    zero_shot = None
    if args.zero_shot_dir and os.path.exists(os.path.join(args.zero_shot_dir, "eval_summary.json")):
        with open(os.path.join(args.zero_shot_dir, "eval_summary.json")) as f:
            zs = json.load(f)
        zero_shot = {m: zs[m]["mean"] for m in METRICS}

    report = {"zero_shot": zero_shot, "arms": {}}
    for K in (32, 118):
        vals = {m: [] for m in METRICS}
        per_seed = []
        for gpu in (0, 1, 2):
            job = os.path.join(args.output_dir, f"train_h120_gpu{gpu}")
            for seed in sorted(os.listdir(os.path.join(job, "c30", f"k{K}"))):
                summ = os.path.join(job, "c30", f"k{K}", seed, "evaluation", "eval_summary.json")
                if not os.path.exists(summ):
                    print(f"  [warn] missing {summ}")
                    continue
                mean = load_eval_summary(summ)
                row = {"seed": seed.lstrip("tr"), **{m: mean[m] for m in METRICS}}
                per_seed.append(row)
                for m in METRICS:
                    vals[m].append(mean[m])
        row = {
            m: {
                "n": len(vals[m]),
                "mean": statistics.fmean(vals[m]),
                "std": statistics.pstdev(vals[m]) if len(vals[m]) > 1 else 0.0,
                "min": min(vals[m]),
                "max": max(vals[m]),
            }
            for m in METRICS
        }
        if zero_shot:
            row["delta_vs_zero_shot"] = {
                m: round(row[m]["mean"] - zero_shot[m], 2) for m in METRICS
            }
        report["arms"][f"H120_K{K}"] = {"aggregate": row, "per_seed": per_seed}
        print(f"\nH120 K={K} (n={row['ATT']['n']}):")
        for m in METRICS:
            z = f"  Δ={row['delta_vs_zero_shot'][m]:+.2f}" if zero_shot else ""
            print(f"  {m}: {row[m]['mean']:.2f}±{row[m]['std']:.2f} [{row[m]['min']:.2f}, {row[m]['max']:.2f}]{z}")
        print("  per-seed AWT:", {s["seed"]: round(s["AWT"], 2) for s in sorted(per_seed, key=lambda x: int(x["seed"]))})

    out = os.path.join(args.output_dir, "h120_aggregate.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
