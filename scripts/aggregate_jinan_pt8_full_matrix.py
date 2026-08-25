#!/usr/bin/env python3
"""Aggregate the pt8 jinan_real full-matrix results.

Reads zero-shot + 3 teacher arms (frozen c30, responsive H30, responsive H120),
each with K={32,118} x 9 training seeds, and reports ATT/AQL/AWT mean/std/min/max
plus delta vs zero-shot. Output: full_matrix_aggregate.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

METRICS = ["ATT", "AQL", "AWT"]

# arm label -> training output subdirectory (each holds c30/k{K}/tr{seed}/...)
TEACHERS = {
    "frozen_c30": "train_frozen_c30",
    "responsive_h30": "train_h30",
    "responsive_h120": "train_h120",
}


def load_mean(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    return payload.get("mean", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = args.output_dir

    zero_shot = None
    zs_path = os.path.join(out, "zero_shot", "eval_summary.json")
    if os.path.exists(zs_path):
        with open(zs_path) as f:
            zs = json.load(f)
        zero_shot = {m: zs[m]["mean"] for m in METRICS}

    report = {"zero_shot": zero_shot, "arms": {}}
    for arm, subdir in TEACHERS.items():
        for K in (32, 118):
            vals = {m: [] for m in METRICS}
            per_seed = []
            kdir = os.path.join(out, subdir, "c30", f"k{K}")
            if not os.path.isdir(kdir):
                continue
            for seed in sorted(os.listdir(kdir)):
                summ = os.path.join(kdir, seed, "evaluation", "eval_summary.json")
                if not os.path.exists(summ):
                    continue
                mean = load_mean(summ)
                per_seed.append({"seed": seed.lstrip("tr"), **{m: mean[m] for m in METRICS}})
                for m in METRICS:
                    vals[m].append(mean[m])
            if not vals["ATT"]:
                continue
            agg = {
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
                agg["delta_vs_zero_shot"] = {
                    m: round(agg[m]["mean"] - zero_shot[m], 2) for m in METRICS
                }
            key = f"{arm}_K{K}"
            report["arms"][key] = {"aggregate": agg, "per_seed": per_seed}

    out_path = os.path.join(out, "full_matrix_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=== pt8 jinan_real full matrix ===")
    if zero_shot:
        print("zero-shot:", {m: round(zero_shot[m], 2) for m in METRICS})
    for key, arm in report["arms"].items():
        agg = arm["aggregate"]
        z = ""
        if "delta_vs_zero_shot" in agg:
            z = "".join(f" Δ{m}={agg['delta_vs_zero_shot'][m]:+.2f}" for m in METRICS)
        print(f"{key} (n={agg['ATT']['n']}): "
              f"ATT={agg['ATT']['mean']:.2f}±{agg['ATT']['std']:.2f} "
              f"AQL={agg['AQL']['mean']:.2f}±{agg['AQL']['std']:.2f} "
              f"AWT={agg['AWT']['mean']:.2f}±{agg['AWT']['std']:.2f}{z}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
