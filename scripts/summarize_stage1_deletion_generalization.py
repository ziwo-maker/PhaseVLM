#!/usr/bin/env python3
"""Summarize paired single-seed Stage-I deletion transfer evaluations."""

import argparse
import json
import statistics
from pathlib import Path

METRICS = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
}


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for pair in manifest["pairs"]:
        item = {key: pair[key] for key in ("id", "source", "target", "scope")}
        for arm in ("full", "without_stage1"):
            path = root / "results" / pair["id"] / arm / "result.json"
            if not path.is_file():
                item[arm] = {"status": "missing", "path": str(path)}
                continue
            payload = json.loads(path.read_text())
            args = payload["args"]
            if args["seed"] != manifest["seed"] or args["traffic_file"] != pair["traffic_file"]:
                raise ValueError(f"Evaluation contract mismatch: {path}")
            result = payload["results"]
            if isinstance(result, list):
                result = result[-1]
            item[arm] = {
                "status": "ok", "path": str(path),
                **{name: float(result[key]) for name, key in METRICS.items()},
            }
        if all(item[arm]["status"] == "ok" for arm in ("full", "without_stage1")):
            item["without_minus_full"] = {
                name: item["without_stage1"][name] - item["full"][name]
                for name in METRICS
            }
            item["without_minus_full_percent"] = {
                name: 100 * item["without_minus_full"][name] / item["full"][name]
                if item["full"][name] else None
                for name in METRICS
            }
        rows.append(item)

    groups = {}
    for scope in ("within_city", "cross_city"):
        complete = [row for row in rows if row["scope"] == scope and "without_minus_full" in row]
        groups[scope] = {
            "complete_pairs": len(complete),
            "expected_pairs": sum(row["scope"] == scope for row in rows),
            "metrics": {
                name: {
                    "full_better_pairs": sum(row["without_minus_full"][name] > 0 for row in complete),
                    "without_better_pairs": sum(row["without_minus_full"][name] < 0 for row in complete),
                    "ties": sum(row["without_minus_full"][name] == 0 for row in complete),
                    "mean_paired_delta": statistics.mean(row["without_minus_full"][name] for row in complete)
                    if complete else None,
                    "mean_paired_percent": statistics.mean(
                        row["without_minus_full_percent"][name] for row in complete
                        if row["without_minus_full_percent"][name] is not None
                    ) if complete else None,
                }
                for name in METRICS
            },
        }
    return {"seed": manifest["seed"], "pairs": rows, "groups": groups,
            "complete": all("without_minus_full" in row for row in rows)}


def write_report(root: Path, summary: dict) -> None:
    lines = [
        "# Stage I Deletion: Within-City and Cross-City Generalization",
        "",
        f"Single training/evaluation seed: {summary['seed']}. Lower ATT, AQL and AWT are better.",
        "The unit is a directed source-to-target flow pair, not an independent training seed.",
        "",
        "| Scope | Pair | ATT Full / w/o | AQL Full / w/o | AWT Full / w/o |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["pairs"]:
        vals = []
        for name in METRICS:
            vals.append(
                f"{row['full'][name]:.3f} / {row['without_stage1'][name]:.3f}"
                if "without_minus_full" in row else "pending"
            )
        lines.append(f"| {row['scope']} | {row['source']} -> {row['target']} | " + " | ".join(vals) + " |")
    lines.extend(["", "## Paired Direction", "",
                  "Positive w/o-minus-Full values favor Stage I. These descriptive counts do not establish multi-seed stability.", ""])
    for scope, group in summary["groups"].items():
        lines.append(f"### {scope} ({group['complete_pairs']}/{group['expected_pairs']} pairs)")
        lines.append("")
        lines.append("| Metric | Full better | w/o better | Mean paired delta | Mean paired % |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, result in group["metrics"].items():
            delta = result["mean_paired_delta"]
            pct = result["mean_paired_percent"]
            lines.append(f"| {name} | {result['full_better_pairs']} | {result['without_better_pairs']} | "
                         f"{delta:.3f} | {pct:.2f}% |" if delta is not None and pct is not None
                         else f"| {name} | 0 | 0 | pending | pending |")
        lines.append("")
    (root / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run_root)
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.run_root, summary)


if __name__ == "__main__":
    main()
