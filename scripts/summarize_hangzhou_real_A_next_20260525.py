#!/usr/bin/env python3
"""Summarize Hangzhou-real Stage3 A-variant online runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RESULT_KEYS = {
    "reward": "test_reward_over",
    "queue": "test_avg_queue_len_over",
    "wait": "test_avg_waiting_time_over",
    "travel": "test_avg_travel_time_over",
    "queuing": "test_queuing_vehicle_num_over",
}


BASELINE = {
    "name": "Stage3 pre-A baseline",
    "reward": -625.1703,
    "queue": 70.7167,
    "wait": 55.1263,
    "travel": 325.8897,
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_state_action(repo_root: Path, run_root: Path, variant: str) -> Path | None:
    candidates: list[Path] = []
    relative_root = run_root
    if run_root.is_absolute():
        try:
            relative_root = run_root.relative_to(repo_root)
        except ValueError:
            relative_root = Path(run_root.name)
    parent = repo_root / "records" / relative_root / variant / "test_runs"
    if parent.is_dir():
        candidates.extend(parent.rglob("state_action.json"))
    direct_parent = run_root / variant
    if direct_parent.is_dir():
        candidates.extend(direct_parent.rglob("state_action.json"))
    existing = sorted({path for path in candidates if path.is_file()})
    return existing[-1] if existing else None


def _summarize_state_action(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "state_action_path": str(path) if path else "",
            "checked": 0,
            "overridden": 0,
            "max_phase_match": 0.0,
            "avg_missed_gap": 0.0,
            "strong_misses": 0,
            "action_counts": {},
        }

    payload = _load_json(path)
    checked = 0
    overridden = 0
    matches = 0
    missed_gaps: list[float] = []
    strong_misses = 0
    action_counts: Counter[str] = Counter()
    protect_counts: Counter[str] = Counter()

    for inter_log in payload if isinstance(payload, list) else []:
        for item in inter_log if isinstance(inter_log, list) else []:
            action = str(item.get("action", ""))
            if action:
                action_counts[action] += 1
            diag = item.get("max_phase_override") or {}
            if not diag.get("enabled"):
                continue
            checked += 1
            overridden += int(bool(diag.get("overridden")))
            final_action = str(diag.get("final_action", action))
            target_phase = str(diag.get("target_phase", ""))
            gap = _as_float(diag.get("gap"), 0.0)
            if final_action == target_phase:
                matches += 1
            else:
                missed_gaps.append(max(0.0, gap))
                if gap >= 10.0:
                    strong_misses += 1
            reason = str(diag.get("protect_reason") or "")
            if reason:
                protect_counts[reason.split(":", 1)[0]] += 1

    avg_missed_gap = sum(missed_gaps) / len(missed_gaps) if missed_gaps else 0.0
    return {
        "state_action_path": str(path),
        "checked": checked,
        "overridden": overridden,
        "override_rate": (overridden / checked) if checked else 0.0,
        "max_phase_match": (matches / checked) if checked else 0.0,
        "avg_missed_gap": avg_missed_gap,
        "strong_misses": strong_misses,
        "action_counts": dict(action_counts),
        "protect_counts": dict(protect_counts),
    }


def _collect_variant(repo_root: Path, run_root: Path, variant_dir: Path) -> dict[str, Any] | None:
    result_path = variant_dir / "artifacts" / "test_result.json"
    if not result_path.is_file():
        return None
    payload = _load_json(result_path)
    result = payload.get("results", payload) if isinstance(payload, dict) else {}
    metrics = {
        key: _as_float(result.get(result_key), 0.0)
        for key, result_key in RESULT_KEYS.items()
    }
    state_action_path = _find_state_action(repo_root, run_root, variant_dir.name)
    diagnostics = _summarize_state_action(state_action_path)
    return {
        "variant": variant_dir.name,
        "result_path": str(result_path),
        **metrics,
        **diagnostics,
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines: list[str] = []
    lines.append("# Hangzhou Real Stage3 A-Next Results")
    lines.append("")
    lines.append(
        "Baseline: reward `-625.1703`, avg queue `70.7167`, "
        "avg wait `55.1263`, avg travel `325.8897`."
    )
    lines.append("")
    lines.append("## Raw Results")
    lines.append("")
    lines.append(
        "| Variant | Reward | Delta | Avg queue | Queue delta | "
        "Avg wait | Wait delta | Avg travel | Travel delta | Queuing vehicles |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: item["reward"], reverse=True):
        queue_delta = (row["queue"] - BASELINE["queue"]) / BASELINE["queue"] * 100.0
        wait_delta = (row["wait"] - BASELINE["wait"]) / BASELINE["wait"] * 100.0
        travel_delta = (row["travel"] - BASELINE["travel"]) / BASELINE["travel"] * 100.0
        lines.append(
            f"| {row['variant']} | {_fmt(row['reward'])} | "
            f"{_fmt(row['reward'] - BASELINE['reward'])} | {_fmt(row['queue'])} | "
            f"{queue_delta:.1f}% | {_fmt(row['wait'])} | {wait_delta:.1f}% | "
            f"{_fmt(row['travel'])} | {travel_delta:.1f}% | {_fmt(row['queuing'], 0)} |"
        )

    lines.append("")
    lines.append("## State-Action Diagnostics")
    lines.append("")
    lines.append(
        "| Variant | Max-phase match | Avg missed gap | Strong misses | "
        "Override applied | Protect reasons |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: item["reward"], reverse=True):
        protect_counts = row.get("protect_counts") or {}
        protect_text = ", ".join(f"{key}:{value}" for key, value in sorted(protect_counts.items()))
        lines.append(
            f"| {row['variant']} | {row['max_phase_match'] * 100.0:.2f}% | "
            f"{_fmt(row['avg_missed_gap'])} | {int(row['strong_misses'])} | "
            f"{int(row['overridden'])}/{int(row['checked'])} | {protect_text or '-'} |"
        )

    if rows:
        best_reward = max(rows, key=lambda item: item["reward"])
        best_wait = min(rows, key=lambda item: item["wait"])
        best_queue = min(rows, key=lambda item: item["queue"])
        lines.append("")
        lines.append("## Quick Read")
        lines.append("")
        lines.append(f"- Best reward: `{best_reward['variant']}` with `{_fmt(best_reward['reward'])}`.")
        lines.append(f"- Best queue: `{best_queue['variant']}` with `{_fmt(best_queue['queue'])}`.")
        lines.append(f"- Best wait: `{best_wait['variant']}` with `{_fmt(best_wait['wait'])}`.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(""))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = args.root
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    run_root = run_root.resolve()
    output = args.output if str(args.output) else run_root / "summary.md"
    if not output.is_absolute():
        output = repo_root / output

    rows: list[dict[str, Any]] = []
    for variant_dir in sorted(run_root.iterdir() if run_root.is_dir() else []):
        if not variant_dir.is_dir():
            continue
        row = _collect_variant(repo_root, run_root, variant_dir)
        if row is not None:
            rows.append(row)

    _write_markdown(rows, output)
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} and {json_output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
