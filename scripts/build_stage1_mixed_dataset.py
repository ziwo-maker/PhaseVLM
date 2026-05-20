#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage1_compact_dataset import (  # noqa: E402
    _assign_bins,
    _dedup_by_sequence,
    _read_groups,
    _select_subset,
    _summary,
)


def build_mixed_dataset(
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    seq_len: int,
    target_total_groups: int | None,
    secondary_quota: int | None,
    min_per_bucket: int,
    per_caption_cap: int,
    per_bucket_cap: int,
    per_intersection_cap: int | None,
) -> dict[str, Any]:
    if (target_total_groups is None) == (secondary_quota is None):
        raise ValueError("Specify exactly one of --target_total_groups or --secondary_quota.")

    primary_groups = _read_groups(primary_path, seq_len=seq_len)
    secondary_groups = _read_groups(secondary_path, seq_len=seq_len)
    if not primary_groups:
        raise SystemExit(f"No terminal clip_caption groups found in primary file: {primary_path}")
    if not secondary_groups:
        raise SystemExit(f"No terminal clip_caption groups found in secondary file: {secondary_path}")

    combined = list(primary_groups) + list(secondary_groups)
    _assign_bins(combined)

    primary_dedup = sorted(_dedup_by_sequence(primary_groups), key=lambda g: g.source_index)
    secondary_dedup = sorted(_dedup_by_sequence(secondary_groups), key=lambda g: g.source_index)

    primary_signatures = {group.sequence_signature for group in primary_dedup}
    overlap_count = sum(1 for group in secondary_dedup if group.sequence_signature in primary_signatures)
    secondary_unique = [group for group in secondary_dedup if group.sequence_signature not in primary_signatures]

    if target_total_groups is not None:
        resolved_secondary_quota = max(0, int(target_total_groups) - len(primary_dedup))
    else:
        resolved_secondary_quota = max(0, int(secondary_quota))
    resolved_secondary_quota = min(resolved_secondary_quota, len(secondary_unique))

    if resolved_secondary_quota > 0:
        intersection_count = len({g.intersection_id for g in secondary_unique}) or 1
        resolved_intersection_cap = (
            int(per_intersection_cap)
            if per_intersection_cap is not None and int(per_intersection_cap) > 0
            else max(8, math.ceil(max(1, resolved_secondary_quota) / intersection_count * 1.35))
        )
        selected_secondary = _select_subset(
            secondary_unique,
            target_groups=resolved_secondary_quota,
            min_per_bucket=min_per_bucket,
            per_caption_cap=per_caption_cap,
            per_bucket_cap=per_bucket_cap,
            per_intersection_cap=resolved_intersection_cap,
        )
    else:
        resolved_intersection_cap = int(per_intersection_cap or 0)
        selected_secondary = []

    final_groups = list(primary_dedup) + sorted(selected_secondary, key=lambda g: g.source_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in final_groups:
            for line in sample.raw_lines:
                handle.write(line)
                handle.write("\n")

    summary = {
        "primary_input": str(primary_path),
        "secondary_input": str(secondary_path),
        "output": str(output_path),
        "seq_len": int(seq_len),
        "target_total_groups": int(target_total_groups) if target_total_groups is not None else None,
        "requested_secondary_quota": int(secondary_quota) if secondary_quota is not None else None,
        "resolved_secondary_quota": int(resolved_secondary_quota),
        "resolved_secondary_intersection_cap": int(resolved_intersection_cap),
        "primary_before": _summary(primary_groups),
        "primary_after_dedup": _summary(primary_dedup),
        "secondary_before": _summary(secondary_groups),
        "secondary_after_dedup": _summary(secondary_dedup),
        "secondary_overlap_with_primary": int(overlap_count),
        "secondary_unique_after_cross_dedup": _summary(secondary_unique),
        "selected_secondary": _summary(selected_secondary),
        "final_dataset": _summary(final_groups),
        "selected_secondary_lines": int(sum(len(sample.raw_lines) for sample in selected_secondary)),
        "final_selected_lines": int(sum(len(sample.raw_lines) for sample in final_groups)),
        "secondary_share_of_final_groups": (
            round(len(selected_secondary) / len(final_groups), 6) if final_groups else 0.0
        ),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a mixed Stage-1 dataset from a primary compact jsonl and a supplemental compact jsonl."
    )
    parser.add_argument("--primary", required=True, help="primary compact jsonl path")
    parser.add_argument("--secondary", required=True, help="secondary compact jsonl path")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--target_total_groups", type=int, default=None)
    parser.add_argument("--secondary_quota", type=int, default=None)
    parser.add_argument("--min_per_bucket", type=int, default=1)
    parser.add_argument("--per_caption_cap", type=int, default=3)
    parser.add_argument("--per_bucket_cap", type=int, default=18)
    parser.add_argument("--per_intersection_cap", type=int, default=0)
    args = parser.parse_args()

    summary = build_mixed_dataset(
        primary_path=Path(args.primary),
        secondary_path=Path(args.secondary),
        output_path=Path(args.output),
        seq_len=args.seq_len,
        target_total_groups=args.target_total_groups,
        secondary_quota=args.secondary_quota,
        min_per_bucket=args.min_per_bucket,
        per_caption_cap=args.per_caption_cap,
        per_bucket_cap=args.per_bucket_cap,
        per_intersection_cap=args.per_intersection_cap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
