#!/usr/bin/env python3
"""Create deterministic P1/P2 snapshot partitions for the JSCD pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_groups(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            snapshot_path = str(record.get("snapshot_path", ""))
            if not snapshot_path:
                raise ValueError(f"{path}:{line_number}: missing snapshot_path")
            groups[snapshot_path].append(line)
    if not groups:
        raise ValueError(f"No snapshot records found in {path}")
    return dict(groups)


def partition_snapshot_paths(
    snapshot_paths: list[str], holdout_count: int, seed: int
) -> tuple[list[str], list[str]]:
    if holdout_count <= 0 or holdout_count >= len(snapshot_paths):
        raise ValueError(
            f"holdout_count must be in [1, {len(snapshot_paths) - 1}], got {holdout_count}"
        )
    shuffled = sorted(snapshot_paths)
    random.Random(seed).shuffle(shuffled)
    holdout = sorted(shuffled[:holdout_count])
    training = sorted(shuffled[holdout_count:])
    return holdout, training


def write_partition(path: Path, groups: dict[str, list[str]], snapshot_paths: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for snapshot_path in snapshot_paths:
            for line in groups[snapshot_path]:
                handle.write(line if line.endswith("\n") else f"{line}\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--p1-output", required=True, type=Path)
    parser.add_argument("--p2-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--holdout-snapshots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    groups = load_groups(args.input)
    holdout, training = partition_snapshot_paths(
        list(groups), args.holdout_snapshots, args.seed
    )
    p1_records = write_partition(args.p1_output, groups, holdout)
    p2_records = write_partition(args.p2_output, groups, training)
    manifest = {
        "study": "JSCD P1/P2 deterministic snapshot partition",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "seed": args.seed,
        "total_snapshots": len(groups),
        "p1_holdout_snapshots": holdout,
        "p2_training_snapshots": training,
        "p1_records": p1_records,
        "p2_records": p2_records,
        "p1_output": str(args.p1_output.resolve()),
        "p2_output": str(args.p2_output.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"P1 holdout: {len(holdout)} snapshots / {p1_records} records; "
        f"P2 pool: {len(training)} snapshots / {p2_records} records"
    )


if __name__ == "__main__":
    main()
