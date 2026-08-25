#!/usr/bin/env python3
"""Reclaim only intermediate checkpoints from completed joint-coreset runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def completed_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    for base in (root / "runs", root / "pilots"):
        if not base.is_dir():
            continue
        for status_path in sorted(base.glob("*/status.json")):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("state") == "complete":
                runs.append(status_path.parent)
    return runs


def candidates(run_dir: Path) -> list[Path]:
    stage1 = run_dir / "stage1"
    stage2 = run_dir / "stage2"
    return [
        *stage1.glob("align_epoch_*.pt"),
        stage1 / "last.pt",
        *stage2.glob("epoch_*.pt"),
        stage2 / "last.pt",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    total_bytes = 0
    total_files = 0
    per_run: list[tuple[str, int, int]] = []
    for run_dir in completed_runs(root):
        files = [path for path in candidates(run_dir) if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        total_files += len(files)
        total_bytes += size
        per_run.append((run_dir.name, len(files), size))
        if args.apply:
            for path in files:
                path.unlink()

    for name, count, size in per_run:
        print(f"{name}\tfiles={count}\tbytes={size}")
    print(f"runs={len(per_run)} files={total_files} bytes={total_bytes} applied={args.apply}")


if __name__ == "__main__":
    main()
