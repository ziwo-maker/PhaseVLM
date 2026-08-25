#!/usr/bin/env python3
"""Build reproducible, non-overlapping CityFlow time windows from Hangzhou_real.

This produces demand-only evaluation inputs. It neither reads labels nor changes
the Stage 2 training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SLICE_WIDTH_S = 1200
SOURCE_NAME = "anon_4_4_hangzhou_real.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/Hangzhou/4_4"))
    parser.add_argument("--source", default=SOURCE_NAME)
    parser.add_argument("--slice-width-s", type=int, default=SLICE_WIDTH_S)
    parser.add_argument("--slice-count", type=int, default=3)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    source = data_dir / args.source
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.slice_width_s <= 0 or args.slice_count <= 0:
        raise ValueError("slice width and count must be positive")

    flows: list[dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(flows, list):
        raise ValueError(f"Expected a JSON flow list: {source}")

    outputs: list[dict[str, Any]] = []
    for slice_index in range(args.slice_count):
        start_s = slice_index * args.slice_width_s
        end_s = start_s + args.slice_width_s
        selected: list[dict[str, Any]] = []
        for flow in flows:
            if not isinstance(flow, dict):
                raise ValueError("Flow records must be objects")
            flow_start = int(flow.get("startTime", -1))
            flow_end = int(flow.get("endTime", -1))
            if flow_end < flow_start:
                raise ValueError(f"Invalid flow interval: {flow_start}>{flow_end}")
            if flow_start < start_s or flow_end >= end_s:
                continue
            shifted = dict(flow)
            shifted["startTime"] = flow_start - start_s
            shifted["endTime"] = flow_end - start_s
            selected.append(shifted)
        if not selected:
            raise RuntimeError(f"No flows found in slice [{start_s}, {end_s})")

        output_name = f"anon_4_4_hangzhou_real_t{start_s:04d}_{end_s:04d}.json"
        output_path = data_dir / output_name
        output_path.write_text(json.dumps(selected, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        outputs.append(
            {
                "name": output_name,
                "source_time_window_s": [start_s, end_s],
                "run_counts": args.slice_width_s,
                "flow_records": len(selected),
                "sha256": sha256(output_path),
            }
        )

    manifest = {
        "schema": "hangzhou_real_time_slices_v1",
        "semantics": (
            "non-overlapping start/end-time windows from the original CityFlow demand; "
            "start/end times are shifted to each 0-based evaluation horizon; no labels involved"
        ),
        "source": {"name": source.name, "sha256": sha256(source), "flow_records": len(flows)},
        "outputs": outputs,
    }
    manifest_path = data_dir / "hangzhou_real_time_slices_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
