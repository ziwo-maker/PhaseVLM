import argparse
import csv
import json
import re
import sys
from typing import Dict, List, TextIO


MOVEMENT_ORDER = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
_SECONDS_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
_INT_RE = re.compile(r"^-?\d+$")


def _open_output(path: str) -> TextIO:
    if path == "-":
        return sys.stdout
    return open(path, "w", encoding="utf-8", newline="")


def _parse_movement_counts(
    counts: Dict[str, object], line_no: int, accum: List[int]
) -> None:
    for idx, key in enumerate(MOVEMENT_ORDER):
        value = counts.get(key, 0)
        try:
            accum[idx] += int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric count for {key} at line {line_no}: {value!r}"
            ) from exc


def _infer_time_label(value: object) -> str:
    if isinstance(value, bool):
        return "time_key"
    if isinstance(value, (int, float)):
        return "time_s"
    if isinstance(value, str):
        if _SECONDS_RE.search(value):
            return "date"
        if _INT_RE.match(value):
            return "time_s"
        try:
            float(value)
            return "time_s"
        except ValueError:
            return "time_key"
    return "time_key"


def _coerce_time_value(value: object, label: str, line_no: int) -> object:
    if label == "date":
        if not isinstance(value, str) or not _SECONDS_RE.search(value):
            raise ValueError(
                f"Expected second-precision date string at line {line_no}: {value!r}"
            )
        return value
    if label == "time_s":
        if isinstance(value, bool):
            raise ValueError(f"Expected numeric time at line {line_no}: {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, str):
            stripped = value.strip()
            if _INT_RE.match(stripped):
                return int(stripped)
            try:
                numeric = float(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"Expected numeric time at line {line_no}: {value!r}"
                ) from exc
            return int(numeric) if numeric.is_integer() else numeric
        raise ValueError(f"Expected numeric time at line {line_no}: {value!r}")
    return str(value)


def _convert_sorted(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f, _open_output(output_path) as out:
        writer = csv.writer(out)

        time_label = None
        current_time = None
        prev_time = None
        per_intersection: Dict[str, List[int]] = {}

        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            raw_time = record.get("date")
            if raw_time is None or raw_time == "":
                raise ValueError(f"Missing date/time value at line {line_no}")
            if time_label is None:
                time_label = _infer_time_label(raw_time)
                writer.writerow([time_label, "intersection_id"] + MOVEMENT_ORDER)
            time_key = _coerce_time_value(raw_time, time_label, line_no)

            intersection_id = record.get("intersection_id")
            if not intersection_id:
                raise ValueError(f"Missing intersection_id at line {line_no}")

            if prev_time is not None and time_key < prev_time:
                raise ValueError(
                    "Input is not sorted by time key. Re-run with --sort to aggregate safely."
                )
            prev_time = time_key

            if current_time is None:
                current_time = time_key
            if time_key != current_time:
                for inter_id in sorted(per_intersection.keys()):
                    writer.writerow([current_time, inter_id] + per_intersection[inter_id])
                current_time = time_key
                per_intersection = {}

            movement_counts = record.get("movement_counts") or {}
            if intersection_id not in per_intersection:
                per_intersection[intersection_id] = [0] * len(MOVEMENT_ORDER)
            _parse_movement_counts(
                movement_counts, line_no, per_intersection[intersection_id]
            )

        if current_time is not None:
            for inter_id in sorted(per_intersection.keys()):
                writer.writerow([current_time, inter_id] + per_intersection[inter_id])


def _convert_unsorted(input_path: str, output_path: str) -> None:
    aggregated: Dict[object, Dict[str, List[int]]] = {}
    time_label = None
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            raw_time = record.get("date")
            if raw_time is None or raw_time == "":
                raise ValueError(f"Missing date/time value at line {line_no}")
            if time_label is None:
                time_label = _infer_time_label(raw_time)
            time_key = _coerce_time_value(raw_time, time_label, line_no)

            intersection_id = record.get("intersection_id")
            if not intersection_id:
                raise ValueError(f"Missing intersection_id at line {line_no}")

            if time_key not in aggregated:
                aggregated[time_key] = {}
            if intersection_id not in aggregated[time_key]:
                aggregated[time_key][intersection_id] = [0] * len(MOVEMENT_ORDER)
            movement_counts = record.get("movement_counts") or {}
            _parse_movement_counts(
                movement_counts, line_no, aggregated[time_key][intersection_id]
            )

    with _open_output(output_path) as out:
        writer = csv.writer(out)
        if time_label is None:
            return
        writer.writerow([time_label, "intersection_id"] + MOVEMENT_ORDER)
        for time_key in sorted(aggregated.keys()):
            for intersection_id in sorted(aggregated[time_key].keys()):
                writer.writerow(
                    [time_key, intersection_id] + aggregated[time_key][intersection_id]
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a traffic counts JSONL file to a CSV with a date column plus "
            "eight movement columns."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSONL path.",
        default="records/Random/anon_3_4_jinan_real.json01_19_03_22_21/traffic_counts_2s.jsonl",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output CSV path (use '-' for stdout).",
        default="records/Random/anon_3_4_jinan_real.json01_19_03_22_21/traffic_counts_2s.csv",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Allow unsorted input by aggregating then sorting all timestamps.",
        default=False
    )
    args = parser.parse_args()

    if args.sort:
        _convert_unsorted(args.input, args.output)
    else:
        _convert_sorted(args.input, args.output)


if __name__ == "__main__":
    main()
