import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple


def _detect_time_keys(item: Dict[str, Any]) -> Tuple[str, str]:
    if "startTime" in item:
        return "startTime", "endTime" if "endTime" in item else ""
    if "start_time" in item:
        return "start_time", "end_time" if "end_time" in item else ""
    raise KeyError("Missing startTime/start_time in data items.")


def _load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list at the top level.")
    return data


def _maybe_cast_times(values: List[float], prefer_int: bool) -> List[Any]:
    if prefer_int:
        return [int(round(v)) for v in values]
    return values


def _compute_interval(
    count: int, target_interval: float, total_duration: Optional[float]
) -> float:
    if total_duration is None:
        return target_interval
    if count <= 1:
        return 0.0
    return float(total_duration) / float(count - 1)


def _make_new_starts(
    count: int,
    base_time: float,
    interval: float,
    total_duration: Optional[float],
    jitter: float,
    seed: Optional[int],
) -> List[float]:
    new_starts = [base_time + i * interval for i in range(count)]
    if jitter and jitter > 0:
        rnd = random.Random(seed)
        max_time = base_time + (total_duration if total_duration is not None else interval * (count - 1))
        jittered = []
        for t in new_starts:
            jt = t + rnd.uniform(-jitter, jitter)
            if jt < base_time:
                jt = base_time
            elif jt > max_time:
                jt = max_time
            jittered.append(jt)
        jittered.sort()
        new_starts = jittered
    return new_starts


def _normalize_start_times(
    data: List[Dict[str, Any]],
    target_interval: float,
    base_time: float,
    total_duration: Optional[float],
    jitter: float,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    if not data:
        return data

    start_key, end_key = _detect_time_keys(data[0])
    sorted_data = sorted(data, key=lambda x: x.get(start_key, 0))

    starts = [item[start_key] for item in sorted_data]
    all_int = all(isinstance(v, int) for v in starts)
    base_is_int = isinstance(base_time, int) or float(base_time).is_integer()
    interval = _compute_interval(len(sorted_data), target_interval, total_duration)
    interval_is_int = float(interval).is_integer()
    prefer_int = all_int and base_is_int and interval_is_int and (not jitter or jitter <= 0)

    new_starts = _make_new_starts(
        len(sorted_data), base_time, interval, total_duration, jitter, seed
    )
    new_starts = _maybe_cast_times(new_starts, prefer_int)

    for item, new_start in zip(sorted_data, new_starts):
        old_start = item[start_key]
        item[start_key] = new_start
        if end_key:
            old_end = item[end_key]
            duration = old_end - old_start
            new_end = new_start + duration
            item[end_key] = int(round(new_end)) if prefer_int else new_end

    return sorted_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize start times to a fixed interval or a total duration window."
    )
    parser.add_argument("input", help="Input JSON path.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path. Defaults to input path with a suffix based on the mode.",
    )
    parser.add_argument(
        "--target-interval",
        type=float,
        default=2.0,
        help="Target interval in seconds between consecutive items.",
    )
    parser.add_argument(
        "--total-duration",
        type=float,
        default=None,
        help="If set, map all items into [base_time, base_time + total_duration].",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Uniform random jitter in seconds (applied then sorted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for jitter.",
    )
    parser.add_argument(
        "--base-time",
        type=float,
        default=None,
        help="Base start time for the first item. Defaults to min start time.",
    )
    args = parser.parse_args()

    data = _load_data(args.input)
    if not data:
        raise ValueError("Input JSON is empty.")

    start_key, _ = _detect_time_keys(data[0])
    if args.base_time is None:
        base_time = min(item[start_key] for item in data)
    else:
        base_time = args.base_time

    normalized = _normalize_start_times(
        data,
        args.target_interval,
        base_time,
        args.total_duration,
        args.jitter,
        args.seed,
    )

    output = args.output
    if not output:
        if args.input.endswith(".json"):
            base = args.input[:-5]
        else:
            base = args.input
        if args.total_duration is not None:
            if float(args.total_duration).is_integer():
                suffix = f"_{int(args.total_duration)}s"
            else:
                suffix = "_compressed"
        elif float(args.target_interval).is_integer():
            suffix = f"_{int(args.target_interval)}s"
        else:
            suffix = "_normalized"
        output = base + suffix + ".json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)

    print(f"Wrote {len(normalized)} items to {output}")


if __name__ == "__main__":
    main()
