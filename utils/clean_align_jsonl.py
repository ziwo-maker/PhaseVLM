import argparse
import json
import os
from typing import Iterable, List, Optional


DEFAULT_MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]


def _parse_movement_keys(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_MOVEMENT_KEYS)
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys if keys else list(DEFAULT_MOVEMENT_KEYS)


def _sequence_total(
    counts_history: List[dict],
    seq_len: int,
    movement_keys: Iterable[str],
) -> float:
    if not counts_history:
        return 0.0
    window = counts_history[-seq_len:]
    if len(window) < seq_len:
        pad = window[0] if window else {}
        window = [pad] * (seq_len - len(window)) + window
    total = 0.0
    for counts in window:
        total += sum(float(counts.get(k, 0.0) or 0.0) for k in movement_keys)
    return total


def _should_keep(
    total: float,
    drop_zero_seq: bool,
    min_total: Optional[float],
    max_total: Optional[float],
) -> bool:
    if drop_zero_seq and total == 0.0:
        return False
    if min_total is not None and total < min_total:
        return False
    if max_total is not None and total > max_total:
        return False
    return True


def clean_jsonl(
    inputs: List[str],
    output_path: str,
    seq_len: int = 16,
    movement_keys: Optional[List[str]] = None,
    drop_zero_seq: bool = True,
    min_total: Optional[float] = None,
    max_total: Optional[float] = None,
    keep_incomplete: bool = False,
) -> dict:
    movement_keys = movement_keys or list(DEFAULT_MOVEMENT_KEYS)

    stats = {
        "total_lines": 0,
        "bad_lines": 0,
        "total_groups": 0,
        "kept_groups": 0,
        "dropped_zero_groups": 0,
        "dropped_range_groups": 0,
        "written_lines": 0,
        "incomplete_groups_dropped": 0,
        "incomplete_groups_kept": 0,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for src in inputs:
            current_group: List[dict] = []
            counts_history: List[dict] = []
            with open(src, "r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    stats["total_lines"] += 1
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_lines"] += 1
                        continue
                    if not isinstance(record, dict):
                        stats["bad_lines"] += 1
                        continue

                    current_group.append(record)
                    counts_history.append(record.get("movement_counts") or {})
                    clip_caption = record.get("clip_caption") or ""
                    if not clip_caption:
                        continue

                    stats["total_groups"] += 1
                    total = _sequence_total(counts_history, seq_len, movement_keys)
                    keep = _should_keep(total, drop_zero_seq, min_total, max_total)
                    if keep:
                        for item in current_group:
                            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stats["written_lines"] += 1
                        stats["kept_groups"] += 1
                    else:
                        if drop_zero_seq and total == 0.0:
                            stats["dropped_zero_groups"] += 1
                        else:
                            stats["dropped_range_groups"] += 1

                    current_group = []
                    counts_history = []

            if current_group:
                if keep_incomplete:
                    total = _sequence_total(counts_history, seq_len, movement_keys)
                    keep = _should_keep(total, drop_zero_seq, min_total, max_total)
                    if keep:
                        for item in current_group:
                            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stats["written_lines"] += 1
                        stats["incomplete_groups_kept"] += 1
                    else:
                        stats["incomplete_groups_dropped"] += 1
                else:
                    stats["incomplete_groups_dropped"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="input jsonl paths")
    parser.add_argument(
        "--output",
        default="data/recorded/align_train.cleaned.jsonl",
        help="output jsonl path",
    )
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument(
        "--movement_keys",
        type=str,
        default=",".join(DEFAULT_MOVEMENT_KEYS),
        help="comma-separated movement keys",
    )
    parser.add_argument("--drop_zero_seq", action="store_true", default=True)
    parser.add_argument("--min_total", type=float, default=None)
    parser.add_argument("--max_total", type=float, default=None)
    parser.add_argument("--keep_incomplete", action="store_true")
    args = parser.parse_args()

    stats = clean_jsonl(
        inputs=args.inputs,
        output_path=args.output,
        seq_len=args.seq_len,
        movement_keys=_parse_movement_keys(args.movement_keys),
        drop_zero_seq=args.drop_zero_seq,
        min_total=args.min_total,
        max_total=args.max_total,
        keep_incomplete=args.keep_incomplete,
    )

    print("done:")
    for key in (
        "total_lines",
        "bad_lines",
        "total_groups",
        "kept_groups",
        "dropped_zero_groups",
        "dropped_range_groups",
        "written_lines",
        "incomplete_groups_dropped",
        "incomplete_groups_kept",
    ):
        print(f"  {key}={stats[key]}")


if __name__ == "__main__":
    main()
