import argparse
import hashlib
import json
import re
from collections import defaultdict
from typing import Iterable, List, Tuple


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    words = text.split()
    if not words:
        return []
    tokens = list(words)
    # Add bigrams for better sensitivity to ordering.
    tokens.extend([f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)])
    return tokens


def _simhash(tokens: Iterable[str]) -> int:
    # 64-bit simhash
    vec = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            if (h >> i) & 1:
                vec[i] += 1
            else:
                vec[i] -= 1
    out = 0
    for i, v in enumerate(vec):
        if v >= 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _band_keys(hash64: int, bands: int) -> List[Tuple[int, int]]:
    bands = max(1, int(bands))
    bits_per = 64 // bands
    keys = []
    for i in range(bands):
        start = i * bits_per
        end = 64 if i == bands - 1 else (i + 1) * bits_per
        mask = (1 << (end - start)) - 1
        val = (hash64 >> start) & mask
        keys.append((i, val))
    return keys


def dedup_jsonl(
    input_path: str,
    output_path: str,
    hamming_threshold: int = 3,
    bands: int = 4,
    keep_incomplete: bool = False,
) -> dict:
    seen_hashes: List[int] = []
    bucket = defaultdict(list)
    seen_exact = set()

    kept_groups = 0
    dropped_groups = 0
    total_groups = 0
    total_records = 0
    written_records = 0

    current_group = []

    def _is_similar(caption: str) -> bool:
        norm = _normalize_text(caption)
        if not norm:
            return False
        if norm in seen_exact:
            return True
        tokens = _tokenize(norm)
        if not tokens:
            return False
        h = _simhash(tokens)
        candidate_ids = set()
        for key in _band_keys(h, bands):
            candidate_ids.update(bucket.get(key, []))
        for idx in candidate_ids:
            if _hamming(h, seen_hashes[idx]) <= hamming_threshold:
                return True
        # Keep: register
        idx = len(seen_hashes)
        seen_hashes.append(h)
        seen_exact.add(norm)
        for key in _band_keys(h, bands):
            bucket[key].append(idx)
        return False

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            total_records += 1
            current_group.append(record)

            clip_caption = record.get("clip_caption") or ""
            if not clip_caption:
                continue

            total_groups += 1
            if _is_similar(str(clip_caption)):
                dropped_groups += 1
            else:
                kept_groups += 1
                for item in current_group:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written_records += 1
            current_group = []

        if current_group and keep_incomplete:
            for item in current_group:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                written_records += 1

    return {
        "total_groups": total_groups,
        "kept_groups": kept_groups,
        "dropped_groups": dropped_groups,
        "total_records": total_records,
        "written_records": written_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input jsonl path")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--hamming_threshold", type=int, default=3)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--keep_incomplete", action="store_true")
    args = parser.parse_args()

    stats = dedup_jsonl(
        input_path=args.input,
        output_path=args.output,
        hamming_threshold=args.hamming_threshold,
        bands=args.bands,
        keep_incomplete=args.keep_incomplete,
    )
    print(
        "done: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
    )


if __name__ == "__main__":
    main()
