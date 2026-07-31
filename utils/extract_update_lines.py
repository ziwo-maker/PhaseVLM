import argparse
import re
import sys
from typing import TextIO


UPDATE_TAG_RE = re.compile(r"\[update (100|[1-9]?\d)/100\]")


def _open_output(path: str) -> TextIO:
    if path == "-":
        return sys.stdout
    return open(path, "w", encoding="utf-8")


def extract_update_lines(input_path: str, output_path: str, only_tag: bool) -> int:
    matched = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        with _open_output(output_path) as fout:
            for line in fin:
                match = UPDATE_TAG_RE.search(line)
                if not match:
                    continue
                matched += 1
                fout.write((match.group(0) if only_tag else line.rstrip("\n")) + "\n")
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract lines containing tags like "[update X/100]" from a .out file.'
    )
    parser.add_argument("input", help="Input .out file path")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help='Output path, use "-" for stdout (default: "-")',
    )
    parser.add_argument(
        "--only-tag",
        action="store_true",
        help='Only output the matched "[update X/100]" part instead of the full line',
    )
    args = parser.parse_args()

    matched = extract_update_lines(args.input, args.output, args.only_tag)
    print(f"matched_lines={matched}", file=sys.stderr)


if __name__ == "__main__":
    main()
