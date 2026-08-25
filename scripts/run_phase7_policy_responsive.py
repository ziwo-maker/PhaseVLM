#!/usr/bin/env python3
"""Train/evaluate a single responsive teacher using the canonical Phase-7 runner.

The existing orchestrator's ``c30`` variant only consumes its 30-second input.
Passing the responsive label file for all three horizon arguments is therefore
intentional: it reuses the established split, training, checkpoint-selection,
and online-evaluation protocol without pretending the labels are frozen CRCD
horizons.  The output directory and config record the responsive provenance.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-7 policy-responsive teacher experiment")
    parser.add_argument("--labels", required=True, help="Responsive-label JSONL at one selected horizon")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_budgets", type=int, nargs="+", default=[8, 32, 118])
    parser.add_argument("--selection_seed", type=int, default=2024)
    parser.add_argument("--training_seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--train_gpus", default="0,1")
    parser.add_argument("--eval_gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--loss", choices=["mixed", "hard"], default="mixed")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dataset", default=None,
                        help="Target dataset name for online eval (defaults to the runner's hangzhou default).")
    parser.add_argument("--test_flow", default=None,
                        help="Target test traffic flow for online eval.")
    args = parser.parse_args()

    label_path = os.path.abspath(args.labels)
    if not os.path.isfile(label_path):
        raise FileNotFoundError(label_path)
    command = [
        sys.executable,
        os.path.join(_REPO, "scripts", "run_phase7_crcd.py"),
        "--f30", label_path,
        "--f120", label_path,
        "--f300", label_path,
        "--output_dir", os.path.abspath(args.output_dir),
        "--teachers", "c30",
        "--label_budgets", *[str(value) for value in args.label_budgets],
        "--selection_seed", str(args.selection_seed),
        "--training_seeds", *[str(value) for value in args.training_seeds],
        "--eval_seeds", *[str(value) for value in args.eval_seeds],
        "--train_gpus", args.train_gpus,
        "--eval_gpus", *[str(value) for value in args.eval_gpus],
        "--loss", args.loss,
    ]
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.test_flow:
        command.extend(["--test_flow", args.test_flow])
    if args.only:
        # Canonical tags remain c30:<budget>:<training-seed> for compatibility.
        command.extend(["--only", *args.only])
    print("[Phase7Responsive] executing canonical runner", flush=True)
    completed = subprocess.run(command, cwd=_REPO, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
