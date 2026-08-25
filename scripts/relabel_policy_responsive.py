#!/usr/bin/env python3
"""Create policy-responsive counterfactual labels for the Phase-7 teacher study.

Unlike ``relabel_simexpert.py``, candidate branches use the frozen source policy
after the first forced target action.  The output has the standard SimExpert
schema and can be passed to the existing BC/Phase-7 runners.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from projection_train.policy_responsive_simexpert import add_common_args, relabel_from_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy-responsive SimExpert counterfactual relabeling"
    )
    add_common_args(parser)
    relabel_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
