#!/usr/bin/env python3
"""Create adjacent-intersection joint counterfactual SimExpert labels.

Each selected adjacent edge evaluates all 16 joint action pairs from a common
snapshot.  Per-intersection labels are min-marginals of that 4x4 edge table;
the full table is kept in the JSONL provenance.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from projection_train.joint_counterfactual_simexpert import (  # noqa: E402
    add_joint_args,
    relabel_joint_from_args,
)
from projection_train.policy_responsive_simexpert import add_common_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjacent-intersection joint counterfactual SimExpert relabeling"
    )
    add_common_args(parser)
    add_joint_args(parser)
    relabel_joint_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
