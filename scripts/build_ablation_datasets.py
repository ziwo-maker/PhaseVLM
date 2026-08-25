#!/usr/bin/env python3
"""Build R1 (uncorrected) and R2 (corrected) datasets for A-override distillation ablation.

R1 (Baseline): Same data but expert_action = base_action at terminal rows (no override)
R2 (Distill):  Original data with expert_action = override-corrected action at terminal rows
"""

import json
from pathlib import Path

RULE_GAP2_DATA = "data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl"
OUT_DIR = Path("data/ablation/a_override_distill")
OUT_DIR.mkdir(parents=True, exist_ok=True)

r1_path = OUT_DIR / "R1_baseline_uncorrected.jsonl"
r2_path = OUT_DIR / "R2_distill_corrected.jsonl"

r1_rows = 0
r2_rows = 0
overridden_count = 0
total_terminal = 0

with open(RULE_GAP2_DATA) as f_in, \
     open(r1_path, "w") as f_r1, \
     open(r2_path, "w") as f_r2:

    for line in f_in:
        record = json.loads(line)
        mo = record.get("max_phase_override", {})
        base_action = record.get("base_action", "")

        # R2: Keep as-is (with corrections - expert_action already = final_action)
        f_r2.write(json.dumps(record, ensure_ascii=False) + "\n")
        r2_rows += 1

        # R1: For terminal rows, use base_action as expert_action
        if mo.get("enabled"):
            total_terminal += 1
            if mo.get("overridden"):
                overridden_count += 1
            record_r1 = dict(record)
            if base_action:
                record_r1["expert_action"] = base_action
            f_r1.write(json.dumps(record_r1, ensure_ascii=False) + "\n")
            r1_rows += 1
        else:
            # History row (no override info) - keep as-is for both
            f_r1.write(json.dumps(record, ensure_ascii=False) + "\n")
            r1_rows += 1

print(f"Total terminal rows: {total_terminal}")
print(f"Overridden rows corrected in R2 only: {overridden_count}")
print(f"R1 (uncorrected): {r1_rows} rows -> {r1_path}")
print(f"R2 (corrected):   {r2_rows} rows -> {r2_path}")

# Verify: both should have same row count
assert r1_rows == r2_rows, f"Row count mismatch: R1={r1_rows} vs R2={r2_rows}"

# Check action distributions
from collections import Counter
r1_actions = Counter()
r2_actions = Counter()
with open(r1_path) as f:
    for line in f:
        d = json.loads(line)
        if d.get("max_phase_override", {}).get("enabled"):
            r1_actions[d.get("expert_action", "?")] += 1
with open(r2_path) as f:
    for line in f:
        d = json.loads(line)
        if d.get("max_phase_override", {}).get("enabled"):
            r2_actions[d.get("expert_action", "?")] += 1
print(f"\nR1 action distribution (uncorrected): {dict(r1_actions)}")
print(f"R2 action distribution (corrected):   {dict(r2_actions)}")

summary = {
    "source": RULE_GAP2_DATA,
    "total_rows": r1_rows,
    "total_terminal_rows": total_terminal,
    "overridden_rows": overridden_count,
    "r1_path": str(r1_path),
    "r2_path": str(r2_path),
    "r1_action_dist": dict(r1_actions),
    "r2_action_dist": dict(r2_actions),
    "r1_description": "Baseline: expert_action = base_action (model's original action, no override)",
    "r2_description": "Distill: expert_action = override-corrected action (A-override gap=2 protect=60)",
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary: {OUT_DIR / 'summary.json'}")
