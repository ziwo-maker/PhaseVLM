#!/usr/bin/env python3
"""Build all ablation variant datasets. Only modify expert_action on TERMINAL rows."""
import json
from pathlib import Path
from collections import Counter

BASE = "data/processed/hangzhou_real_stage3_queue_wait_rule_gap2_p60_20260528.jsonl"
OUT = Path("data/ablation/a_override_distill")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
with open(BASE) as f:
    for line in f:
        rows.append(json.loads(line))

def build_variant(name, override_filter_fn):
    out_path = OUT / f"{name}.jsonl"
    stats = {"total": 0, "terminal": 0, "overridden": 0, "non_terminal": 0}
    actions = Counter()
    with open(out_path, "w") as f:
        for record in rows:
            mo = record.get("max_phase_override", {})
            record_out = dict(record)
            if mo.get("enabled"):
                # TERMINAL row: apply the override filter
                stats["terminal"] += 1
                base_action = record.get("base_action", "")
                final_action = override_filter_fn(mo, base_action)
                record_out["expert_action"] = final_action
                if final_action != base_action:
                    stats["overridden"] += 1
                actions[final_action] += 1
            else:
                # HISTORY row: keep as-is (expert_action is already empty/correct)
                stats["non_terminal"] += 1
            f.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            stats["total"] += 1
    return out_path, stats, actions

def filter_gap(target_gap):
    def fn(mo, base_action):
        gap = mo.get("gap", 0)
        target = mo.get("target_phase", base_action)
        if gap >= target_gap and target != base_action:
            return target
        return base_action
    return fn

def filter_protect(protect_threshold):
    def fn(mo, base_action):
        gap = mo.get("gap", 0)
        target = mo.get("target_phase", base_action)
        original_action = mo.get("original_action", base_action)
        phase_wait = mo.get("phase_wait", {})
        left_phases = {"ELWL", "NLSL"}
        through_phases = {"ETWT", "NTST"}
        if gap >= 2 and target != base_action:
            if protect_threshold > 0 and original_action in left_phases and target in through_phases:
                ow = float(phase_wait.get(original_action, 0))
                if ow >= protect_threshold:
                    return base_action
            return target
        return base_action
    return fn

print("Building variant datasets...\n")

variants = [
    ("A4_protect0", filter_protect(0)),
    ("B1_gap3", filter_gap(3)),
    ("B1_gap5", filter_gap(5)),
    ("B2_protect30", filter_protect(30)),
    ("B2_protect45", filter_protect(45)),
]

for name, fn in variants:
    path, stats, actions = build_variant(name, fn)
    print(f"{name}: terminal={stats['terminal']} overridden={stats['overridden']} "
          f"total={stats['total']} actions={dict(actions)}")

print(f"\nAll variant datasets in: {OUT}")
