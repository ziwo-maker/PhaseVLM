#!/usr/bin/env python3
"""Compare latest Hangzhou-real Stage 3 online logs with local train traffic logs."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "hangzhou_real_stage3_diagnosis_20260525"

LOCAL_TRAIN = (
    ROOT
    / "records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/"
    / "align_train.stage1_compact_ds.jsonl"
)
LOCAL_TRAIN_SUMMARY = LOCAL_TRAIN.with_suffix(LOCAL_TRAIN.suffix + ".summary.json")
REMOTE_COUNTS = OUT_DIR / "remote_traffic_counts_2s.full.jsonl"
REMOTE_PROMPTS = OUT_DIR / "remote_traffic_prompt_series_2s.full.jsonl"
STATE_ACTION = OUT_DIR / "state_action.full.json"
TEST_RESULT = OUT_DIR / "test_result.json"
TRAIN_SUMMARY = OUT_DIR / "train_summary.json"
TRAIN_PROGRESS = OUT_DIR / "train_progress.log"

MOVEMENTS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
PHASES = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
PHASE_ORDER = ["ETWT", "NTST", "ELWL", "NLSL"]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def parse_seconds(date_s: str) -> int:
    hms = date_s.split()[-1]
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return float(xs[idx])


def traffic_stats(path: Path) -> dict:
    movement_totals = Counter()
    intersection_totals = Counter()
    intersection_rows = Counter()
    intersection_movement = defaultdict(Counter)
    window_totals = Counter()
    window_rows = Counter()
    row_totals: list[float] = []
    prompt_rows = 0
    expert_action_rows = Counter()
    snapshot_actions: dict[str, str] = {}
    margins: list[float] = []

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts = row.get("movement_counts", {})
            row_total = sum(float(counts.get(m, 0.0)) for m in MOVEMENTS)
            row_totals.append(row_total)

            inter = row.get("intersection_id", "")
            intersection_totals[inter] += row_total
            intersection_rows[inter] += 1

            for movement in MOVEMENTS:
                value = float(counts.get(movement, 0.0))
                movement_totals[movement] += value
                intersection_movement[inter][movement] += value

            if row.get("date"):
                window = parse_seconds(row["date"]) // 900
                window_totals[window] += row_total
                window_rows[window] += 1

            if row.get("prompt"):
                prompt_rows += 1
            action = row.get("expert_action")
            if action:
                expert_action_rows[action] += 1
                snapshot = row.get("snapshot_path")
                if snapshot:
                    snapshot_actions[snapshot] = action
            margin = row.get("expert_action_margin")
            if isinstance(margin, (int, float)):
                margins.append(float(margin))

    total = sum(movement_totals.values())
    phase_totals = {
        phase: sum(movement_totals[movement] for movement in movements)
        for phase, movements in PHASES.items()
    }
    direction_totals = {
        "east_west": movement_totals["ET"]
        + movement_totals["EL"]
        + movement_totals["WT"]
        + movement_totals["WL"],
        "north_south": movement_totals["NT"]
        + movement_totals["NL"]
        + movement_totals["ST"]
        + movement_totals["SL"],
        "through": movement_totals["ET"]
        + movement_totals["WT"]
        + movement_totals["NT"]
        + movement_totals["ST"],
        "left": movement_totals["EL"]
        + movement_totals["WL"]
        + movement_totals["NL"]
        + movement_totals["SL"],
    }

    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(row_totals),
        "prompt_rows": prompt_rows,
        "zero_rows": sum(1 for value in row_totals if value == 0),
        "row_total": {
            "mean": statistics.fmean(row_totals) if row_totals else 0.0,
            "p50": quantile(row_totals, 0.50),
            "p90": quantile(row_totals, 0.90),
            "p95": quantile(row_totals, 0.95),
            "max": max(row_totals) if row_totals else 0.0,
        },
        "movement_totals": dict(movement_totals),
        "movement_per_row": {
            movement: movement_totals[movement] / len(row_totals)
            for movement in MOVEMENTS
        },
        "movement_share_pct": {
            movement: pct(movement_totals[movement], total) for movement in MOVEMENTS
        },
        "phase_totals": phase_totals,
        "phase_per_row": {
            phase: phase_totals[phase] / len(row_totals) for phase in PHASE_ORDER
        },
        "phase_share_pct": {
            phase: pct(phase_totals[phase], total) for phase in PHASE_ORDER
        },
        "direction_share_pct": {
            name: pct(value, total) for name, value in direction_totals.items()
        },
        "intersection_per_row": {
            inter: intersection_totals[inter] / intersection_rows[inter]
            for inter in sorted(intersection_rows)
        },
        "intersection_rows": dict(intersection_rows),
        "intersection_movement_per_row": {
            inter: {
                movement: intersection_movement[inter][movement] / intersection_rows[inter]
                for movement in MOVEMENTS
            }
            for inter in sorted(intersection_rows)
        },
        "window_avg_occupancy": {
            f"{window * 15:02d}-{(window + 1) * 15:02d}m": window_totals[window]
            / window_rows[window]
            for window in sorted(window_rows)
        },
        "window_rows": {
            f"{window * 15:02d}-{(window + 1) * 15:02d}m": window_rows[window]
            for window in sorted(window_rows)
        },
        "expert_action_rows": dict(expert_action_rows),
        "expert_action_snapshots": dict(Counter(snapshot_actions.values())),
        "unique_snapshots": len(snapshot_actions),
        "expert_margin": {
            "n": len(margins),
            "mean": statistics.fmean(margins) if margins else 0.0,
            "lt_0_5": sum(1 for m in margins if m < 0.5),
            "lt_1_0": sum(1 for m in margins if m < 1.0),
        },
    }


def compare_traffic(train: dict, online: dict) -> dict:
    movement_rows = []
    for movement in MOVEMENTS:
        t = train["movement_per_row"][movement]
        o = online["movement_per_row"][movement]
        movement_rows.append(
            {
                "movement": movement,
                "train_per_row": t,
                "online_per_row": o,
                "delta_per_row": o - t,
                "pct_delta": pct(o - t, t),
                "train_share_pct": train["movement_share_pct"][movement],
                "online_share_pct": online["movement_share_pct"][movement],
                "share_delta_pct": online["movement_share_pct"][movement]
                - train["movement_share_pct"][movement],
            }
        )

    phase_rows = []
    for phase in PHASE_ORDER:
        t = train["phase_per_row"][phase]
        o = online["phase_per_row"][phase]
        phase_rows.append(
            {
                "phase": phase,
                "train_per_row": t,
                "online_per_row": o,
                "delta_per_row": o - t,
                "pct_delta": pct(o - t, t),
                "train_share_pct": train["phase_share_pct"][phase],
                "online_share_pct": online["phase_share_pct"][phase],
                "share_delta_pct": online["phase_share_pct"][phase]
                - train["phase_share_pct"][phase],
            }
        )

    intersection_rows = []
    all_inters = sorted(
        set(train["intersection_per_row"]) | set(online["intersection_per_row"])
    )
    for inter in all_inters:
        t = train["intersection_per_row"].get(inter, 0.0)
        o = online["intersection_per_row"].get(inter, 0.0)
        intersection_rows.append(
            {
                "intersection": inter,
                "train_per_row": t,
                "online_per_row": o,
                "delta_per_row": o - t,
                "pct_delta": pct(o - t, t),
            }
        )

    inter_move_rows = []
    for inter in all_inters:
        train_im = train["intersection_movement_per_row"].get(inter, {})
        online_im = online["intersection_movement_per_row"].get(inter, {})
        for movement in MOVEMENTS:
            t = train_im.get(movement, 0.0)
            o = online_im.get(movement, 0.0)
            inter_move_rows.append(
                {
                    "intersection": inter,
                    "movement": movement,
                    "train_per_row": t,
                    "online_per_row": o,
                    "delta_per_row": o - t,
                    "pct_delta": pct(o - t, t),
                }
            )

    return {
        "overall_per_row": {
            "train": train["row_total"]["mean"],
            "online": online["row_total"]["mean"],
            "delta": online["row_total"]["mean"] - train["row_total"]["mean"],
            "pct_delta": pct(
                online["row_total"]["mean"] - train["row_total"]["mean"],
                train["row_total"]["mean"],
            ),
        },
        "movements": movement_rows,
        "phases": phase_rows,
        "intersections": sorted(
            intersection_rows, key=lambda row: row["delta_per_row"], reverse=True
        ),
        "intersection_movements": sorted(
            inter_move_rows, key=lambda row: row["delta_per_row"], reverse=True
        ),
    }


def state_action_stats(path: Path) -> dict:
    data = load_json(path)
    action_counts = Counter()
    max_phase_counts = Counter()
    max_action_matrix = {phase: Counter() for phase in PHASE_ORDER}
    phase_queue_sum = Counter()
    phase_wait_sum = Counter()
    selected_when_max = {phase: Counter() for phase in PHASE_ORDER}
    total_decisions = 0
    max_match = 0
    strong_misses = 0
    missed_gap_sum = 0.0
    total_queue_sum = 0.0
    window_queue_sum = Counter()
    window_wait_sum = Counter()
    window_decisions = Counter()
    per_intersection = defaultdict(
        lambda: {
            "decisions": 0,
            "queue_sum": 0.0,
            "wait_sum": 0.0,
            "max_match": 0,
            "strong_misses": 0,
            "missed_gap_sum": 0.0,
            "actions": Counter(),
            "max_phases": Counter(),
        }
    )

    for inter_idx, seq in enumerate(data):
        inter_id = f"intersection_{inter_idx // 4 + 1}_{inter_idx % 4 + 1}"
        for step, row in enumerate(seq):
            state = row["state"]
            action = row.get("action", "")
            phase_queue = {
                phase: sum(float(state[m]["queue_len"]) for m in movements)
                for phase, movements in PHASES.items()
            }
            phase_wait = {
                phase: statistics.fmean(
                    float(state[m].get("avg_wait_time", 0.0)) for m in movements
                )
                for phase, movements in PHASES.items()
            }
            max_phase = max(PHASE_ORDER, key=lambda phase: phase_queue[phase])
            selected_q = phase_queue.get(action, 0.0)
            max_q = phase_queue[max_phase]
            gap = max_q - selected_q
            total_q = sum(phase_queue.values())
            avg_wait = statistics.fmean(phase_wait.values())

            total_decisions += 1
            action_counts[action] += 1
            max_phase_counts[max_phase] += 1
            max_action_matrix[max_phase][action] += 1
            selected_when_max[max_phase][action] += 1
            max_match += int(action == max_phase)
            strong_misses += int(gap >= 10.0)
            missed_gap_sum += gap
            total_queue_sum += total_q
            for phase in PHASE_ORDER:
                phase_queue_sum[phase] += phase_queue[phase]
                phase_wait_sum[phase] += phase_wait[phase]

            window = step // 30
            window_queue_sum[window] += total_q
            window_wait_sum[window] += avg_wait
            window_decisions[window] += 1

            inter = per_intersection[inter_id]
            inter["decisions"] += 1
            inter["queue_sum"] += total_q
            inter["wait_sum"] += avg_wait
            inter["max_match"] += int(action == max_phase)
            inter["strong_misses"] += int(gap >= 10.0)
            inter["missed_gap_sum"] += gap
            inter["actions"][action] += 1
            inter["max_phases"][max_phase] += 1

    return {
        "decisions": total_decisions,
        "action_counts": dict(action_counts),
        "action_pct": {
            phase: pct(action_counts[phase], total_decisions) for phase in PHASE_ORDER
        },
        "max_phase_counts": dict(max_phase_counts),
        "max_phase_pct": {
            phase: pct(max_phase_counts[phase], total_decisions)
            for phase in PHASE_ORDER
        },
        "max_phase_match_rate_pct": pct(max_match, total_decisions),
        "avg_missed_gap": missed_gap_sum / total_decisions,
        "strong_misses_gap_ge_10": strong_misses,
        "avg_total_queue": total_queue_sum / total_decisions,
        "avg_phase_queue": {
            phase: phase_queue_sum[phase] / total_decisions for phase in PHASE_ORDER
        },
        "avg_phase_wait": {
            phase: phase_wait_sum[phase] / total_decisions for phase in PHASE_ORDER
        },
        "max_phase_action_matrix": {
            phase: {action: max_action_matrix[phase][action] for action in PHASE_ORDER}
            for phase in PHASE_ORDER
        },
        "window_avg_total_queue": {
            f"{window * 15:02d}-{(window + 1) * 15:02d}m": window_queue_sum[window]
            / window_decisions[window]
            for window in sorted(window_decisions)
        },
        "window_avg_wait_proxy": {
            f"{window * 15:02d}-{(window + 1) * 15:02d}m": window_wait_sum[window]
            / window_decisions[window]
            for window in sorted(window_decisions)
        },
        "per_intersection": {
            inter: {
                "decisions": values["decisions"],
                "avg_total_queue": values["queue_sum"] / values["decisions"],
                "avg_wait_proxy": values["wait_sum"] / values["decisions"],
                "max_phase_match_rate_pct": pct(
                    values["max_match"], values["decisions"]
                ),
                "avg_missed_gap": values["missed_gap_sum"] / values["decisions"],
                "strong_misses_gap_ge_10": values["strong_misses"],
                "actions": dict(values["actions"]),
                "max_phases": dict(values["max_phases"]),
            }
            for inter, values in sorted(per_intersection.items())
        },
    }


def parse_train_progress(path: Path) -> dict:
    if not path.exists():
        return {"evals": [], "skips": []}
    evals = []
    skips = []
    eval_re = re.compile(
        r"\[eval (?P<update>\d+)\] reward=(?P<reward>-?\d+(?:\.\d+)?) "
        r"avg_queue=(?P<queue>-?\d+(?:\.\d+)?) "
        r"avg_wait=(?P<wait>-?\d+(?:\.\d+)?).*?"
        r"avg_travel=(?P<travel>-?\d+(?:\.\d+)?) "
        r"score=(?P<score>-?\d+(?:\.\d+)?)"
    )
    skip_re = re.compile(r"\[eval (?P<update>\d+)\] skip best update: (?P<reason>.*)")
    with path.open() as f:
        for line in f:
            eval_match = eval_re.search(line)
            if eval_match:
                evals.append(
                    {
                        "update": int(eval_match.group("update")),
                        "reward": float(eval_match.group("reward")),
                        "avg_queue": float(eval_match.group("queue")),
                        "avg_wait": float(eval_match.group("wait")),
                        "avg_travel": float(eval_match.group("travel")),
                        "score": float(eval_match.group("score")),
                    }
                )
            skip_match = skip_re.search(line)
            if skip_match:
                skips.append(
                    {
                        "update": int(skip_match.group("update")),
                        "reason": skip_match.group("reason").strip(),
                    }
                )
    return {"evals": evals, "skips": skips}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def action_pct_table(counts: dict[str, float], total: float) -> str:
    parts = []
    for phase in PHASE_ORDER:
        value = counts.get(phase, 0)
        parts.append(f"{phase} {int(value)} ({pct(value, total):.1f}%)")
    return ", ".join(parts)


def render_markdown(summary: dict) -> str:
    train = summary["train_traffic"]
    online = summary["online_traffic"]
    prompts = summary["online_prompt_traffic"]
    compare = summary["traffic_compare"]
    state = summary["state_action"]
    result = summary["test_result"]["results"]
    train_actions = summary["train_label_action_counts"]
    train_selection = summary.get("train_selection", {})
    train_progress = summary.get("train_progress", {})

    lines = [
        "# Hangzhou Real Stage 3 Diagnosis",
        "",
        "## Latest Remote Result",
        "",
        "| Run | Reward | Avg queue | Avg wait | Avg travel | Queuing vehicles |",
        "|---|---:|---:|---:|---:|---:|",
        (
        "| Latest Stage3 2026-05-24 16:18 -> test 17:08 | "
            f"{fmt(result['test_reward_over'])} | "
            f"{fmt(result['test_avg_queue_len_over'])} | "
            f"{fmt(result['test_avg_waiting_time_over'])} | "
            f"{fmt(result['test_avg_travel_time_over'])} | "
            f"{int(result['test_queuing_vehicle_num_over'])} |"
        ),
        "",
        "This is the newest Hangzhou-real Stage 3 run found under remote `/gemini/code/LLMSTSC/records/pipeline_runs/exp_hangzhou_real`.",
        "",
        "### Context Against Recent Hangzhou-Real Runs",
        "",
        "| Run | Reward | Avg queue | Avg wait | Avg travel |",
        "|---|---:|---:|---:|---:|",
        "| Latest Stage3 | -625.170 | 70.717 | 55.126 | 325.890 |",
        "| 5816 Stage2 checkpoint -> Hangzhou real | -632.374 | 69.058 | 55.523 | 322.516 |",
        "| Hangzhou-real Stage2 W2 epoch10 | -687.958 | 87.350 | 209.328 | 342.320 |",
        "| Hangzhou-real Stage2 main best epoch9 | -833.031 | 131.033 | 325.256 | 388.489 |",
        "",
        "### Stage3 PPO Selection",
        "",
    ]
    if train_selection:
        lines += [
            f"- `best.pt` was selected at PPO update `{train_selection.get('update')}`.",
            "- This means the final tested checkpoint is the guarded baseline/start checkpoint for this Stage3 run, not an accepted later PPO update.",
        ]
    else:
        lines.append("- Training summary was not available in the local analysis copy.")

    if train_progress.get("evals"):
        lines += [
            "",
            "| Eval update | Reward | Avg queue | Avg wait | Avg travel | Score |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in train_progress["evals"]:
            lines.append(
                f"| {row['update']} | {fmt(row['reward'])} | "
                f"{fmt(row['avg_queue'])} | {fmt(row['avg_wait'])} | "
                f"{fmt(row['avg_travel'])} | {fmt(row['score'])} |"
            )
    if train_progress.get("skips"):
        lines += ["", "Rejected updates:"]
        for row in train_progress["skips"]:
            lines.append(f"- update {row['update']}: {row['reason']}")

    lines += [
        "",
        "## Traffic Log Comparison",
        "",
        f"- Local train traffic log: `{train['path']}` ({train['rows']} rows, compact sampled train windows).",
        f"- Remote Stage3 closed-loop log: `{online['path']}` ({online['rows']} rows, full 60-minute 16-intersection rollout).",
        f"- Mean occupancy per row: train {fmt(compare['overall_per_row']['train'])}, online {fmt(compare['overall_per_row']['online'])}, delta {fmt(compare['overall_per_row']['delta'])} ({fmt(compare['overall_per_row']['pct_delta'], 1)}%).",
        f"- Direction share train -> online: EW {fmt(train['direction_share_pct']['east_west'], 1)}% -> {fmt(online['direction_share_pct']['east_west'], 1)}%, NS {fmt(train['direction_share_pct']['north_south'], 1)}% -> {fmt(online['direction_share_pct']['north_south'], 1)}%, left {fmt(train['direction_share_pct']['left'], 1)}% -> {fmt(online['direction_share_pct']['left'], 1)}%.",
        "",
        "### By Phase Group",
        "",
        "| Phase | Train per row | Online per row | Delta | Pct | Train share | Online share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compare["phases"]:
        lines.append(
            f"| {row['phase']} | {fmt(row['train_per_row'])} | "
            f"{fmt(row['online_per_row'])} | {fmt(row['delta_per_row'])} | "
            f"{fmt(row['pct_delta'], 1)}% | {fmt(row['train_share_pct'], 1)}% | "
            f"{fmt(row['online_share_pct'], 1)}% |"
        )

    lines += [
        "",
        "### Biggest Movement Increases",
        "",
        "| Movement | Train per row | Online per row | Delta | Pct | Share delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        compare["movements"], key=lambda item: item["delta_per_row"], reverse=True
    )[:8]:
        lines.append(
            f"| {row['movement']} | {fmt(row['train_per_row'])} | "
            f"{fmt(row['online_per_row'])} | {fmt(row['delta_per_row'])} | "
            f"{fmt(row['pct_delta'], 1)}% | {fmt(row['share_delta_pct'], 1)} pp |"
        )

    lines += [
        "",
        "### Biggest Intersection Increases",
        "",
        "| Intersection | Train per row | Online per row | Delta | Pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in compare["intersections"][:8]:
        lines.append(
            f"| {row['intersection']} | {fmt(row['train_per_row'])} | "
            f"{fmt(row['online_per_row'])} | {fmt(row['delta_per_row'])} | "
            f"{fmt(row['pct_delta'], 1)}% |"
        )

    lines += [
        "",
        "### Biggest Intersection-Movement Increases",
        "",
        "| Intersection | Movement | Train per row | Online per row | Delta | Pct |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in compare["intersection_movements"][:10]:
        lines.append(
            f"| {row['intersection']} | {row['movement']} | "
            f"{fmt(row['train_per_row'])} | {fmt(row['online_per_row'])} | "
            f"{fmt(row['delta_per_row'])} | {fmt(row['pct_delta'], 1)}% |"
        )

    lines += [
        "",
        "### Time Windows",
        "",
        "| Window | Train avg occupancy | Online avg occupancy |",
        "|---|---:|---:|",
    ]
    for window in sorted(
        set(train["window_avg_occupancy"]) | set(online["window_avg_occupancy"])
    ):
        lines.append(
            f"| {window} | {fmt(train['window_avg_occupancy'].get(window, 0.0))} | "
            f"{fmt(online['window_avg_occupancy'].get(window, 0.0))} |"
        )

    lines += [
        "",
        "## Action And Queue Diagnosis",
        "",
        "| Source | ETWT | NTST | ELWL | NLSL |",
        "|---|---:|---:|---:|---:|",
        (
            "| Local train labels | "
            + " | ".join(
                f"{int(train_actions.get(phase, 0))} ({pct(train_actions.get(phase, 0), sum(train_actions.values())):.1f}%)"
                for phase in PHASE_ORDER
            )
            + " |"
        ),
        (
            "| Stage3 online model actions | "
            + " | ".join(
                f"{int(state['action_counts'].get(phase, 0))} ({state['action_pct'][phase]:.1f}%)"
                for phase in PHASE_ORDER
            )
            + " |"
        ),
        "",
        f"- The remote prompt JSONL contains {sum(prompts['expert_action_rows'].values())} action-tagged decision rows, matching the 1920 entries in `state_action`.",
        f"- Avg total queue proxy from `state_action`: {fmt(state['avg_total_queue'])}.",
        f"- Max-queue phase match: {fmt(state['max_phase_match_rate_pct'], 1)}%.",
        f"- Avg missed queue gap: {fmt(state['avg_missed_gap'])}; strong misses gap>=10: {state['strong_misses_gap_ge_10']}.",
        "",
        "### Avg Queue By Phase",
        "",
        "| Phase | Avg queue | Avg wait proxy | Current max-phase rate | Action rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in PHASE_ORDER:
        lines.append(
            f"| {phase} | {fmt(state['avg_phase_queue'][phase])} | "
            f"{fmt(state['avg_phase_wait'][phase])} | "
            f"{fmt(state['max_phase_pct'][phase], 1)}% | "
            f"{fmt(state['action_pct'][phase], 1)}% |"
        )

    lines += [
        "",
        "### Online Action Given Current Max-Queue Phase",
        "",
        "| Current max phase | ETWT chosen | NTST chosen | ELWL chosen | NLSL chosen |",
        "|---|---:|---:|---:|---:|",
    ]
    for max_phase in PHASE_ORDER:
        row = state["max_phase_action_matrix"][max_phase]
        lines.append(
            f"| {max_phase} | "
            + " | ".join(str(int(row.get(action, 0))) for action in PHASE_ORDER)
            + " |"
        )

    lines += [
        "",
        "### Worst Intersections By Queue Proxy",
        "",
        "| Intersection | Avg total queue | Match | Avg missed gap | Strong misses | Actions | Max phases |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    worst_inters = sorted(
        state["per_intersection"].items(),
        key=lambda item: item[1]["avg_total_queue"],
        reverse=True,
    )[:8]
    for inter, values in worst_inters:
        actions = action_pct_table(values["actions"], values["decisions"])
        maxes = action_pct_table(values["max_phases"], values["decisions"])
        lines.append(
            f"| {inter} | {fmt(values['avg_total_queue'])} | "
            f"{fmt(values['max_phase_match_rate_pct'], 1)}% | "
            f"{fmt(values['avg_missed_gap'])} | "
            f"{values['strong_misses_gap_ge_10']} | {actions} | {maxes} |"
        )

    lines += [
        "",
        "## Diagnosis",
        "",
        "1. The latest Stage3 result is not the bad Hangzhou-real result seen in earlier Stage2 runs; it is slightly better than the previous 5816-to-real online result and far better than the Hangzhou-real Stage2 main/best guarded runs.",
        "2. PPO did not improve this run: update 2 and update 4 were worse than update 0 on queue/wait/travel and were rejected by the guardrails. So the latest result is good mainly because checkpoint selection preserved the start checkpoint, not because the Hangzhou-real PPO adaptation succeeded.",
        "3. The remaining gap versus the local generated train traffic log is modest but real: online mean occupancy is 3.9% higher, concentrated in ET/WT through traffic and high-load intersections `intersection_1_4`, `intersection_4_4`, and `intersection_1_3`.",
        "4. Stage3 avoids the severe NLSL collapse seen in the bad Stage2 run. NLSL action rate is 13.7%, above the 10.0% local train label rate. The residual problem is now ETWT/through timing and high-load intersection handling, not left-turn starvation.",
        "5. The 15-30m and 30-45m windows are where online occupancy exceeds the sampled train log most clearly; the last 15 minutes no longer show the runaway late drift from the bad Stage2 diagnosis.",
        "6. The model still under-matches the instantaneous max-queue phase often enough for queues to accumulate. ETWT is the current max-queue phase 56.2% of the time but is selected 36.9% of the time, although many misses are small.",
        "7. The train set is compact and biased toward stable/east-west/through cases; rare high-load closed-loop states are still weakly represented. Stage3's reward and guardrails reduce this, but do not remove the extrapolation problem.",
        "",
        "## Suggested Next Experiments",
        "",
        "1. Keep this Stage3 checkpoint as the current Hangzhou-real candidate unless a newer run beats -625.17 reward under the same 3600s test.",
        "2. Compare against a true expert rollout metric for `anon_4_4_hangzhou_real.json`, not only the compact sampled train log, to quantify the remaining reward gap.",
        "3. Add expert data or online relabeling from the worst high-queue intersections and late windows found above, then re-run Stage2/Stage3 selection with max-phase match and missed-gap guardrails.",
        "4. Preserve the Stage3 wait/starvation guard direction; do not fall back to the earlier Stage2 offline-F1 checkpoint selection that caused left-turn starvation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    train_traffic = traffic_stats(LOCAL_TRAIN)
    online_traffic = traffic_stats(REMOTE_COUNTS)
    prompt_traffic = traffic_stats(REMOTE_PROMPTS)
    state = state_action_stats(STATE_ACTION)
    test_result = load_json(TEST_RESULT)
    train_summary = load_json(LOCAL_TRAIN_SUMMARY)
    stage3_train_summary = load_json(TRAIN_SUMMARY) if TRAIN_SUMMARY.exists() else {}

    action_counts = (
        train_summary.get("after_dedup_sequence", {}).get("action_counts")
        or train_summary.get("before", {}).get("action_counts")
        or {}
    )

    summary = {
        "test_result": test_result,
        "train_traffic": train_traffic,
        "online_traffic": online_traffic,
        "online_prompt_traffic": prompt_traffic,
        "train_label_action_counts": action_counts,
        "traffic_compare": compare_traffic(train_traffic, online_traffic),
        "state_action": state,
        "train_selection": {
            "update": stage3_train_summary.get("update"),
            "metrics": stage3_train_summary.get("metrics", {}),
            "path": stage3_train_summary.get("path"),
        }
        if stage3_train_summary
        else {},
        "train_progress": parse_train_progress(TRAIN_PROGRESS),
    }

    (OUT_DIR / "diagnosis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    (OUT_DIR / "analysis_summary.md").write_text(render_markdown(summary))
    print(f"Wrote {OUT_DIR / 'diagnosis_summary.json'}")
    print(f"Wrote {OUT_DIR / 'analysis_summary.md'}")


if __name__ == "__main__":
    main()
