#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projection_train.action_space import ACTION_LIST


RESULT_KEYS = {
    "reward": "test_reward_over",
    "queue": "test_avg_queue_len_over",
    "waiting": "test_avg_waiting_time_over",
    "travel": "test_avg_travel_time_over",
}


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(path_text: str, base: Path) -> Path:
    text = str(path_text or "").strip()
    if not text:
        return Path()
    path = Path(text)
    if path.is_absolute():
        return path
    return base / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _threshold(value: float | None) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if resolved > 1.0:
        resolved /= 100.0
    return resolved


def _iter_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _normalize_action(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text in ACTION_LIST:
            return text
        if text.isdigit():
            idx = int(text)
            if 0 <= idx < len(ACTION_LIST):
                return ACTION_LIST[idx]
    if isinstance(value, int) and 0 <= value < len(ACTION_LIST):
        return ACTION_LIST[int(value)]
    return ""


def _summarize_action_distribution(state_action_path: Path) -> Dict[str, Any]:
    if not state_action_path.is_file():
        return {
            "state_action_path": str(state_action_path),
            "total_actions": 0,
            "counts": {},
            "pct": {},
            "missing": True,
        }

    payload = _load_json(state_action_path)
    counts = {name: 0 for name in ACTION_LIST}
    total = 0
    for item in _iter_dicts(payload):
        action = ""
        for key in ("action", "select_action", "chosen_action", "pred_action", "phase", "phase_action"):
            if key in item:
                action = _normalize_action(item[key])
                break
        if not action:
            for key in ("action_prob", "action_probs", "prob", "probs", "policy", "action_distribution"):
                probs = item.get(key)
                if not isinstance(probs, list) or len(probs) < len(ACTION_LIST):
                    continue
                try:
                    values = [float(value) for value in probs[: len(ACTION_LIST)]]
                except (TypeError, ValueError):
                    continue
                action = ACTION_LIST[max(range(len(ACTION_LIST)), key=lambda idx: values[idx])]
                break
        if action:
            counts[action] += 1
            total += 1

    pct = {name: (counts[name] / total if total else 0.0) for name in ACTION_LIST}
    return {
        "state_action_path": str(state_action_path),
        "total_actions": total,
        "counts": counts,
        "pct": pct,
        "missing": False,
    }


def _find_state_action(repo_root: Path, work_dir: str) -> Path | None:
    if not work_dir:
        return None
    root = _resolve_path(work_dir, repo_root)
    direct = root / "state_action.json"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("state_action.json")) if root.is_dir() else []
    return matches[0] if matches else direct


def _result_score(results: Dict[str, Any], ranking_metric: str) -> float:
    if ranking_metric == "composite":
        reward = float(results.get(RESULT_KEYS["reward"], float("-inf")))
        queue = float(results.get(RESULT_KEYS["queue"], float("inf")))
        waiting = float(results.get(RESULT_KEYS["waiting"], float("inf")))
        travel = float(results.get(RESULT_KEYS["travel"], float("inf")))
        return reward - 0.05 * queue - 0.05 * waiting - 0.01 * travel
    key = RESULT_KEYS[ranking_metric]
    value = float(results.get(key, float("nan")))
    if math.isnan(value):
        return float("-inf")
    if ranking_metric == "reward":
        return value
    return -value


def _tie_breakers(results: Dict[str, Any]) -> List[float]:
    return [
        float(results.get(RESULT_KEYS["reward"], float("-inf"))),
        -float(results.get(RESULT_KEYS["queue"], float("inf"))),
        -float(results.get(RESULT_KEYS["travel"], float("inf"))),
        -float(results.get(RESULT_KEYS["waiting"], float("inf"))),
    ]


def _constraint_reasons(candidate: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    reasons: List[str] = []
    action_dist = candidate.get("action_distribution") or {}
    pct = action_dist.get("pct") or {}
    if action_dist.get("missing") and any(
        value is not None
        for value in (args.max_etwt, args.max_ntst, args.max_elwl, args.max_nlsl, args.min_etwt, args.min_ntst, args.min_elwl, args.min_nlsl)
    ):
        reasons.append("missing action distribution")
        return reasons

    checks = [
        ("ETWT", "max", _threshold(args.max_etwt)),
        ("NTST", "max", _threshold(args.max_ntst)),
        ("ELWL", "max", _threshold(args.max_elwl)),
        ("NLSL", "max", _threshold(args.max_nlsl)),
        ("ETWT", "min", _threshold(args.min_etwt)),
        ("NTST", "min", _threshold(args.min_ntst)),
        ("ELWL", "min", _threshold(args.min_elwl)),
        ("NLSL", "min", _threshold(args.min_nlsl)),
    ]
    for action, mode, threshold in checks:
        if threshold is None:
            continue
        value = float(pct.get(action, 0.0))
        if mode == "max" and value > threshold:
            reasons.append(f"{action}={value:.4f} > max {threshold:.4f}")
        if mode == "min" and value < threshold:
            reasons.append(f"{action}={value:.4f} < min {threshold:.4f}")
    return reasons


def _next_backup_path(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = path.name
    for idx in range(1000):
        suffix = f".{timestamp}" if idx == 0 else f".{timestamp}.{idx}"
        candidate = path.with_name(stem + suffix)
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Failed to find backup path for {path}")


def _promote_to_best(selected: Dict[str, Any], stage_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(selected["checkpoint_path"]).resolve()
    output_dir = stage_root / "checkpoints"
    best_path = output_dir / "best.pt"
    backup_path = output_dir / args.best_backup_name

    payload: Dict[str, Any] = {
        "selected_checkpoint": str(ckpt_path),
        "best_path": str(best_path),
        "backup_path": "",
        "promoted": False,
        "dry_run": bool(args.dry_run),
    }
    if ckpt_path == best_path.resolve():
        payload["note"] = "Selected checkpoint is already best.pt."
        return payload
    if args.dry_run:
        payload["backup_path"] = str(_next_backup_path(backup_path) if best_path.exists() else "")
        payload["promoted"] = True
        return payload

    output_dir.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        backup = _next_backup_path(backup_path)
        shutil.copy2(best_path, backup)
        payload["backup_path"] = str(backup)
    shutil.copy2(ckpt_path, best_path)
    payload["promoted"] = True
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and optionally promote a Stage 2 projection checkpoint using online eval results.",
    )
    parser.add_argument("--stage_root", type=str, required=True)
    parser.add_argument("--repo_root", type=str, default=str(_REPO_ROOT))
    parser.add_argument("--eval_glob", type=str, default="artifacts/short_eval*.json")
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default="reward",
        choices=["reward", "queue", "waiting", "travel", "composite"],
    )
    parser.add_argument("--max_etwt", type=float, default=None)
    parser.add_argument("--max_ntst", type=float, default=None)
    parser.add_argument("--max_elwl", type=float, default=None)
    parser.add_argument("--max_nlsl", type=float, default=None)
    parser.add_argument("--min_etwt", type=float, default=None)
    parser.add_argument("--min_ntst", type=float, default=None)
    parser.add_argument("--min_elwl", type=float, default=None)
    parser.add_argument("--min_nlsl", type=float, default=None)
    parser.add_argument("--allow_external_ckpt", action="store_true", default=False)
    parser.add_argument("--fallback_to_unconstrained", action="store_true", default=False)
    parser.add_argument("--promote_to_best", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--best_backup_name", type=str, default="best.offline.pt")
    parser.add_argument("--selection_json_name", type=str, default="best_selection.json")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    stage_root = _resolve_path(args.stage_root, repo_root).resolve()
    checkpoint_root = (stage_root / "checkpoints").resolve()
    result_paths = sorted(stage_root.glob(args.eval_glob))
    if not result_paths:
        raise SystemExit(f"No eval result files matched {stage_root / args.eval_glob}")

    candidates: List[Dict[str, Any]] = []
    for result_path in result_paths:
        payload = _load_json(result_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected a JSON object: {result_path}")
        results = payload.get("results") or {}
        run_args = payload.get("args") or {}
        ckpt_path = _resolve_path(str(run_args.get("projection_ckpt", "")), repo_root).resolve()
        internal_ckpt = _is_relative_to(ckpt_path, checkpoint_root)
        name = result_path.stem
        for prefix in ("short_eval_1200_", "short_eval_"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        action_path = _find_state_action(repo_root, str(payload.get("work_dir", "")))
        candidate = {
            "name": name,
            "result_json": str(result_path),
            "checkpoint_path": str(ckpt_path),
            "internal_checkpoint": internal_ckpt,
            "results": results,
            "work_dir": payload.get("work_dir", ""),
            "action_distribution": _summarize_action_distribution(action_path) if action_path is not None else {},
            "score": _result_score(results, args.ranking_metric),
            "ranking_metric": args.ranking_metric,
        }
        reasons = []
        if not ckpt_path.is_file():
            reasons.append(f"missing checkpoint: {ckpt_path}")
        if not internal_ckpt and not args.allow_external_ckpt:
            reasons.append("external checkpoint")
        reasons.extend(_constraint_reasons(candidate, args))
        candidate["eligible"] = not reasons
        candidate["ineligible_reasons"] = reasons
        candidates.append(candidate)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    used_constraint_fallback = False
    if not eligible and args.fallback_to_unconstrained:
        eligible = [
            candidate
            for candidate in candidates
            if Path(candidate["checkpoint_path"]).is_file()
            and (candidate["internal_checkpoint"] or args.allow_external_ckpt)
        ]
        used_constraint_fallback = True
    if not eligible:
        report = {
            "stage_root": str(stage_root),
            "ranking_metric": args.ranking_metric,
            "selected": None,
            "candidates": candidates,
        }
        output_json = _resolve_path(args.output_json, repo_root) if args.output_json else stage_root / "artifacts" / "online_checkpoint_selection.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(_json_dump(report) + "\n", encoding="utf-8")
        raise SystemExit(f"No eligible checkpoint found. Wrote report: {output_json}")

    selected = max(eligible, key=lambda item: (float(item["score"]), *_tie_breakers(item.get("results") or {})))
    promotion: Dict[str, Any] | None = None
    if args.promote_to_best:
        promotion = _promote_to_best(selected, stage_root, args)
        sidecar = stage_root / "checkpoints" / args.selection_json_name
        sidecar.write_text(
            _json_dump(
                {
                    "selected": selected,
                    "promotion": promotion,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "stage_root": str(stage_root),
        "ranking_metric": args.ranking_metric,
        "constraints": {
            "max_etwt": _threshold(args.max_etwt),
            "max_ntst": _threshold(args.max_ntst),
            "max_elwl": _threshold(args.max_elwl),
            "max_nlsl": _threshold(args.max_nlsl),
            "min_etwt": _threshold(args.min_etwt),
            "min_ntst": _threshold(args.min_ntst),
            "min_elwl": _threshold(args.min_elwl),
            "min_nlsl": _threshold(args.min_nlsl),
        },
        "used_constraint_fallback": used_constraint_fallback,
        "selected": selected,
        "promotion": promotion,
        "candidates": sorted(candidates, key=lambda item: (bool(item["eligible"]), float(item["score"])), reverse=True),
    }
    output_json = _resolve_path(args.output_json, repo_root) if args.output_json else stage_root / "artifacts" / "online_checkpoint_selection.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json_dump(report) + "\n", encoding="utf-8")
    print(_json_dump({"selected": selected, "promotion": promotion, "output_json": str(output_json)}))


if __name__ == "__main__":
    main()
