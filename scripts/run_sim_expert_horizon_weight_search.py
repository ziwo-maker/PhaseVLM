#!/usr/bin/env python3
"""Run SimExpert horizon ablations and expert-weight validation search.

The script intentionally delegates each simulation to run_sim_expert_action.py.
It only owns experiment orchestration, log capture, metric parsing, and ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRIC_RE = re.compile(
    r"^\s*(?P<traffic_file>[^:]+):\s*"
    r"ATT=(?P<att>[-+0-9.eE]+|N/A)\s*\|\s*"
    r"AQL=(?P<aql>[-+0-9.eE]+|N/A)\s*\|\s*"
    r"AWT=(?P<awt>[-+0-9.eE]+|N/A)\s*$"
)

DEFAULT_WEIGHT_SETS = (
    (1.0, 1.0, 1.0, 0.2),
    (2.0, 1.0, 1.0, 0.2),
    (1.0, 2.0, 1.0, 0.2),
    (1.0, 1.0, 2.0, 0.2),
    (1.0, 1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0, 0.5),
)


@dataclass(frozen=True)
class WeightSet:
    att: float
    aql: float
    awt: float
    spillback: float

    @property
    def label(self) -> str:
        return (
            f"att{format_float(self.att)}_"
            f"aql{format_float(self.aql)}_"
            f"awt{format_float(self.awt)}_"
            f"spill{format_float(self.spillback)}"
        )


@dataclass
class RunRow:
    run_id: str
    dataset: str
    traffic_file: str
    horizon_s: int
    weight: WeightSet
    status: str
    elapsed_sec: float
    log_file: str
    candidate_dir: str = ""
    traffic_series_output_path: str = ""
    path_to_work_directory: str = ""
    path_to_model: str = ""
    att: float | None = None
    aql: float | None = None
    awt: float | None = None


def format_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def bounded_label(value: str, max_chars: int = 120) -> str:
    """Keep external experiment labels within service limits while preserving identity."""
    label = str(value or "").strip()
    if len(label) <= max_chars:
        return label
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    keep_chars = max(1, max_chars - len(digest) - 1)
    return f"{label[:keep_chars]}_{digest}"


def parse_number_list(raw: str, kind: type = float) -> list:
    values = []
    for token in str(raw or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(kind(token))
    return values


def parse_traffic_files(raw_values: Iterable[str]) -> list[str]:
    files: list[str] = []
    for raw in raw_values:
        for token in str(raw or "").replace(",", " ").split():
            token = token.strip()
            if token:
                files.append(token)
    if not files:
        raise SystemExit("--traffic_files requires at least one file name.")
    return files


def parse_weight_sets(raw: str) -> list[WeightSet]:
    weight_sets = []
    for item in str(raw or "").split(";"):
        item = item.strip()
        if not item:
            continue
        values = parse_number_list(item, float)
        if len(values) != 4:
            raise SystemExit(
                "--weight_sets entries must have four numbers: "
                "att,aql,awt,spillback"
            )
        weight_sets.append(WeightSet(*values))
    return weight_sets


def resolve_weight_sets(args: argparse.Namespace) -> list[WeightSet]:
    explicit_sets = parse_weight_sets(args.weight_sets)
    if explicit_sets:
        return explicit_sets

    list_args = (
        args.att_weights,
        args.aql_weights,
        args.awt_weights,
        args.spillback_weights,
    )
    if any(str(item or "").strip() for item in list_args):
        att_values = parse_number_list(args.att_weights or "1.0", float)
        aql_values = parse_number_list(args.aql_weights or "1.0", float)
        awt_values = parse_number_list(args.awt_weights or "1.0", float)
        spill_values = parse_number_list(args.spillback_weights or "0.2", float)
        return [
            WeightSet(att, aql, awt, spill)
            for att, aql, awt, spill in itertools.product(
                att_values,
                aql_values,
                awt_values,
                spill_values,
            )
        ]

    return [WeightSet(*values) for values in DEFAULT_WEIGHT_SETS]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def parse_metrics_from_log(log_path: Path, traffic_file: str) -> tuple[float | None, float | None, float | None]:
    if not log_path.exists():
        return None, None, None
    found: tuple[float | None, float | None, float | None] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_RE.match(line.rstrip("\n"))
            if not match:
                continue
            if match.group("traffic_file").strip() != traffic_file:
                continue
            found = (
                parse_metric(match.group("att")),
                parse_metric(match.group("aql")),
                parse_metric(match.group("awt")),
            )
    return found if found is not None else (None, None, None)


def parse_metric(raw: str) -> float | None:
    if raw == "N/A":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def default_run_id(traffic_files: list[str], dataset: str) -> str:
    if len(traffic_files) == 1:
        return Path(traffic_files[0]).stem
    return str(dataset).strip() or "multi_traffic_files"


def candidate_id(traffic_file: str, horizon: int, weight: WeightSet) -> str:
    return f"{Path(traffic_file).stem}.h{horizon}.{weight.label}"


def parse_key_values_from_log(log_path: Path) -> dict[str, str]:
    keys = {
        "traffic_series_output_path",
        "path_to_work_directory",
        "path_to_model",
        "pipeline_input_path",
    }
    values: dict[str, str] = {}
    if not log_path.exists():
        return values
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in keys:
                values[key] = value.strip()
    return values


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    if path.resolve() == allowed_root.resolve() or not is_relative_to(path, allowed_root):
        raise RuntimeError(f"Refusing to delete outside allowed root: {path}")
    shutil.rmtree(path)


def build_command(
    args: argparse.Namespace,
    run_id: str,
    traffic_file: str,
    horizon: int,
    weight: WeightSet,
    candidate_dir: Path,
) -> list[str]:
    traffic_stem = Path(traffic_file).stem
    memo = (
        f"{args.memo_prefix}_{run_id}_{args.dataset}_{traffic_stem}_"
        f"h{horizon}_{weight.label}"
    )
    memo = bounded_label(memo, 120)
    cmd = [
        args.python,
        "run_sim_expert_action.py",
        "--dataset",
        args.dataset,
        "--traffic_file",
        traffic_file,
        "--llm_model",
        args.llm_model,
        "--llm_path",
        args.llm_path,
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--sim_horizon_s",
        str(horizon),
        "--expert_att_weight",
        str(weight.att),
        "--expert_aql_weight",
        str(weight.aql),
        "--expert_awt_weight",
        str(weight.awt),
        "--expert_spillback_weight",
        str(weight.spillback),
        "--memo",
        memo,
        "--proj_name",
        memo,
        "--record_output_dir",
        str(candidate_dir / "raw"),
        "--work_dir",
        str(candidate_dir / "work"),
        "--model_dir",
        str(candidate_dir / "model"),
    ]
    if args.skip_clean:
        cmd.append("--skip_clean")
    if args.skip_llm_backfill:
        cmd.append("--skip_llm_backfill")
    if args.skip_llm_init:
        cmd.append("--skip_llm_init")
    if args.clean_seq_len is not None:
        cmd.extend(["--clean_seq_len", str(args.clean_seq_len)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def run_one(
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    traffic_file: str,
    horizon: int,
    weight: WeightSet,
) -> RunRow:
    traffic_stem = Path(traffic_file).stem
    log_name = f"{args.dataset}.{traffic_stem}.h{horizon}.{weight.label}.log"
    log_path = output_dir / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / ".candidates" / candidate_id(traffic_file, horizon, weight)
    safe_rmtree(candidate_dir, output_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, run_id, traffic_file, horizon, weight, candidate_dir)

    start = time.time()
    if args.dry_run:
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# dry_run created_at={timestamp()}\n")
            handle.write(shlex.join(cmd) + "\n")
        status = "DRY_RUN"
    else:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "UTF-8",
                "LANG": env.get("LANG", "C.UTF-8"),
                "LC_ALL": env.get("LC_ALL", "C.UTF-8"),
                "TERM": "dumb",
                "NO_COLOR": "1",
                "FORCE_COLOR": "0",
                "VLLM_LOGGING_COLOR": "0",
                "TQDM_DISABLE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "WANDB_MODE": args.wandb_mode,
                "WANDB_SILENT": "true",
            }
        )
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# started_at={timestamp()}\n")
            handle.write("# command=" + shlex.join(cmd) + "\n")
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=args.repo_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(f"\n# finished_at={timestamp()} exit_code={proc.returncode}\n")
        status = "SUCCESS" if proc.returncode == 0 else f"FAILED_{proc.returncode}"

    elapsed = time.time() - start
    att, aql, awt = parse_metrics_from_log(log_path, traffic_file)
    logged_paths = parse_key_values_from_log(log_path)
    traffic_series_output_path = logged_paths.get(
        "traffic_series_output_path",
        str(candidate_dir / "raw" / "traffic_prompt_series_2s.jsonl"),
    )
    return RunRow(
        run_id=run_id,
        dataset=args.dataset,
        traffic_file=traffic_file,
        horizon_s=horizon,
        weight=weight,
        status=status,
        elapsed_sec=elapsed,
        log_file=str(log_path),
        candidate_dir=str(candidate_dir),
        traffic_series_output_path=traffic_series_output_path,
        path_to_work_directory=logged_paths.get("path_to_work_directory", str(candidate_dir / "work")),
        path_to_model=logged_paths.get("path_to_model", str(candidate_dir / "model")),
        att=att,
        aql=aql,
        awt=awt,
    )


def write_run_rows(path: Path, rows: list[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "dataset",
                "traffic_file",
                "horizon_s",
                "expert_att_weight",
                "expert_aql_weight",
                "expert_awt_weight",
                "expert_spillback_weight",
                "status",
                "elapsed_sec",
                "ATT",
                "AQL",
                "AWT",
                "log_file",
                "candidate_dir",
                "traffic_series_output_path",
                "path_to_work_directory",
                "path_to_model",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))


def row_to_dict(row: RunRow) -> dict:
    return {
        "run_id": row.run_id,
        "dataset": row.dataset,
        "traffic_file": row.traffic_file,
        "horizon_s": row.horizon_s,
        "expert_att_weight": row.weight.att,
        "expert_aql_weight": row.weight.aql,
        "expert_awt_weight": row.weight.awt,
        "expert_spillback_weight": row.weight.spillback,
        "status": row.status,
        "elapsed_sec": f"{row.elapsed_sec:.3f}",
        "ATT": "" if row.att is None else f"{row.att:.6f}",
        "AQL": "" if row.aql is None else f"{row.aql:.6f}",
        "AWT": "" if row.awt is None else f"{row.awt:.6f}",
        "log_file": row.log_file,
        "candidate_dir": row.candidate_dir,
        "traffic_series_output_path": row.traffic_series_output_path,
        "path_to_work_directory": row.path_to_work_directory,
        "path_to_model": row.path_to_model,
    }


def summarize_rows(
    rows: list[RunRow],
    selection_weights: tuple[float, float, float],
) -> list[dict]:
    grouped: dict[tuple, list[RunRow]] = defaultdict(list)
    for row in rows:
        key = (
            row.horizon_s,
            row.weight.att,
            row.weight.aql,
            row.weight.awt,
            row.weight.spillback,
        )
        grouped[key].append(row)

    summaries = []
    for key, group in grouped.items():
        success = [
            row
            for row in group
            if row.status == "SUCCESS"
            and row.att is not None
            and row.aql is not None
            and row.awt is not None
        ]
        horizon, att_w, aql_w, awt_w, spill_w = key
        summaries.append(
            {
                "horizon_s": horizon,
                "expert_att_weight": att_w,
                "expert_aql_weight": aql_w,
                "expert_awt_weight": awt_w,
                "expert_spillback_weight": spill_w,
                "num_files": len(group),
                "num_success": len(success),
                "mean_ATT": mean([row.att for row in success]),
                "mean_AQL": mean([row.aql for row in success]),
                "mean_AWT": mean([row.awt for row in success]),
                "total_elapsed_sec": sum(row.elapsed_sec for row in group),
                "objective": None,
            }
        )

    score_successful_summaries(summaries, selection_weights)
    summaries.sort(
        key=lambda item: (
            item["objective"] is None,
            float("inf") if item["objective"] is None else float(item["objective"]),
            item["mean_ATT"] is None,
            float("inf") if item["mean_ATT"] is None else float(item["mean_ATT"]),
        )
    )
    return summaries


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def score_successful_summaries(
    summaries: list[dict],
    selection_weights: tuple[float, float, float],
) -> None:
    metrics = ("mean_ATT", "mean_AQL", "mean_AWT")
    ranges: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = [float(item[metric]) for item in summaries if item[metric] is not None]
        if not values:
            ranges[metric] = (0.0, 0.0)
        else:
            ranges[metric] = (min(values), max(values))

    for item in summaries:
        if any(item[metric] is None for metric in metrics):
            item["objective"] = None
            continue
        objective = 0.0
        for metric, weight in zip(metrics, selection_weights):
            min_value, max_value = ranges[metric]
            value = float(item[metric])
            if max_value - min_value <= 1e-12:
                normalized = 0.0
            else:
                normalized = (value - min_value) / (max_value - min_value)
            item[f"norm_{metric[5:]}"] = normalized
            objective += float(weight) * normalized
        item["objective"] = objective


def write_summary(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "horizon_s",
        "expert_att_weight",
        "expert_aql_weight",
        "expert_awt_weight",
        "expert_spillback_weight",
        "num_files",
        "num_success",
        "mean_ATT",
        "mean_AQL",
        "mean_AWT",
        "norm_ATT",
        "norm_AQL",
        "norm_AWT",
        "objective",
        "total_elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(summaries, start=1):
            row = {"rank": idx}
            for field in fieldnames:
                if field == "rank":
                    continue
                value = item.get(field)
                row[field] = "" if value is None else value
            writer.writerow(row)


def row_matches_summary(row: RunRow, summary: dict) -> bool:
    return (
        int(row.horizon_s) == int(summary.get("horizon_s"))
        and abs(float(row.weight.att) - float(summary.get("expert_att_weight"))) <= 1e-12
        and abs(float(row.weight.aql) - float(summary.get("expert_aql_weight"))) <= 1e-12
        and abs(float(row.weight.awt) - float(summary.get("expert_awt_weight"))) <= 1e-12
        and abs(float(row.weight.spillback) - float(summary.get("expert_spillback_weight"))) <= 1e-12
    )


def append_jsonl_with_rewritten_snapshots(
    src_path: Path,
    output_handle,
    new_snapshot_dir: Path,
) -> int:
    written = 0
    with src_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                output_handle.write(line)
                written += 1
                continue
            snapshot_path = str(record.get("snapshot_path", "") or "").strip()
            if snapshot_path:
                snapshot_name = Path(snapshot_path).name
                if snapshot_name:
                    record["snapshot_path"] = str(new_snapshot_dir / snapshot_name)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def copytree_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def materialize_best_artifacts(
    output_dir: Path,
    summaries: list[dict],
    rows: list[RunRow],
    keep_candidate_logs: bool,
) -> dict:
    best = next((item for item in summaries if item.get("objective") is not None), None)
    if not best:
        return {}

    best_rows = [
        row
        for row in rows
        if row.status == "SUCCESS"
        and row.traffic_series_output_path
        and row_matches_summary(row, best)
    ]
    if not best_rows:
        return {}

    best_dir = output_dir / "best"
    safe_rmtree(best_dir, output_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    merged_raw_path = best_dir / "traffic_prompt_series_2s.jsonl"
    total_records = 0
    with merged_raw_path.open("w", encoding="utf-8") as output_handle:
        for row in best_rows:
            raw_src = Path(row.traffic_series_output_path)
            if not raw_src.exists():
                continue
            if len(best_rows) == 1:
                snapshot_dst_dir = best_dir / "snapshots"
            else:
                snapshot_dst_dir = best_dir / "snapshots" / Path(row.traffic_file).stem
            snapshot_src_dir = raw_src.parent / "snapshots"
            if snapshot_src_dir.exists():
                shutil.copytree(snapshot_src_dir, snapshot_dst_dir, dirs_exist_ok=True)
            total_records += append_jsonl_with_rewritten_snapshots(
                raw_src,
                output_handle,
                snapshot_dst_dir,
            )

    for row in best_rows:
        label = Path(row.traffic_file).stem if len(best_rows) > 1 else ""
        work_dst = best_dir / "work" / label if label else best_dir / "work"
        model_dst = best_dir / "model" / label if label else best_dir / "model"
        copytree_if_exists(Path(row.path_to_work_directory), work_dst)
        copytree_if_exists(Path(row.path_to_model), model_dst)

    log_paths = []
    existing_logs = [Path(row.log_file) for row in best_rows if row.log_file and Path(row.log_file).exists()]
    if len(existing_logs) == 1:
        best_log = best_dir / "best.log"
        shutil.copy2(existing_logs[0], best_log)
        log_paths.append(str(best_log))
    elif existing_logs:
        best_logs_dir = best_dir / "logs"
        best_logs_dir.mkdir(parents=True, exist_ok=True)
        for log_path in existing_logs:
            dst = best_logs_dir / log_path.name
            shutil.copy2(log_path, dst)
            log_paths.append(str(dst))

    if not keep_candidate_logs:
        safe_rmtree(output_dir / "logs", output_dir)

    return {
        "best_dir": str(best_dir),
        "best_raw_jsonl": str(merged_raw_path),
        "best_logs": log_paths,
        "best_num_rows": len(best_rows),
        "best_num_records": total_records,
    }


def write_best_config(
    path: Path,
    summaries: list[dict],
    args: argparse.Namespace,
    artifacts: dict | None = None,
) -> None:
    best = next((item for item in summaries if item.get("objective") is not None), None)
    payload = {
        "created_at": timestamp(),
        "selection_objective": "weighted_minmax_full_run_metrics",
        "selection_weights": {
            "ATT": args.selection_att_weight,
            "AQL": args.selection_aql_weight,
            "AWT": args.selection_awt_weight,
        },
        "best": best,
        "artifacts": artifacts or {},
    }
    if artifacts:
        payload["best_dir"] = artifacts.get("best_dir", "")
        payload["best_raw_jsonl"] = artifacts.get("best_raw_jsonl", "")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run --sim_horizon_s ablations and expert weight validation search "
            "for run_sim_expert_action.py."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--traffic_files",
        nargs="+",
        required=True,
        help="Validation traffic files. Comma-separated and space-separated forms are both accepted.",
    )
    parser.add_argument("--horizons", default="30,60,90")
    parser.add_argument(
        "--weight_sets",
        default="",
        help=(
            "Semicolon-separated att,aql,awt,spillback rows. "
            "Example: '1,1,1,0.2;1,2,1,0.2'."
        ),
    )
    parser.add_argument("--att_weights", default="")
    parser.add_argument("--aql_weights", default="")
    parser.add_argument("--awt_weights", default="")
    parser.add_argument("--spillback_weights", default="")
    parser.add_argument("--selection_att_weight", type=float, default=1.0)
    parser.add_argument("--selection_aql_weight", type=float, default=1.0)
    parser.add_argument("--selection_awt_weight", type=float, default=1.0)
    parser.add_argument("--run_id", default="")
    parser.add_argument(
        "--output_dir",
        default="",
        help=(
            "Defaults to records/sim_expert_validation_search/<run_id>. "
            "When --run_id is omitted, a single traffic file uses its file stem."
        ),
    )
    parser.add_argument("--memo_prefix", default="SimExpertSearch")
    parser.add_argument("--llm_model", default="gemma-3-27B")
    parser.add_argument("--llm_path", default="/home/data/model/gemma-3-27B/")
    parser.add_argument("--tensor_parallel_size", type=int, default=0)
    parser.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clean_seq_len", type=int, default=16)
    parser.add_argument("--skip_clean", action="store_true", default=True)
    parser.add_argument("--run_clean", dest="skip_clean", action="store_false")
    parser.add_argument("--skip_llm_backfill", action="store_true", default=True)
    parser.add_argument("--run_llm_backfill", dest="skip_llm_backfill", action="store_false")
    parser.add_argument("--skip_llm_init", action="store_true", default=True)
    parser.add_argument("--run_llm_init", dest="skip_llm_init", action="store_false")
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "disabled", "online"])
    parser.add_argument(
        "--keep_candidate_artifacts",
        action="store_true",
        default=False,
        help="Keep records/sim_expert_validation_search/<run_id>/.candidates after selecting best.",
    )
    parser.add_argument(
        "--keep_candidate_logs",
        action="store_true",
        default=False,
        help="Keep all per-candidate logs. By default only best.log is copied into best/.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run_sim_expert_action.py after '--'.",
    )
    args = parser.parse_args()
    if not str(args.python or "").strip():
        args.python = sys.executable
    args.repo_root = str(Path(__file__).resolve().parents[1])
    return args


def main() -> int:
    args = parse_args()
    traffic_files = parse_traffic_files(args.traffic_files)
    run_id = args.run_id or default_run_id(traffic_files, args.dataset)
    output_dir = Path(args.output_dir or Path("records") / "sim_expert_validation_search" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_rmtree(output_dir / ".candidates", output_dir)
    safe_rmtree(output_dir / "logs", output_dir)

    horizons = parse_number_list(args.horizons, int)
    weights = resolve_weight_sets(args)
    selection_weights = (
        args.selection_att_weight,
        args.selection_aql_weight,
        args.selection_awt_weight,
    )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": timestamp(),
                "run_id": run_id,
                "dataset": args.dataset,
                "traffic_files": traffic_files,
                "horizons": horizons,
                "weight_sets": [weight.__dict__ for weight in weights],
                "selection_weights": {
                    "ATT": args.selection_att_weight,
                    "AQL": args.selection_aql_weight,
                    "AWT": args.selection_awt_weight,
                },
                "dry_run": args.dry_run,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    rows: list[RunRow] = []
    total = len(traffic_files) * len(horizons) * len(weights)
    current = 0
    for horizon, weight, traffic_file in itertools.product(horizons, weights, traffic_files):
        current += 1
        print(
            f"[{current}/{total}] dataset={args.dataset} traffic_file={traffic_file} "
            f"horizon={horizon} weights={weight.label}",
            flush=True,
        )
        row = run_one(args, run_id, output_dir, traffic_file, horizon, weight)
        rows.append(row)
        write_run_rows(output_dir / "run_results.csv", rows)

    summaries = summarize_rows(rows, selection_weights)
    write_summary(output_dir / "summary.csv", summaries)
    artifacts = materialize_best_artifacts(
        output_dir,
        summaries,
        rows,
        keep_candidate_logs=args.keep_candidate_logs,
    )
    if not args.keep_candidate_artifacts:
        safe_rmtree(output_dir / ".candidates", output_dir)
    write_best_config(output_dir / "best_config.json", summaries, args, artifacts=artifacts)

    print(f"run_results_csv={output_dir / 'run_results.csv'}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print(f"best_config_json={output_dir / 'best_config.json'}")
    if args.dry_run:
        return 0
    best = next((item for item in summaries if item.get("objective") is not None), None)
    if not best:
        print(
            "error=no_successful_candidate_run; "
            f"inspect run_results_csv={output_dir / 'run_results.csv'} "
            f"and logs_dir={output_dir / 'logs'}",
            file=sys.stderr,
        )
        return 1
    if not artifacts.get("best_raw_jsonl"):
        print(
            "error=best_artifacts_not_materialized; "
            "the selected run finished but no raw JSONL was copied into best/; "
            f"inspect run_results_csv={output_dir / 'run_results.csv'}",
            file=sys.stderr,
        )
        return 2
    if artifacts.get("best_raw_jsonl"):
        print(f"best_raw_jsonl={artifacts['best_raw_jsonl']}")
    print(
        "best="
        f"horizon_s={best['horizon_s']} "
        f"weights=({best['expert_att_weight']},"
        f"{best['expert_aql_weight']},"
        f"{best['expert_awt_weight']},"
        f"{best['expert_spillback_weight']}) "
        f"ATT={best['mean_ATT']} AQL={best['mean_AQL']} AWT={best['mean_AWT']} "
        f"objective={best['objective']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
