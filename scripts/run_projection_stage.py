#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_ROOT = (REPO_ROOT / "records").resolve()


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except TypeError:
                reconfigure(line_buffering=True)


def _default_python_bin() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable or "python3"


def _default_torchrun_bin() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "torchrun"
    if candidate.is_file():
        return str(candidate)
    return "torchrun"


def _strip_remainder(args: Sequence[str]) -> List[str]:
    values = list(args)
    if values and values[0] == "--":
        return values[1:]
    return values


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _first_positive_int(*values: Any, fallback: int) -> int:
    for value in values:
        resolved = _safe_int(value, 0)
        if resolved > 0:
            return resolved
    return _safe_int(fallback, 1)


def _normalize_model_path(model_path: Any) -> str:
    text = str(model_path or "").strip()
    if not text:
        return ""
    trimmed = text.rstrip("/\\")
    return trimmed or text


def _abs_path(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute():
        return raw
    return REPO_ROOT / raw


def _resolve_stage_root(path_text: str) -> Path:
    stage_root = _abs_path(path_text).resolve()
    try:
        stage_root.relative_to(RECORDS_ROOT)
    except ValueError as exc:
        raise SystemExit("--stage_root must be inside records/.") from exc
    return stage_root


def _memo_from_stage_root(stage_root: Path) -> str:
    return (stage_root.relative_to(RECORDS_ROOT) / "test_runs").as_posix()


def _read_option_values(args: Sequence[str], option: str) -> List[str] | None:
    idx = 0
    found: List[str] | None = None
    values = list(args)
    while idx < len(values):
        if values[idx] != option:
            idx += 1
            continue
        idx += 1
        current: List[str] = []
        while idx < len(values) and not values[idx].startswith("--"):
            current.append(values[idx])
            idx += 1
        found = current
    return found


def _read_option_text(args: Sequence[str], option: str, default: str = "") -> str:
    values = _read_option_values(args, option)
    if not values:
        return default
    return str(values[-1])


def _has_flag(args: Sequence[str], flag: str) -> bool:
    return any(item == flag for item in args)


def _ensure_option(args: List[str], option: str, values: Sequence[str]) -> None:
    if _read_option_values(args, option) is not None:
        return
    args.append(option)
    args.extend(str(item) for item in values)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, _json_text(payload) + "\n")


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _render_log_block(title: str, payload: Any) -> str:
    lines = [f"\n===== {title} =====\n"]
    if isinstance(payload, str):
        lines.append(payload)
        if not payload.endswith("\n"):
            lines.append("\n")
    else:
        lines.append(_json_text(payload))
        lines.append("\n")
    return "".join(lines)


def _append_log_block(log_path: Path, title: str, payload: Any) -> None:
    _append_text(log_path, _render_log_block(title, payload))


def _emit_live_text(text: str, *log_paths: Path) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()
    seen: set[Path] = set()
    for log_path in log_paths:
        if log_path in seen:
            continue
        seen.add(log_path)
        _append_text(log_path, text)


def _emit_live_block(title: str, payload: Any, *log_paths: Path) -> None:
    _emit_live_text(_render_log_block(title, payload), *log_paths)


def _write_command_script(path: Path, env_overrides: Dict[str, str], command: Sequence[str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}"]
    for key, value in env_overrides.items():
        if str(value).strip():
            lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append("exec " + shlex.join([str(item) for item in command]))
    _write_text(path, "\n".join(lines) + "\n")
    path.chmod(0o755)


def _run_command_logged(
    command: Sequence[str],
    env: Dict[str, str],
    log_path: Path,
    step_name: str,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    started_clock = perf_counter()
    with log_path.open("a", encoding="utf-8") as log_handle:
        start_line = f"===== START {step_name} {_format_timestamp(started_at)} ====="
        sys.stdout.write(f"{start_line}\n")
        sys.stdout.flush()
        log_handle.write(f"\n{start_line}\n")
        log_handle.flush()
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
        ended_at = datetime.now()
        duration_seconds = perf_counter() - started_clock
        end_line = (
            f"===== END {step_name} rc={return_code} {_format_timestamp(ended_at)} "
            f"duration={_format_duration(duration_seconds)} ({duration_seconds:.1f}s) ====="
        )
        sys.stdout.write(f"{end_line}\n")
        sys.stdout.flush()
        log_handle.write(f"{end_line}\n")
        log_handle.flush()
        return {
            "return_code": return_code,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "duration": _format_duration(duration_seconds),
        }


def _load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {path}")
    return payload


def _extract_checkpoint_args(payload: Dict[str, Any]) -> Dict[str, Any]:
    args = payload.get("args")
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def _checkpoint_summary(path: Path) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(path)
    args = _extract_checkpoint_args(payload)
    return {
        "path": str(path),
        "keys": sorted(payload.keys()),
        "args": args,
        "metrics": payload.get("metrics") or {},
        "checkpoint_selection": payload.get("checkpoint_selection") or {},
        "epoch": payload.get("epoch"),
        "update": payload.get("update"),
        "update_idx": payload.get("update_idx"),
    }


def _select_checkpoint(output_dir: Path) -> Dict[str, Any]:
    best_path = output_dir / "best.pt"
    if best_path.is_file():
        return {"path": best_path, "selected": "best.pt"}
    last_path = output_dir / "last.pt"
    if last_path.is_file():
        return {"path": last_path, "selected": "last.pt"}
    raise FileNotFoundError(f"No best.pt or last.pt found under {output_dir}")


def _resolve_projection_runtime(
    primary_args: Dict[str, Any],
    fallback_args: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    secondary = fallback_args or {}
    llm_path = _normalize_model_path(
        primary_args.get("llm_path")
        or primary_args.get("model_path")
        or secondary.get("llm_path")
        or secondary.get("model_path")
    )
    if not llm_path:
        raise RuntimeError("Failed to resolve llm_path from checkpoint args.")
    return {
        "llm_path": llm_path,
        "seq_len": _first_positive_int(primary_args.get("seq_len"), secondary.get("seq_len"), fallback=16),
        "image_size": _first_positive_int(
            primary_args.get("image_size"),
            secondary.get("image_size"),
            fallback=224,
        ),
        "periodicity": _first_positive_int(
            primary_args.get("periodicity"),
            secondary.get("periodicity"),
            fallback=16,
        ),
        "input_dim": _first_positive_int(primary_args.get("input_dim"), secondary.get("input_dim"), fallback=8),
        "hidden_dim": _first_positive_int(primary_args.get("hidden_dim"), secondary.get("hidden_dim"), fallback=64),
        "prefix_tokens": _first_positive_int(
            primary_args.get("prefix_tokens"),
            secondary.get("prefix_tokens"),
            fallback=16,
        ),
        "prefix_mode": str(primary_args.get("prefix_mode") or secondary.get("prefix_mode") or "prefix"),
        "max_length": _first_positive_int(
            primary_args.get("max_length"),
            secondary.get("max_length"),
            fallback=128,
        ),
        "prompt_mode": str(primary_args.get("prompt_mode") or secondary.get("prompt_mode") or "image_only_short"),
    }


def _resolve_stage2_runtime(ckpt_path: Path) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(ckpt_path)
    return _resolve_projection_runtime(_extract_checkpoint_args(payload))


def _resolve_stage3_runtime(ckpt_path: Path) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(ckpt_path)
    primary_args = _extract_checkpoint_args(payload)
    stage2_ckpt = _normalize_model_path(primary_args.get("stage2_ckpt"))
    if not stage2_ckpt:
        raise RuntimeError(f"Missing stage2_ckpt in {ckpt_path}")
    stage2_path = _abs_path(stage2_ckpt).resolve()
    stage2_payload = _load_checkpoint_payload(stage2_path)
    runtime = _resolve_projection_runtime(primary_args, _extract_checkpoint_args(stage2_payload))
    runtime["stage2_ckpt"] = str(stage2_path)
    return runtime


def _precision_flags(choice: str, train_args: Sequence[str]) -> List[str]:
    resolved = str(choice or "follow_train").strip().lower()
    if resolved == "follow_train":
        if _has_flag(train_args, "--fp16"):
            return ["--fp16"]
        if _has_flag(train_args, "--bf16"):
            return ["--bf16"]
        return []
    if resolved == "fp16":
        return ["--fp16"]
    if resolved == "bf16":
        return ["--bf16"]
    return []


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timing_fields(prefix: str, timing: Dict[str, Any] | None) -> Dict[str, Any]:
    if not timing:
        return {
            f"{prefix}_started_at": "",
            f"{prefix}_ended_at": "",
            f"{prefix}_duration": "",
            f"{prefix}_duration_seconds": 0.0,
        }
    return {
        f"{prefix}_started_at": _format_timestamp(timing["started_at"]),
        f"{prefix}_ended_at": _format_timestamp(timing["ended_at"]),
        f"{prefix}_duration": str(timing["duration"]),
        f"{prefix}_duration_seconds": round(float(timing["duration_seconds"]), 3),
    }


def _append_report(report_path: Path, payload: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        report_path.write_text("# Projection Stage Report\n\n", encoding="utf-8")

    lines = [
        f"## {payload['stage_name']} | {payload['started_at']}",
        "",
        f"- status: `{payload['status']}`",
        f"- stage_root: `{payload['stage_root']}`",
        f"- started_at: `{payload['started_at']}`",
        f"- ended_at: `{payload['ended_at']}`",
        f"- duration: `{payload['duration']}`",
        f"- duration_seconds: `{payload.get('duration_seconds', 0.0)}`",
        f"- train_started_at: `{payload.get('train_started_at', '')}`",
        f"- train_ended_at: `{payload.get('train_ended_at', '')}`",
        f"- train_duration: `{payload.get('train_duration', '')}`",
        f"- train_duration_seconds: `{payload.get('train_duration_seconds', 0.0)}`",
        f"- test_started_at: `{payload.get('test_started_at', '')}`",
        f"- test_ended_at: `{payload.get('test_ended_at', '')}`",
        f"- test_duration: `{payload.get('test_duration', '')}`",
        f"- test_duration_seconds: `{payload.get('test_duration_seconds', 0.0)}`",
        f"- train_log: `{payload['train_log']}`",
        f"- test_log: `{payload['test_log']}`",
        f"- checkpoint: `{payload.get('checkpoint_path', '')}`",
        f"- checkpoint_selected: `{payload.get('checkpoint_selected', '')}`",
        f"- test_work_dir: `{payload.get('test_work_dir', '')}`",
        "",
        "### Train Command",
        "",
        "```bash",
        payload["train_command"],
        "```",
        "",
        "### Test Command",
        "",
        "```bash",
        payload.get("test_command", ""),
        "```",
        "",
        "### Environment",
        "",
        "```json",
        _json_text(payload["environment"]),
        "```",
        "",
        "### Training Args",
        "",
        "```json",
        _json_text(payload.get("training_args", {})),
        "```",
        "",
        "### Training Metrics",
        "",
        "```json",
        _json_text(payload.get("training_metrics", {})),
        "```",
        "",
        "### Test Args",
        "",
        "```json",
        _json_text(payload.get("test_args", {})),
        "```",
        "",
        "### Test Results",
        "",
        "```json",
        _json_text(payload.get("test_results", {})),
        "```",
        "",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def _common_stage_dirs(stage_root: Path) -> Dict[str, Path]:
    return {
        "logs_dir": stage_root / "logs",
        "commands_dir": stage_root / "commands",
        "artifacts_dir": stage_root / "artifacts",
    }


def _wrapper_option_names(stage: str) -> set[str]:
    option_names = {
        "--stage_root",
        "--report_md",
        "--python_bin",
        "--torchrun_bin",
        "--nproc_per_node",
        "--cuda_visible_devices",
        "--test_dataset",
        "--test_traffic_file",
        "--test_device",
        "--test_num_rounds",
        "--test_proj_name",
        "--test_run_counts",
        "--test_precision",
    }
    if stage == "stage3":
        option_names.add("--pytorch_cuda_alloc_conf")
    return option_names


def _split_wrapper_and_train_args(argv: Sequence[str]) -> tuple[List[str], List[str]]:
    values = list(argv)
    if not values:
        return [], []

    stage = values[0]
    if stage not in {"stage2", "stage3"}:
        return values, []

    wrapper_args = [stage]
    train_args: List[str] = []
    option_names = _wrapper_option_names(stage)
    idx = 1

    while idx < len(values):
        token = values[idx]
        if token == "--":
            train_args.extend(values[idx + 1 :])
            break
        if token in {"-h", "--help"}:
            wrapper_args.append(token)
            idx += 1
            continue

        matched = next(
            (name for name in option_names if token == name or token.startswith(f"{name}=")),
            None,
        )
        if matched is None:
            train_args.extend(values[idx:])
            break

        wrapper_args.append(token)
        if token == matched:
            idx += 1
            if idx >= len(values):
                raise SystemExit(f"Missing value for {matched}")
            wrapper_args.append(values[idx])
        idx += 1

    return wrapper_args, train_args


def _run_stage(stage_name: str, args: argparse.Namespace) -> int:
    stage_root = _resolve_stage_root(args.stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    dirs = _common_stage_dirs(stage_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    train_log = dirs["logs_dir"] / "train.log"
    test_log = dirs["logs_dir"] / "test.log"
    train_command_path = dirs["commands_dir"] / "train.sh"
    test_command_path = dirs["commands_dir"] / "test.sh"
    train_summary_path = dirs["artifacts_dir"] / "train_summary.json"
    test_summary_path = dirs["artifacts_dir"] / "test_summary.json"
    stage_summary_path = dirs["artifacts_dir"] / "stage_summary.json"
    test_result_json = dirs["artifacts_dir"] / "test_result.json"
    report_path = _abs_path(args.report_md).resolve()

    raw_train_args = _strip_remainder(args.train_args)
    if not raw_train_args:
        raise SystemExit("Training arguments are required.")

    output_dir_default = stage_root / "checkpoints"
    _ensure_option(raw_train_args, "--output_dir", [str(output_dir_default)])

    train_script = REPO_ROOT / "projection_train" / (
        "train_projection_bc.py" if args.stage == "stage2" else "train_projection_ppo_traj.py"
    )
    if args.stage == "stage3":
        _ensure_option(raw_train_args, "--work_root", [str(stage_root / "env_work")])

    output_dir = _abs_path(_read_option_text(raw_train_args, "--output_dir")).resolve()
    train_env = os.environ.copy()
    env_summary: Dict[str, str] = {}
    if str(args.cuda_visible_devices or "").strip():
        train_env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
        env_summary["CUDA_VISIBLE_DEVICES"] = train_env["CUDA_VISIBLE_DEVICES"]
    if args.stage == "stage3" and str(args.pytorch_cuda_alloc_conf or "").strip():
        train_env["PYTORCH_CUDA_ALLOC_CONF"] = str(args.pytorch_cuda_alloc_conf).strip()
        env_summary["PYTORCH_CUDA_ALLOC_CONF"] = train_env["PYTORCH_CUDA_ALLOC_CONF"]
    train_env.setdefault("PYTHONUNBUFFERED", "1")
    env_summary["PYTHONUNBUFFERED"] = train_env["PYTHONUNBUFFERED"]

    train_command: List[str] = [str(args.torchrun_bin)]
    if args.stage == "stage3":
        train_command.append("--standalone")
    train_command.extend(
        [
            "--nproc_per_node",
            str(args.nproc_per_node),
            str(train_script),
            *raw_train_args,
        ]
    )

    _append_log_block(
        train_log,
        "Wrapper Train Context",
        {
            "stage": stage_name,
            "stage_root": str(stage_root),
            "output_dir": str(output_dir),
            "train_script": str(train_script),
            "train_command": shlex.join(train_command),
            "env_overrides": env_summary,
            "requested_train_args": raw_train_args,
        },
    )
    _write_command_script(train_command_path, env_summary, train_command)

    started_at = datetime.now()
    started_clock = perf_counter()
    status = "train_failed"
    checkpoint_path = ""
    checkpoint_selected = ""
    test_command: List[str] = []
    test_work_dir = ""
    training_args: Dict[str, Any] = {}
    training_metrics: Dict[str, Any] = {}
    test_args: Dict[str, Any] = {}
    test_results: Dict[str, Any] = {}
    train_timing: Dict[str, Any] | None = None
    test_timing: Dict[str, Any] | None = None

    train_timing = _run_command_logged(train_command, train_env, train_log, "train")
    train_rc = int(train_timing["return_code"])
    if train_rc != 0:
        ended_at = datetime.now()
        duration_seconds = perf_counter() - started_clock
        stage_summary = {
            "stage_name": stage_name,
            "status": status,
            "stage_root": str(stage_root),
            "started_at": _format_timestamp(started_at),
            "ended_at": _format_timestamp(ended_at),
            "duration": _format_duration(duration_seconds),
            "duration_seconds": round(duration_seconds, 3),
            **_timing_fields("train", train_timing),
            **_timing_fields("test", test_timing),
        }
        _write_json(stage_summary_path, stage_summary)
        _emit_live_block("Stage Timing Summary", stage_summary, train_log)
        _append_report(
            report_path,
            {
                "stage_name": stage_name,
                "status": status,
                "stage_root": str(stage_root),
                "started_at": _format_timestamp(started_at),
                "ended_at": _format_timestamp(ended_at),
                "duration": _format_duration(duration_seconds),
                "duration_seconds": round(duration_seconds, 3),
                "train_log": str(train_log),
                "test_log": str(test_log),
                "train_command": shlex.join(train_command),
                "test_command": "",
                "environment": env_summary,
                "training_args": {},
                "training_metrics": {},
                "test_args": {},
                "test_results": {},
                **_timing_fields("train", train_timing),
                **_timing_fields("test", test_timing),
            },
        )
        return train_rc

    selected_ckpt = _select_checkpoint(output_dir)
    checkpoint_path = str(selected_ckpt["path"])
    checkpoint_selected = str(selected_ckpt["selected"])
    train_summary = _checkpoint_summary(selected_ckpt["path"])
    training_args = train_summary["args"]
    training_metrics = train_summary["metrics"]
    _write_json(train_summary_path, train_summary)
    _append_log_block(train_log, "Checkpoint Summary", train_summary)

    if args.stage == "stage2":
        runtime = _resolve_stage2_runtime(selected_ckpt["path"])
        test_dataset = str(args.test_dataset or "").strip()
        test_traffic_file = str(args.test_traffic_file or "").strip()
        if not test_dataset or not test_traffic_file:
            raise SystemExit("Stage 2 requires --test_dataset and --test_traffic_file.")
    else:
        runtime = _resolve_stage3_runtime(selected_ckpt["path"])
        test_dataset = str(args.test_dataset or _read_option_text(raw_train_args, "--dataset", "jinan")).strip()
        test_traffic_file = str(args.test_traffic_file or "").strip()
        if not test_traffic_file:
            test_traffic_file = _read_option_text(raw_train_args, "--eval_traffic_file")
        if not test_traffic_file:
            test_traffic_file = _read_option_text(raw_train_args, "--traffic_file")
        if not test_dataset or not test_traffic_file:
            raise SystemExit("Stage 3 failed to resolve final test dataset or traffic file.")

    test_env = os.environ.copy()
    for key, value in env_summary.items():
        test_env[key] = value

    test_command = [
        str(args.python_bin),
        str(REPO_ROOT / "projection_train" / "run_projection_online.py"),
        "--memo",
        _memo_from_stage_root(stage_root),
        "--projection_ckpt",
        checkpoint_path,
        "--dataset",
        test_dataset,
        "--traffic_file",
        test_traffic_file,
        "--device",
        str(args.test_device),
        "--seed",
        str(_safe_int(training_args.get("seed"), 2024)),
        "--num_rounds",
        str(args.test_num_rounds),
        "--proj_name",
        str(args.test_proj_name),
        "--run_counts",
        str(args.test_run_counts),
        "--result_json",
        str(test_result_json),
        "--llm_path",
        str(runtime["llm_path"]),
        "--seq_len",
        str(runtime["seq_len"]),
        "--image_size",
        str(runtime["image_size"]),
        "--periodicity",
        str(runtime["periodicity"]),
        "--input_dim",
        str(runtime["input_dim"]),
        "--hidden_dim",
        str(runtime["hidden_dim"]),
        "--prefix_tokens",
        str(runtime["prefix_tokens"]),
        "--prefix_mode",
        str(runtime["prefix_mode"]),
        "--max_length",
        str(runtime["max_length"]),
        "--prompt_mode",
        str(runtime["prompt_mode"]),
        *_precision_flags(args.test_precision, raw_train_args),
    ]
    test_args = {
        "dataset": test_dataset,
        "traffic_file": test_traffic_file,
        "device": args.test_device,
        "seed": _safe_int(training_args.get("seed"), 2024),
        "num_rounds": args.test_num_rounds,
        "proj_name": args.test_proj_name,
        "run_counts": args.test_run_counts,
        "precision": args.test_precision,
        "runtime": runtime,
    }

    _append_log_block(
        test_log,
        "Wrapper Test Context",
        {
            "stage": stage_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_selected": checkpoint_selected,
            "test_command": shlex.join(test_command),
            "env_overrides": env_summary,
            "test_args": test_args,
        },
    )
    _write_command_script(test_command_path, env_summary, test_command)

    test_timing = _run_command_logged(test_command, test_env, test_log, "test")
    test_rc = int(test_timing["return_code"])
    if test_rc == 0 and test_result_json.is_file():
        test_payload = json.loads(test_result_json.read_text(encoding="utf-8"))
        test_results = test_payload.get("results") or {}
        test_work_dir = str(test_payload.get("work_dir") or "")
        _write_json(test_summary_path, test_payload)
        _append_log_block(test_log, "Structured Test Result", test_payload)
        status = "success"
    else:
        status = "test_failed"

    ended_at = datetime.now()
    duration_seconds = perf_counter() - started_clock
    stage_summary = {
        "stage_name": stage_name,
        "status": status,
        "stage_root": str(stage_root),
        "started_at": _format_timestamp(started_at),
        "ended_at": _format_timestamp(ended_at),
        "duration": _format_duration(duration_seconds),
        "duration_seconds": round(duration_seconds, 3),
        **_timing_fields("train", train_timing),
        **_timing_fields("test", test_timing),
    }
    _write_json(stage_summary_path, stage_summary)
    _emit_live_block("Stage Timing Summary", stage_summary, train_log, test_log)
    _append_report(
        report_path,
        {
            "stage_name": stage_name,
            "status": status,
            "stage_root": str(stage_root),
            "started_at": _format_timestamp(started_at),
            "ended_at": _format_timestamp(ended_at),
            "duration": _format_duration(duration_seconds),
            "duration_seconds": round(duration_seconds, 3),
            "train_log": str(train_log),
            "test_log": str(test_log),
            "checkpoint_path": checkpoint_path,
            "checkpoint_selected": checkpoint_selected,
            "test_work_dir": test_work_dir,
            "train_command": shlex.join(train_command),
            "test_command": shlex.join(test_command),
            "environment": env_summary,
            "training_args": training_args,
            "training_metrics": training_metrics,
            "test_args": test_args,
            "test_results": test_results,
            **_timing_fields("train", train_timing),
            **_timing_fields("test", test_timing),
        },
    )
    return test_rc if test_rc != 0 else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run projection stage training, then final online test, and append results to a markdown report.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--stage_root", type=str, required=True)
        subparser.add_argument("--report_md", type=str, required=True)
        subparser.add_argument("--python_bin", type=str, default=_default_python_bin())
        subparser.add_argument("--torchrun_bin", type=str, default=_default_torchrun_bin())
        subparser.add_argument("--nproc_per_node", type=int, default=1)
        subparser.add_argument("--cuda_visible_devices", type=str, default="")
        subparser.add_argument("--test_dataset", type=str, default="")
        subparser.add_argument("--test_traffic_file", type=str, default="")
        subparser.add_argument("--test_device", type=str, default="auto")
        subparser.add_argument("--test_num_rounds", type=int, default=1)
        subparser.add_argument("--test_proj_name", type=str, default="LLM-TSCS-extreme")
        subparser.add_argument("--test_run_counts", type=int, default=0)
        subparser.add_argument(
            "--test_precision",
            type=str,
            default="follow_train",
            choices=["follow_train", "bf16", "fp16", "none"],
        )
        subparser.add_argument("train_args", nargs=argparse.REMAINDER)

    parser_stage2 = subparsers.add_parser("stage2", help="Run Stage 2 BC training and final test.")
    add_common(parser_stage2)

    parser_stage3 = subparsers.add_parser("stage3", help="Run Stage 3 PPO training and final test.")
    add_common(parser_stage3)
    parser_stage3.add_argument(
        "--pytorch_cuda_alloc_conf",
        type=str,
        default="expandable_segments:True",
    )

    return parser


def main() -> int:
    _configure_stdio()
    wrapper_argv, train_args = _split_wrapper_and_train_args(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(wrapper_argv)
    args.train_args = list(getattr(args, "train_args", [])) + train_args
    stage_name = "Stage 2 BC" if args.stage == "stage2" else "Stage 3 PPO"
    return _run_stage(stage_name, args)


if __name__ == "__main__":
    raise SystemExit(main())
