#!/usr/bin/env python3
"""Prepare mismatched-caption negative control and aggregate one-flow Stage-I study."""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

ARMS = ("full", "without_stage1", "shuffled_alignment")
EPOCHS = (1, 3, 5, 10, 15)
METRICS = {"ATT": "test_avg_travel_time_over",
           "AQL": "test_avg_queue_len_over",
           "AWT": "test_avg_waiting_time_over"}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def shuffle(source, target, info_path, seed):
    import random
    rng = random.Random(seed)
    records = []
    indices, captions = [], []
    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r.get("clip_caption") and str(r.get("expert_action") or "").strip():
                indices.append(len(records))
                captions.append(str(r["clip_caption"]))
            records.append(r)
    if len(captions) < 10:
        raise RuntimeError("Insufficient captioned decision samples")
    # Rotation guarantees no identical row pairing, and preserves the caption distribution.
    candidates = [rng.randrange(1, len(captions)) for _ in range(64)]
    shift = min(candidates, key=lambda s: sum(
        captions[i] == captions[(i + s) % len(captions)] for i in range(len(captions))))
    shuffled = captions[shift:] + captions[:shift]
    unchanged = sum(a == b for a, b in zip(captions, shuffled))
    for idx, caption in zip(indices, shuffled):
        records[idx]["clip_caption"] = caption
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with Path(target).open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"source": str(source), "source_sha256": sha256(source),
            "output": str(target), "output_sha256": sha256(target), "seed": seed,
            "total_rows": len(records), "captioned_decision_rows": len(indices),
            "rotation": shift, "same_caption_after_shuffle": unchanged,
            "same_caption_fraction": unchanged / len(captions),
            "all_other_fields_unchanged": True,
            "method": "fixed cyclic rotation across captioned decision samples"}
    Path(info_path).write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(info, ensure_ascii=False))

def parse_timing(log_path):
    if not Path(log_path).exists():
        return {}
    started = None
    elapsed = {}
    for line in Path(log_path).read_text(errors="replace").splitlines():
        m = re.search(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?Epoch (\d+)/15 started", line)
        if m and started is None:
            started = datetime.fromisoformat(m.group(1))
        m = re.search(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?\[epoch (\d+)\] val_top1_cost_regret=", line)
        if m and started is not None:
            elapsed[int(m.group(2))] = (datetime.fromisoformat(m.group(1)) - started).total_seconds()
    return elapsed

def summarize(root):
    root = Path(root)
    run_manifest = json.loads((root / "manifest.json").read_text())
    first_stage_seconds = {}
    for arm in ("full", "shuffled_alignment"):
        p = root / arm / "stage1" / "run_summary.json"
        if p.exists():
            first_stage_seconds[arm] = json.loads(p.read_text()).get("duration_seconds", 0)
    records = []
    for arm in ARMS:
        p = root / arm / "stage2" / "metrics.jsonl"
        if not p.exists():
            continue
        timing = parse_timing(root / "logs" / f"stage2_{arm}.log")
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            ep = int(entry["epoch"])
            stage2_seconds = timing.get(ep)
            stage1_seconds = first_stage_seconds.get(arm, 0)
            records.append({
                "arm": arm, "epoch": ep,
                "val_macro_f1": entry.get("val_macro_f1"),
                "val_top1_cost_regret": entry.get("val_top1_cost_regret"),
                "val_expected_cost_regret": entry.get("val_expected_cost_regret"),
                "stage2_elapsed_seconds": stage2_seconds,
                "stage1_seconds": stage1_seconds,
                "end_to_end_gpu_hours": ((stage1_seconds + stage2_seconds) *
                       len(run_manifest["gpu_ids"]) / 3600.0) if stage2_seconds is not None else None,
            })
    with (root / "convergence.csv").open("w", newline="") as stream:
        columns = ["arm", "epoch", "val_macro_f1", "val_top1_cost_regret",
                   "val_expected_cost_regret", "stage2_elapsed_seconds",
                   "stage1_seconds", "end_to_end_gpu_hours"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    online = []
    for arm in ARMS:
        for ep in EPOCHS:
            result_path = root / arm / "online" / f"epoch_{ep}" / "result.json"
            if not result_path.is_file():
                continue
            data = json.loads(result_path.read_text())
            result = data["results"]
            if isinstance(result, list):
                result = result[-1]
            r = {"arm": arm, "epoch": ep, "seed": data.get("args", {}).get("seed")}
            r.update({k: result.get(v) for k, v in METRICS.items()})
            offline = next((x for x in records if x["arm"] == arm and x["epoch"] == ep), None)
            if offline:
                r["end_to_end_gpu_hours"] = offline["end_to_end_gpu_hours"]
            online.append(r)
    with (root / "online_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "epoch", "seed", "ATT",
                                    "AQL", "AWT", "end_to_end_gpu_hours"])
        writer.writeheader()
        writer.writerows(online)
    targets = {}
    for arm in ARMS:
        group = sorted([r for r in records if r["arm"] == arm], key=lambda r: r["epoch"])
        targets[arm] = {}
        for threshold in (0.25, 0.20):
            hit = next((r for r in group if r["val_top1_cost_regret"] is not None
                        and r["val_top1_cost_regret"] <= threshold), None)
            targets[arm][str(threshold)] = hit
    payload = {"dataset": run_manifest["dataset"], "seed": run_manifest["seed"],
               "gpu_ids": run_manifest["gpu_ids"], "stage1_seconds": first_stage_seconds,
               "target_regret_first_reached": targets,
               "offline_epoch_rows": len(records), "online_result_rows": len(online),
               "expected_offline_rows": 45, "expected_online_rows": 15,
               "note": "One training seed; timing and performance estimates are descriptive."}
    (root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    status_lines = ["# Stage I Convergence Efficiency: Jinan-1 / seed 2024", "",
      "Three-arm controlled comparison. Lower cost regret, ATT, AQL, AWT are better.",
      "One training seed does not establish statistical significance.", "",
      "## Training progress", "",
      "| Arm | Epochs | Final validation regret | First epoch regret <= 0.25 | Stage I seconds |",
      "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        group = sorted([r for r in records if r["arm"] == arm], key=lambda r: r["epoch"])
        last = group[-1] if group else None
        hit = targets[arm]["0.25"]
        status_lines.append(f"| {arm} | {len(group)} | {last['val_top1_cost_regret']:.4f} "
                  f"| {hit['epoch'] if hit else 'not reached'} | {first_stage_seconds.get(arm, 0):.1f} |"
                  if last else f"| {arm} | 0 | pending | pending | pending |")
    status_lines += ["", "## Online tests", "",
        "| Arm | Epoch | ATT | AQL | AWT |",
        "|---|---:|---:|---:|---:|"]
    for r in online:
        status_lines.append(f"| {r['arm']} | {r['epoch']} | {r['ATT']:.3f} "
                            f"| {r['AQL']:.3f} | {r['AWT']:.3f} |")
    status_lines += ["", "Stage I cost is included in end-to-end GPU-hours.",
                    "Online evaluations are held out from checkpoint selection.",
                    "The learning rate schedule is fixed for 15 epochs in all arms.", ""]
    (root / "REPORT.md").write_text("\n".join(status_lines))
    print(json.dumps(payload, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    sh = modes.add_parser("shuffle")
    for key in ("source", "target", "info"):
        sh.add_argument("--" + key, required=True)
    sh.add_argument("--seed", type=int, default=99173)
    sm = modes.add_parser("summarize")
    sm.add_argument("--root", required=True)
    opts = parser.parse_args()
    if opts.mode == "shuffle":
        shuffle(opts.source, opts.target, opts.info, opts.seed)
    else:
        summarize(opts.root)
