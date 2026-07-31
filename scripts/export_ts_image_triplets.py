#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_DATA_PATH = (
    "records/sim_expert_validation_search/anon_4_4_hangzhou_real_5816/"
    "best/traffic_prompt_series_2s.jsonl"
)
DEFAULT_CHECKPOINT = "tmp_result/stage2_bc/checkpoints/best.pt"
DEFAULT_MOVEMENT_KEYS = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
CHANNEL_NAMES = ("raw", "delta", "trend")


@dataclass
class TrafficSample:
    sample_index: int
    source_path: str
    sequence: list[list[float]]
    date: str
    intersection_id: str
    expert_action: str
    clip_caption: str
    snapshot_path: str
    source_record: dict[str, Any]

    @property
    def total(self) -> float:
        return float(sum(sum(row) for row in self.sequence))

    @property
    def max_value(self) -> float:
        return max((max(row) for row in self.sequence), default=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-sample time-series image triplets: channel maps, "
            "RGB channel composite, and mapped/projector output."
        )
    )
    parser.add_argument("--data-path", nargs="+", default=[DEFAULT_DATA_PATH])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default="outputs/ts_image_triplets")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sample-mode",
        choices=("diverse", "top", "random", "first"),
        default="diverse",
    )
    parser.add_argument("--min-total", type=float, default=10.0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--trend-kernel", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--movement-keys",
        default=",".join(DEFAULT_MOVEMENT_KEYS),
        help="Comma-separated movement key order used for [T,D] sequences.",
    )
    parser.add_argument("--intersection-id", default="")
    parser.add_argument(
        "--require-expert-action",
        action="store_true",
        help="Skip samples without expert_action.",
    )
    return parser.parse_args()


def split_movement_keys(text: str) -> list[str]:
    keys = [item.strip() for item in str(text or "").split(",") if item.strip()]
    return keys or list(DEFAULT_MOVEMENT_KEYS)


def counts_to_vector(counts: dict[str, Any], movement_keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for key in movement_keys:
        try:
            values.append(float(counts.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def build_sequence(
    history: Sequence[list[float]],
    seq_len: int,
    input_dim: int,
) -> list[list[float]]:
    if not history:
        return [[0.0 for _ in range(input_dim)] for _ in range(seq_len)]
    window = [list(row) for row in history[-seq_len:]]
    if len(window) < seq_len:
        pad = [list(window[0]) for _ in range(seq_len - len(window))]
        window = pad + window
    return window


def load_samples(
    paths: Sequence[str],
    seq_len: int,
    movement_keys: Sequence[str],
    intersection_id: str,
    require_expert_action: bool,
) -> list[TrafficSample]:
    samples: list[TrafficSample] = []
    sample_index = 0
    histories: dict[tuple[str, str], list[list[float]]] = {}
    input_dim = len(movement_keys)

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"data path does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_intersection = str(
                    record.get("intersection_id") or record.get("road_id") or ""
                )
                if intersection_id and record_intersection != intersection_id:
                    continue
                expert_action = str(record.get("expert_action") or "")
                if require_expert_action and not expert_action:
                    continue

                key = (str(path), record_intersection)
                histories.setdefault(key, []).append(
                    counts_to_vector(record.get("movement_counts") or {}, movement_keys)
                )
                sequence = build_sequence(histories[key], seq_len, input_dim)
                samples.append(
                    TrafficSample(
                        sample_index=sample_index,
                        source_path=str(path),
                        sequence=sequence,
                        date=str(record.get("date") or ""),
                        intersection_id=record_intersection,
                        expert_action=expert_action,
                        clip_caption=str(record.get("clip_caption") or ""),
                        snapshot_path=str(record.get("snapshot_path") or ""),
                        source_record=record,
                    )
                )
                sample_index += 1
    return samples


def choose_samples(
    samples: Sequence[TrafficSample],
    count: int,
    min_total: float,
    sample_mode: str,
    seed: int,
) -> list[TrafficSample]:
    pool = [sample for sample in samples if sample.total >= min_total]
    if not pool:
        pool = list(samples)
    if sample_mode == "first":
        return pool[:count]
    if sample_mode == "random":
        rng = random.Random(seed)
        pool = list(pool)
        rng.shuffle(pool)
        return pool[:count]
    if sample_mode == "top":
        return sorted(pool, key=lambda item: (item.total, item.max_value), reverse=True)[:count]

    selected: list[TrafficSample] = []
    seen_actions: set[str] = set()
    seen_intersections: set[str] = set()
    ranked = sorted(pool, key=lambda item: (item.total, item.max_value), reverse=True)
    for sample in ranked:
        if sample.intersection_id in seen_intersections:
            continue
        selected.append(sample)
        seen_actions.add(sample.expert_action or "-")
        seen_intersections.add(sample.intersection_id)
        if len(selected) >= count:
            return selected
    for sample in ranked:
        action_key = sample.expert_action or "-"
        if sample in selected or action_key in seen_actions:
            continue
        selected.append(sample)
        seen_actions.add(action_key)
        if len(selected) >= count:
            return selected
    for sample in ranked:
        if sample not in selected:
            selected.append(sample)
        if len(selected) >= count:
            break
    return selected


def moving_average_by_lane(sequence: Sequence[Sequence[float]], kernel: int) -> list[list[float]]:
    seq_len = len(sequence)
    input_dim = len(sequence[0]) if seq_len else 0
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    radius = kernel // 2
    output: list[list[float]] = []
    for t in range(seq_len):
        lo = max(0, t - radius)
        hi = min(seq_len, t + radius + 1)
        denom = max(1, hi - lo)
        row: list[float] = []
        for lane_idx in range(input_dim):
            row.append(sum(sequence[j][lane_idx] for j in range(lo, hi)) / denom)
        output.append(row)
    return output


def compute_channels(
    sequence: Sequence[Sequence[float]],
    trend_kernel: int,
) -> dict[str, list[list[float]]]:
    raw: list[list[float]] = []
    delta: list[list[float]] = []
    trend = moving_average_by_lane(sequence, trend_kernel)
    for t, row in enumerate(sequence):
        raw.append([math.log1p(max(0.0, float(value))) for value in row])
        if t == 0:
            delta.append([0.0 for _ in row])
        else:
            delta.append(
                [float(row[i]) - float(sequence[t - 1][i]) for i in range(len(row))]
            )
    return {"raw": raw, "delta": delta, "trend": trend}


def normalize_grid(grid: Sequence[Sequence[float]], symmetric: bool = False) -> list[list[int]]:
    values = [float(value) for row in grid for value in row]
    if not values:
        return []
    if symmetric:
        bound = max(abs(value) for value in values)
        if bound < 1e-9:
            return [[128 for _ in row] for row in grid]
        return [
            [int(round((float(value) + bound) * 255.0 / (2.0 * bound))) for value in row]
            for row in grid
        ]
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [[0 for _ in row] for row in grid]
    return [
        [int(round((float(value) - lo) * 255.0 / (hi - lo))) for value in row]
        for row in grid
    ]


def grid_to_image(
    grid: Sequence[Sequence[int]],
    image_size: int,
    mode: str = "L",
) -> Image.Image:
    seq_len = len(grid)
    input_dim = len(grid[0]) if seq_len else 0
    image = Image.new(mode, (max(1, seq_len), max(1, input_dim)))
    pixels = image.load()
    for lane_idx in range(input_dim):
        for t in range(seq_len):
            value = int(max(0, min(255, grid[t][lane_idx])))
            pixels[t, lane_idx] = value
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((image_size, image_size), resampling)


def rgb_channel_image(
    channels: dict[str, list[list[float]]],
    image_size: int,
) -> Image.Image:
    normalized = {
        "raw": normalize_grid(channels["raw"]),
        "delta": normalize_grid(channels["delta"], symmetric=True),
        "trend": normalize_grid(channels["trend"]),
    }
    seq_len = len(normalized["raw"])
    input_dim = len(normalized["raw"][0]) if seq_len else 0
    image = Image.new("RGB", (max(1, seq_len), max(1, input_dim)))
    pixels = image.load()
    for lane_idx in range(input_dim):
        for t in range(seq_len):
            pixels[t, lane_idx] = (
                normalized["raw"][t][lane_idx],
                normalized["delta"][t][lane_idx],
                normalized["trend"][t][lane_idx],
            )
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((image_size, image_size), resampling)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def with_label(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    title_font = load_font(16)
    sub_font = load_font(11)
    label_h = 44 if subtitle else 28
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image.convert("RGB"), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=(20, 20, 20), font=title_font)
    if subtitle:
        draw.text((8, 25), subtitle, fill=(70, 70, 70), font=sub_font)
    return canvas


def make_channel_maps(
    channels: dict[str, list[list[float]]],
    image_size: int,
) -> Image.Image:
    panels: list[Image.Image] = []
    for name in CHANNEL_NAMES:
        image = grid_to_image(
            normalize_grid(channels[name], symmetric=(name == "delta")),
            image_size=image_size,
            mode="L",
        ).convert("RGB")
        panels.append(with_label(image, name, "x=time, y=movement"))
    gap = 12
    width = image_size * len(panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def mapped_fallback_image(
    channels: dict[str, list[list[float]]],
    image_size: int,
) -> Image.Image:
    rgb = rgb_channel_image(channels, image_size)
    r, g, b = rgb.split()
    # A deterministic non-learned mapping: blend intensity, change, and trend into a
    # smoother post-map view for environments without torch/checkpoint execution.
    mapped = Image.merge("RGB", (b, r, g))
    return mapped.filter(ImageFilter.SMOOTH_MORE)


def try_render_learned(
    samples: Sequence[TrafficSample],
    checkpoint_path: str,
    image_size: int,
    hidden_dim: int,
    device: str,
) -> tuple[list[Image.Image] | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "renderer": "learned",
        "checkpoint_path": checkpoint_path,
        "checkpoint_loaded": False,
    }
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        info.update({"renderer": "fallback_hot", "reason": f"torch unavailable: {exc}"})
        return None, info

    class LearnableHotLite(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, image_size: int) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.image_size = int(image_size)
            self.trend_dw = nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.input_dim,
                kernel_size=5,
                padding=2,
                groups=self.input_dim,
                bias=False,
            )
            with torch.no_grad():
                self.trend_dw.weight.fill_(1.0 / 5.0)
            self.conv1d = nn.Conv1d(3, self.hidden_dim, kernel_size=3, padding=1)
            self.conv2d_1 = nn.Conv2d(
                self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1
            )
            self.conv2d_2 = nn.Conv2d(self.hidden_dim // 2, 3, kernel_size=3, padding=1)

        def forward(self, x_enc: Any) -> Any:
            batch_size, seq_len, input_dim = x_enc.shape
            x = x_enc.float()
            raw = torch.log1p(torch.clamp(x, min=0.0))
            delta = torch.zeros_like(x)
            delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            trend = self.trend_dw(x.permute(0, 2, 1)).permute(0, 2, 1)
            feat = torch.stack([raw, delta, trend], dim=-1)
            mean = feat.mean(dim=(1, 2), keepdim=True)
            std = feat.std(dim=(1, 2), keepdim=True) + 1e-6
            feat = (feat - mean) / std
            feat = feat.permute(0, 2, 3, 1).reshape(batch_size * input_dim, 3, seq_len)
            feat = F.silu(self.conv1d(feat))
            feat = feat.reshape(batch_size, input_dim, self.hidden_dim, seq_len)
            feat = feat.permute(0, 2, 1, 3)
            feat = F.silu(self.conv2d_1(feat))
            image = self.conv2d_2(feat)
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            return (torch.tanh(image) + 1.0) * 0.5

    class ProjectorLite(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, image_size: int) -> None:
            super().__init__()
            self.image_module = LearnableHotLite(input_dim, hidden_dim, image_size)

        def forward(self, sequences: Any) -> Any:
            return self.image_module(sequences)

    input_dim = len(samples[0].sequence[0]) if samples and samples[0].sequence else 0
    if input_dim <= 0:
        info.update({"renderer": "fallback_hot", "reason": "empty input sequence"})
        return None, info

    try:
        projector = ProjectorLite(input_dim, hidden_dim, image_size)
        path = Path(checkpoint_path)
        if path.exists():
            checkpoint = torch.load(str(path), map_location="cpu")
            state = checkpoint.get("ts_projector") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict) and isinstance(checkpoint, dict):
                raw_state = checkpoint.get("state_dict")
                if isinstance(raw_state, dict):
                    state = {
                        key[len("ts_projector.") :]: value
                        for key, value in raw_state.items()
                        if key.startswith("ts_projector.")
                    }
            if isinstance(state, dict) and state:
                incompatible = projector.load_state_dict(state, strict=False)
                info.update(
                    {
                        "checkpoint_loaded": True,
                        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
                        "unexpected_keys": list(
                            getattr(incompatible, "unexpected_keys", [])
                        ),
                    }
                )
        torch_device = torch.device(device)
        projector.to(torch_device)
        projector.eval()
        sequences = torch.tensor(
            [sample.sequence for sample in samples],
            dtype=torch.float32,
            device=torch_device,
        )
        with torch.no_grad():
            images = projector(sequences).detach().cpu()
        images = (images * 255.0).clamp(0, 255).to(torch.uint8)
        pil_images: list[Image.Image] = []
        for tensor in images:
            flat = tensor[:3].permute(1, 2, 0).contiguous().view(-1).tolist()
            pil_images.append(Image.frombytes("RGB", (image_size, image_size), bytes(flat)))
        return pil_images, info
    except Exception as exc:
        info.update({"renderer": "fallback_hot", "reason": str(exc)})
        return None, info


def save_sample_outputs(
    sample: TrafficSample,
    mapped_image: Image.Image,
    mapped_info: dict[str, Any],
    output_dir: Path,
    image_size: int,
    movement_keys: Sequence[str],
    trend_kernel: int,
) -> dict[str, Any]:
    channels = compute_channels(sample.sequence, trend_kernel)
    sample_dir = output_dir / f"sample_{sample.sample_index:05d}_{sample.intersection_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    channel_maps = make_channel_maps(channels, image_size)
    rgb = with_label(
        rgb_channel_image(channels, image_size),
        "rgb channels",
        "R=raw, G=delta, B=trend",
    )
    mapped = with_label(
        mapped_image.convert("RGB"),
        "mapped after projector" if mapped_info.get("renderer") == "learned" else "mapped fallback",
        str(mapped_info.get("reason") or "checkpoint projector output"),
    )

    channel_path = sample_dir / "01_channel_maps.png"
    rgb_path = sample_dir / "02_rgb_channels.png"
    mapped_path = sample_dir / "03_mapped_after.png"
    channel_maps.save(channel_path)
    rgb.save(rgb_path)
    mapped.save(mapped_path)

    metadata = {
        "sample_index": sample.sample_index,
        "source_path": sample.source_path,
        "date": sample.date,
        "intersection_id": sample.intersection_id,
        "expert_action": sample.expert_action,
        "clip_caption": sample.clip_caption,
        "snapshot_path": sample.snapshot_path,
        "movement_keys": list(movement_keys),
        "sequence_shape": [
            len(sample.sequence),
            len(sample.sequence[0]) if sample.sequence else 0,
        ],
        "sequence_sum": sample.total,
        "sequence_max": sample.max_value,
        "mapped_info": mapped_info,
        "images": {
            "channel_maps": str(channel_path),
            "rgb_channels": str(rgb_path),
            "mapped_after": str(mapped_path),
        },
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def make_contact_sheet(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    if not records:
        return
    thumb = 180
    label_h = 48
    cols = 3
    rows = len(records)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(11)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    keys = ("channel_maps", "rgb_channels", "mapped_after")
    for row, record in enumerate(records):
        y = row * (thumb + label_h)
        label = (
            f"sample {record['sample_index']} {record['intersection_id']} "
            f"sum={record['sequence_sum']:.1f} action={record['expert_action'] or '-'}"
        )
        draw.text((6, y + thumb + 5), label, fill=(20, 20, 20), font=font)
        for col, key in enumerate(keys):
            image = Image.open(record["images"][key]).convert("RGB")
            image.thumbnail((thumb, thumb), resampling)
            tile = Image.new("RGB", (thumb, thumb), "white")
            tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
            sheet.paste(tile, (col * thumb, y))
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    movement_keys = split_movement_keys(args.movement_keys)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        paths=args.data_path,
        seq_len=args.seq_len,
        movement_keys=movement_keys,
        intersection_id=args.intersection_id,
        require_expert_action=bool(args.require_expert_action),
    )
    selected = choose_samples(
        samples,
        count=max(1, int(args.count)),
        min_total=float(args.min_total),
        sample_mode=args.sample_mode,
        seed=int(args.seed),
    )
    if not selected:
        raise SystemExit("No samples matched the requested filters.")

    learned_images, mapped_info = try_render_learned(
        selected,
        checkpoint_path=str(args.checkpoint or ""),
        image_size=int(args.image_size),
        hidden_dim=int(args.hidden_dim),
        device=str(args.device),
    )
    if learned_images is None:
        learned_images = [
            mapped_fallback_image(compute_channels(sample.sequence, args.trend_kernel), args.image_size)
            for sample in selected
        ]

    records: list[dict[str, Any]] = []
    for sample, mapped_image in zip(selected, learned_images):
        records.append(
            save_sample_outputs(
                sample=sample,
                mapped_image=mapped_image,
                mapped_info=mapped_info,
                output_dir=output_dir,
                image_size=int(args.image_size),
                movement_keys=movement_keys,
                trend_kernel=int(args.trend_kernel),
            )
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    make_contact_sheet(records, output_dir / "contact_sheet.png")
    summary = {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "num_samples": len(records),
        "mapped_info": mapped_info,
        "movement_keys": movement_keys,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"exported {len(records)} sample triplets to {output_dir}")
    print(f"manifest={manifest_path}")
    print(f"mapped_renderer={mapped_info.get('renderer')}")
    if mapped_info.get("reason"):
        print(f"mapped_note={mapped_info.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
