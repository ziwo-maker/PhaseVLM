#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"Pillow is required to export PNG images: {exc}") from exc


DEFAULT_MOVEMENT_KEYS = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
DEFAULT_ACTION_TAGS = ("ETWT", "NTST", "ELWL", "NLSL")
DEFAULT_DATA_PATH = (
    "records/sim_expert_validation_search/anon_4_4_hangzhou_real_5816/"
    "best/traffic_prompt_series_2s.jsonl"
)
DEFAULT_CHECKPOINT = "tmp_result/stage2_bc/checkpoints/best.pt"


@dataclass
class TrafficSample:
    sample_index: int
    source_path: str
    sequence: list[list[float]]
    raw_prompt: str
    clip_caption: str
    expert_action: str
    date: str
    intersection_id: str
    snapshot_path: str
    meta: dict[str, Any]
    source_record: dict[str, Any]

    @property
    def total(self) -> float:
        return float(sum(sum(row) for row in self.sequence))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the time-series images that are passed to the VLM side of "
            "PrefixActionPolicy. With torch available, this reproduces the "
            "TimeSeriesProjector output before encode_vision_tokens()."
        )
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=[DEFAULT_DATA_PATH],
        help="One or more traffic jsonl files to sample from.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Optional stage2/stage3 checkpoint containing ts_projector weights. "
            "Use an empty string together with --no-checkpoint to disable."
        ),
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Do not load ts_projector weights even if --checkpoint exists.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llm_input_images",
        help="Directory for PNG images and manifest.jsonl.",
    )
    parser.add_argument("--count", type=int, default=6, help="Number of samples to export.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed when --sample-mode=random.")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="How to choose samples after filtering.",
    )
    parser.add_argument(
        "--min-total",
        type=float,
        default=1.0,
        help="Prefer samples whose sequence sum is at least this value.",
    )
    parser.add_argument("--intersection-id", default="", help="Optional intersection filter.")
    parser.add_argument(
        "--require-expert-action",
        action="store_true",
        help="Skip samples without an expert_action field.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length. Defaults to checkpoint args when available, otherwise 16.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Output image size. Defaults to checkpoint args when available, otherwise 224.",
    )
    parser.add_argument(
        "--periodicity",
        type=int,
        default=None,
        help="Projector periodicity. Defaults to checkpoint args when available, otherwise 16.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Number of movement dimensions. Defaults to len(--movement-keys).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Projector hidden dim. Defaults to checkpoint args when available, otherwise 64.",
    )
    parser.add_argument(
        "--movement-keys",
        default=",".join(DEFAULT_MOVEMENT_KEYS),
        help="Comma-separated movement key order used to build [T,D] sequences.",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "learned", "hot"),
        default="auto",
        help=(
            "auto: use torch learned projector when possible, otherwise hot; "
            "learned: require torch; hot: pure-PIL raw/delta/trend visualization."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("raw", "image_only_short"),
        default="image_only_short",
        help="Prompt text saved in the manifest for reproducing the LLM call.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        default=True,
        help="Also write contact_sheet.png for human inspection.",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_false",
        dest="contact_sheet",
        help="Skip contact sheet generation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for learned rendering. Use cpu to avoid touching the LLM/GPU.",
    )
    return parser.parse_args()


def split_movement_keys(text: str) -> list[str]:
    keys = [item.strip() for item in str(text or "").split(",") if item.strip()]
    if not keys:
        keys = list(DEFAULT_MOVEMENT_KEYS)
    return keys


def format_action_tags(action_tags: Sequence[str] = DEFAULT_ACTION_TAGS) -> str:
    tags = [str(tag).strip() for tag in action_tags if str(tag).strip()]
    return ", ".join(f"<signal>{tag}</signal>" for tag in tags)


def build_task_prompt(raw_prompt: str, mode: str) -> str:
    if mode == "raw":
        return str(raw_prompt or "").rstrip()
    allowed = format_action_tags()
    return (
        "You are controlling a four-phase traffic signal.\n"
        "The attached image is the only reliable observation of recent traffic dynamics at one intersection.\n"
        "Choose the single best next phase to reduce near-term congestion.\n"
        f"Allowed outputs: {allowed}.\n"
        "Return exactly one <signal>...</signal> tag and nothing else."
    )


def counts_to_vector(counts: dict[str, Any], movement_keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for key in movement_keys:
        raw = counts.get(key, 0.0)
        try:
            values.append(float(raw or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def build_sequence(
    history: Sequence[dict[str, Any]],
    seq_len: int,
    movement_keys: Sequence[str],
) -> list[list[float]]:
    if not history:
        return [[0.0 for _ in movement_keys] for _ in range(seq_len)]
    window = list(history[-seq_len:])
    if len(window) < seq_len:
        pad_item = window[0]
        window = [pad_item for _ in range(seq_len - len(window))] + window
    return [counts_to_vector(item, movement_keys) for item in window]


def load_samples(
    paths: Sequence[str],
    seq_len: int,
    movement_keys: Sequence[str],
    intersection_id: str = "",
    require_expert_action: bool = False,
) -> list[TrafficSample]:
    samples: list[TrafficSample] = []
    sample_index = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"data path does not exist: {path}")
        history: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                raw_intersection_id = record.get("intersection_id") or record.get("road_id") or ""
                record_intersection_id = str(raw_intersection_id)
                if intersection_id and record_intersection_id != intersection_id:
                    history = []
                    continue

                history.append(record.get("movement_counts") or {})
                raw_prompt = str(record.get("prompt") or "")
                expert_action = str(record.get("expert_action") or "")
                clip_caption = str(record.get("clip_caption") or "")
                if not (raw_prompt or expert_action or clip_caption):
                    continue
                if require_expert_action and not expert_action:
                    history = []
                    continue

                samples.append(
                    TrafficSample(
                        sample_index=sample_index,
                        source_path=str(path),
                        sequence=build_sequence(history, seq_len, movement_keys),
                        raw_prompt=raw_prompt,
                        clip_caption=clip_caption,
                        expert_action=expert_action,
                        date=str(record.get("date") or ""),
                        intersection_id=record_intersection_id,
                        snapshot_path=str(record.get("snapshot_path") or ""),
                        meta=record.get("meta") or {},
                        source_record=record,
                    )
                )
                sample_index += 1
                history = []
    return samples


def choose_samples(
    samples: Sequence[TrafficSample],
    count: int,
    min_total: float,
    sample_mode: str,
    seed: int,
) -> list[TrafficSample]:
    if not samples:
        return []
    preferred = [sample for sample in samples if sample.total >= float(min_total)]
    pool = preferred if len(preferred) >= min(count, len(samples)) else list(samples)
    if sample_mode == "random":
        rng = random.Random(seed)
        pool = list(pool)
        rng.shuffle(pool)
    return list(pool[: max(0, int(count))])


def try_load_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        return None, None, None, exc
    return torch, nn, F, None


def make_torch_projector(torch: Any, nn: Any, F: Any):
    class LearnableTimeSeriesToImageHot(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            image_size: int,
            trend_kernel: int = 5,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.image_size = int(image_size)
            if self.input_dim <= 0:
                raise ValueError(f"input_dim must be positive, got {self.input_dim}")
            if self.hidden_dim < 2:
                raise ValueError(f"hidden_dim must be >= 2, got {self.hidden_dim}")
            if self.image_size <= 0:
                raise ValueError(f"image_size must be positive, got {self.image_size}")
            k = int(trend_kernel)
            if k <= 0:
                raise ValueError(f"trend_kernel must be positive, got {k}")
            if k % 2 == 0:
                k += 1
            self.trend_dw = nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.input_dim,
                kernel_size=k,
                padding=k // 2,
                groups=self.input_dim,
                bias=False,
            )
            with torch.no_grad():
                self.trend_dw.weight.fill_(1.0 / k)
            self.conv1d = nn.Conv1d(
                in_channels=3,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            )
            self.conv2d_1 = nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim // 2,
                kernel_size=3,
                padding=1,
            )
            self.conv2d_2 = nn.Conv2d(
                in_channels=self.hidden_dim // 2,
                out_channels=3,
                kernel_size=3,
                padding=1,
            )

        def forward(self, x_enc: Any) -> Any:
            if x_enc.dim() != 3:
                raise ValueError(f"x_enc must be [B,L,D], got {tuple(x_enc.shape)}")
            batch_size, seq_len, input_dim = x_enc.shape
            if input_dim != self.input_dim:
                raise ValueError(
                    f"input_dim={self.input_dim} but sequence dimension is {input_dim}"
                )
            x = x_enc.float()
            eps = 1e-6
            autocast_device = "cuda" if x.is_cuda else "cpu"
            with torch.amp.autocast(autocast_device, enabled=False):
                raw = torch.log1p(torch.clamp(x, min=0.0))
                delta = torch.zeros_like(x)
                delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
                trend = self.trend_dw(x.permute(0, 2, 1)).permute(0, 2, 1)
                feat = torch.stack([raw, delta, trend], dim=-1)
                mean = feat.mean(dim=(1, 2), keepdim=True)
                std = feat.std(dim=(1, 2), keepdim=True) + eps
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
                image = torch.tanh(image)
                return (image + 1.0) * 0.5

    class TimeSeriesProjectorLite(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, image_size: int) -> None:
            super().__init__()
            self.image_module = LearnableTimeSeriesToImageHot(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                image_size=image_size,
            )

        def forward(self, sequences: Any) -> Any:
            return self.image_module(sequences)

    return TimeSeriesProjectorLite


def load_checkpoint_args(torch: Any, checkpoint_path: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not checkpoint_path:
        return {}, None
    path = Path(checkpoint_path)
    if not path.exists():
        return {}, None
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    args = checkpoint.get("args") or {}
    if hasattr(args, "__dict__"):
        args = vars(args)
    if not isinstance(args, dict):
        args = {}
    return dict(args), checkpoint


def resolve_int(
    explicit_value: int | None,
    checkpoint_args: dict[str, Any],
    key: str,
    fallback: int,
) -> int:
    if explicit_value is not None and int(explicit_value) > 0:
        return int(explicit_value)
    raw_value = checkpoint_args.get(key)
    try:
        value = int(raw_value)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return int(fallback)


def tensor_to_uint8_list(torch: Any, images: Any) -> list[Image.Image]:
    images = images.detach().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(images.shape)}")
    images = (images * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    pil_images: list[Image.Image] = []
    for image_tensor in images:
        height = int(image_tensor.shape[-2])
        width = int(image_tensor.shape[-1])
        if int(image_tensor.shape[0]) == 1:
            flat = image_tensor.squeeze(0).contiguous().view(-1).tolist()
            pil_images.append(Image.frombytes("L", (width, height), bytes(flat)))
        else:
            flat = image_tensor[:3].permute(1, 2, 0).contiguous().view(-1).tolist()
            pil_images.append(Image.frombytes("RGB", (width, height), bytes(flat)))
    return pil_images


def render_learned_images(
    samples: Sequence[TrafficSample],
    checkpoint: dict[str, Any] | None,
    runtime: dict[str, int],
    device: str,
) -> tuple[list[Image.Image], dict[str, Any]]:
    torch, nn, F, import_error = try_load_torch()
    if import_error is not None:
        raise RuntimeError(f"torch is not available: {import_error}") from import_error
    projector_cls = make_torch_projector(torch, nn, F)
    projector = projector_cls(
        input_dim=runtime["input_dim"],
        hidden_dim=runtime["hidden_dim"],
        image_size=runtime["image_size"],
    )
    load_info: dict[str, Any] = {"checkpoint_loaded": False}
    if checkpoint is not None:
        state = checkpoint.get("ts_projector")
        if not isinstance(state, dict) and isinstance(checkpoint.get("state_dict"), dict):
            state = {
                key[len("ts_projector.") :]: value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("ts_projector.")
            }
        if isinstance(state, dict) and state:
            incompatible = projector.load_state_dict(state, strict=False)
            load_info = {
                "checkpoint_loaded": True,
                "missing_keys": list(getattr(incompatible, "missing_keys", [])),
                "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
            }
    projector.eval()
    torch_device = torch.device(device)
    projector.to(torch_device)
    sequences = torch.tensor(
        [sample.sequence for sample in samples],
        dtype=torch.float32,
        device=torch_device,
    )
    with torch.no_grad():
        images = projector(sequences)
    return tensor_to_uint8_list(torch, images), load_info


def signed_log1p(value: float) -> float:
    if value == 0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def normalize_grid(grid: Sequence[Sequence[float]]) -> list[list[int]]:
    values = [float(value) for row in grid for value in row]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [[0 for _ in row] for row in grid]
    return [[int(round((float(value) - lo) * 255.0 / (hi - lo))) for value in row] for row in grid]


def moving_average_by_lane(sequence: Sequence[Sequence[float]], kernel: int = 5) -> list[list[float]]:
    seq_len = len(sequence)
    input_dim = len(sequence[0]) if seq_len else 0
    radius = max(0, int(kernel) // 2)
    output: list[list[float]] = []
    for t in range(seq_len):
        row: list[float] = []
        lo = max(0, t - radius)
        hi = min(seq_len, t + radius + 1)
        denom = max(1, hi - lo)
        for lane_idx in range(input_dim):
            row.append(sum(sequence[j][lane_idx] for j in range(lo, hi)) / denom)
        output.append(row)
    return output


def render_hot_image(
    sequence: Sequence[Sequence[float]],
    image_size: int,
) -> Image.Image:
    seq_len = len(sequence)
    input_dim = len(sequence[0]) if seq_len else 0
    raw: list[list[float]] = []
    delta: list[list[float]] = []
    trend = moving_average_by_lane(sequence)
    for t, row in enumerate(sequence):
        raw.append([math.log1p(max(0.0, float(value))) for value in row])
        if t == 0:
            delta.append([0.0 for _ in row])
        else:
            delta.append(
                [
                    signed_log1p(float(row[i]) - float(sequence[t - 1][i]))
                    for i in range(len(row))
                ]
            )
    raw_u8 = normalize_grid(raw)
    delta_u8 = normalize_grid(delta)
    trend_u8 = normalize_grid(trend)

    image = Image.new("RGB", (max(1, seq_len), max(1, input_dim)))
    pixels = image.load()
    for lane_idx in range(input_dim):
        for t in range(seq_len):
            pixels[t, lane_idx] = (
                raw_u8[t][lane_idx],
                delta_u8[t][lane_idx],
                trend_u8[t][lane_idx],
            )
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((image_size, image_size), resampling)


def make_contact_sheet(
    images: Sequence[Image.Image],
    samples: Sequence[TrafficSample],
    output_path: Path,
) -> None:
    if not images:
        return
    thumb_size = 180
    label_h = 42
    cols = min(3, len(images))
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    for idx, (image, sample) in enumerate(zip(images, samples)):
        x = (idx % cols) * thumb_size
        y = (idx // cols) * (thumb_size + label_h)
        sheet.paste(image.resize((thumb_size, thumb_size), resampling), (x, y))
        label = (
            f"#{sample.sample_index} {sample.intersection_id}\n"
            f"sum={sample.total:.1f} action={sample.expert_action or '-'}"
        )
        draw.text((x + 6, y + thumb_size + 4), label, fill=(20, 20, 20))
    sheet.save(output_path)


def write_outputs(
    images: Sequence[Image.Image],
    samples: Sequence[TrafficSample],
    output_dir: Path,
    renderer: str,
    runtime: dict[str, int],
    movement_keys: Sequence[str],
    prompt_mode: str,
    render_info: dict[str, Any],
    contact_sheet: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    image_records: list[dict[str, Any]] = []
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, (image, sample) in enumerate(zip(images, samples)):
            image_path = output_dir / f"llm_input_{idx:03d}_sample_{sample.sample_index}.png"
            image.save(image_path)
            record = {
                "image_path": str(image_path),
                "renderer": renderer,
                "is_exact_projector_input": renderer == "learned",
                "sample_index": sample.sample_index,
                "source_path": sample.source_path,
                "date": sample.date,
                "intersection_id": sample.intersection_id,
                "expert_action": sample.expert_action,
                "clip_caption": sample.clip_caption,
                "snapshot_path": sample.snapshot_path,
                "movement_keys": list(movement_keys),
                "sequence_shape": [len(sample.sequence), len(sample.sequence[0]) if sample.sequence else 0],
                "sequence_sum": sample.total,
                "sequence_min": min((min(row) for row in sample.sequence), default=0.0),
                "sequence_max": max((max(row) for row in sample.sequence), default=0.0),
                "runtime": runtime,
                "render_info": render_info,
                "policy_prompt": build_task_prompt(sample.raw_prompt, mode=prompt_mode),
                "raw_prompt_preview": sample.raw_prompt[:240],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            image_records.append(record)
    if contact_sheet:
        make_contact_sheet(images, samples, output_dir / "contact_sheet.png")
    summary_path = output_dir / "summary.json"
    summary = {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "num_images": len(images),
        "renderer": renderer,
        "runtime": runtime,
        "render_info": render_info,
        "images": [record["image_path"] for record in image_records],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    movement_keys = split_movement_keys(args.movement_keys)
    checkpoint_path = "" if args.no_checkpoint else str(args.checkpoint or "").strip()

    torch, _, _, torch_error = try_load_torch()
    checkpoint_args: dict[str, Any] = {}
    checkpoint: dict[str, Any] | None = None
    checkpoint_error = ""
    if torch is not None and checkpoint_path and Path(checkpoint_path).exists():
        try:
            checkpoint_args, checkpoint = load_checkpoint_args(torch, checkpoint_path)
        except Exception as exc:
            checkpoint_error = str(exc)

    runtime = {
        "seq_len": resolve_int(args.seq_len, checkpoint_args, "seq_len", 16),
        "image_size": resolve_int(args.image_size, checkpoint_args, "image_size", 224),
        "periodicity": resolve_int(args.periodicity, checkpoint_args, "periodicity", 16),
        "input_dim": resolve_int(args.input_dim, checkpoint_args, "input_dim", len(movement_keys)),
        "hidden_dim": resolve_int(args.hidden_dim, checkpoint_args, "hidden_dim", 64),
    }
    if runtime["input_dim"] != len(movement_keys):
        raise SystemExit(
            f"input_dim={runtime['input_dim']} but movement key count is {len(movement_keys)}"
        )

    samples = load_samples(
        args.data_path,
        seq_len=runtime["seq_len"],
        movement_keys=movement_keys,
        intersection_id=args.intersection_id,
        require_expert_action=bool(args.require_expert_action),
    )
    selected = choose_samples(
        samples,
        count=args.count,
        min_total=args.min_total,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("No samples matched the requested filters.")

    renderer = args.renderer
    render_info: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_error": checkpoint_error,
    }
    if renderer == "auto":
        renderer = "learned" if torch is not None else "hot"

    if renderer == "learned":
        try:
            images, learned_info = render_learned_images(
                selected,
                checkpoint=checkpoint,
                runtime=runtime,
                device=args.device,
            )
            render_info.update(learned_info)
            if torch_error is not None:
                render_info["torch_probe_error"] = str(torch_error)
        except Exception as exc:
            if args.renderer == "learned":
                raise SystemExit(f"Failed to render learned projector images: {exc}") from exc
            renderer = "hot"
            render_info["learned_fallback_reason"] = str(exc)
            images = [render_hot_image(sample.sequence, runtime["image_size"]) for sample in selected]
    else:
        if torch_error is not None:
            render_info["torch_probe_error"] = str(torch_error)
        images = [render_hot_image(sample.sequence, runtime["image_size"]) for sample in selected]

    write_outputs(
        images=images,
        samples=selected,
        output_dir=Path(args.output_dir),
        renderer=renderer,
        runtime=runtime,
        movement_keys=movement_keys,
        prompt_mode=args.prompt_mode,
        render_info=render_info,
        contact_sheet=bool(args.contact_sheet),
    )
    print(f"exported {len(images)} images to {args.output_dir}")
    print(f"renderer={renderer}")
    print(f"manifest={Path(args.output_dir) / 'manifest.jsonl'}")
    if renderer != "learned":
        print("note: renderer=hot is a visualization fallback, not the trained projector tensor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
