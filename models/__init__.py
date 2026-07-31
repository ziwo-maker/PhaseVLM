from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

_ALIGNMENT_EXPORTS = {
    "AlignmentModel",
    "Qwen3VLBackbone",
    "TimeSeriesProjector",
    "VisionPrefixProjector",
}
_alignment_module: Optional[ModuleType] = None


def _load_alignment_module() -> ModuleType:
    global _alignment_module
    if _alignment_module is not None:
        return _alignment_module
    root = Path(__file__).resolve().parents[1]
    module_path = root / "models.py"
    spec = importlib.util.spec_from_file_location("_alignment_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load alignment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _alignment_module = module
    return module


def __getattr__(name: str) -> Any:
    if name in _ALIGNMENT_EXPORTS:
        module = _load_alignment_module()
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _ALIGNMENT_EXPORTS)


__all__ = sorted(_ALIGNMENT_EXPORTS)
