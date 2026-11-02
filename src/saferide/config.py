"""Settings loader. Reads configs/project.yaml (or the file in $SAFERIDE_CONFIG)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

_DEFAULT_PATHS = [
    Path(os.getenv("SAFERIDE_CONFIG", "")) if os.getenv("SAFERIDE_CONFIG") else None,
    Path.cwd() / "configs" / "project.yaml",
    Path(__file__).resolve().parents[2] / "configs" / "project.yaml",
]


@dataclass
class Settings:
    violation_classes: list[str] = field(
        default_factory=lambda: ["without_helmet", "no_helmet"]
    )
    safe_classes: list[str] = field(default_factory=lambda: ["with_helmet", "helmet"])
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    min_violation_frames: int = 5
    smoothing_window: int = 15


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load settings from YAML; fall back to defaults if no file is found."""
    candidates = [Path(path)] if path else [p for p in _DEFAULT_PATHS if p]
    for p in candidates:
        if p.is_file():
            raw = yaml.safe_load(p.read_text()) or {}
            valid = {f.name for f in fields(Settings)}
            return Settings(**{k: v for k, v in raw.items() if k in valid})
    return Settings()
