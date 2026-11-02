"""Fine-tune YOLOv8 on the helmet dataset.

Example:
    python -m saferide.train --data datasets/helmet/data.yaml --model yolov8n.pt --epochs 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_settings
from .violations import normalize


def check_dataset(data_yaml: Path) -> list[str]:
    """Print the dataset classes and warn if none match the configured violation classes."""
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    names = [normalize(n) for n in names]
    print(f"[data] classes: {names}")
    settings = load_settings()
    if not set(names) & {normalize(c) for c in settings.violation_classes}:
        print(
            "[WARN] None of your dataset classes match `violation_classes` in "
            "configs/project.yaml. Edit that file, otherwise no violation will ever be counted."
        )
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt (fast) | yolov8s.pt (more accurate)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (default: auto)")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="helmet_v8n")
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"data.yaml not found: {data}")
    check_dataset(data)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[done] best weights: {best}")


if __name__ == "__main__":
    main()
