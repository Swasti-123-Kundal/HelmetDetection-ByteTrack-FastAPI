"""Detect + track helmet violations in a video and write an annotated copy.

Example:
    python -m saferide.track --model models/best_640.onnx --source samples/traffic.mp4 \
        --out outputs/tracked.mp4 --snapshots outputs/violators
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import load_settings
from .pipeline import load_model, track_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".pt or .onnx")
    ap.add_argument("--source", required=True, help="video file")
    ap.add_argument("--out", default="outputs/tracked.mp4")
    ap.add_argument("--snapshots", default=None, help="folder to save a crop of each confirmed violator")
    ap.add_argument("--config", default=None)
    ap.add_argument("--imgsz", type=int, default=None, help="override; must match ONNX export size")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--tracker", default=None, help="bytetrack.yaml | botsort.yaml")
    ap.add_argument("--violation-classes", nargs="+", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    s = load_settings(args.config)
    overrides = {k: v for k, v in {"imgsz": args.imgsz, "conf": args.conf, "tracker": args.tracker}.items() if v is not None}
    s = replace(s, **overrides)

    model = load_model(args.model)
    summary = track_video(
        model,
        args.source,
        args.out,
        s,
        violation_classes=args.violation_classes,
        snapshot_dir=args.snapshots,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
