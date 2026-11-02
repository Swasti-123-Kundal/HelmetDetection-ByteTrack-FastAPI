"""Export trained weights to ONNX at several input sizes.

ONNX models have a FIXED input size, so each size is a separate file:
    models/best_320.onnx, models/best_416.onnx, models/best_640.onnx

Example:
    python -m saferide.export --weights runs/helmet_v8n/weights/best.pt --sizes 320 416 640
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 416, 640])
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    shutil.copy(args.weights, out_dir / "best.pt")

    for sz in args.sizes:
        exported = model.export(format="onnx", imgsz=sz, opset=args.opset, simplify=True, dynamic=False)
        target = out_dir / f"best_{sz}.onnx"
        shutil.copy(exported, target)  # ultralytics overwrites best.onnx each time
        print(f"[export] {target}")


if __name__ == "__main__":
    main()
