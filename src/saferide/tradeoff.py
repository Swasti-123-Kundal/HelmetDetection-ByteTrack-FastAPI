"""Latency vs accuracy trade-off: PyTorch vs ONNX at several input sizes.

For every (backend, imgsz) it measures
  * accuracy  : mAP@0.5 and mAP@0.5:0.95 on the validation split
  * speed     : mean / p50 / p95 end-to-end latency per image and FPS (single image, CPU)
and writes results/tradeoff.csv, results/tradeoff.md and results/tradeoff.png.

Example:
    python -m saferide.tradeoff --weights runs/helmet_v8n/weights/best.pt \
        --data datasets/helmet/data.yaml --sizes 320 416 640
"""
from __future__ import annotations

import argparse
import csv
import itertools
import platform
import statistics
import time
from pathlib import Path

import cv2
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _val_images(data_yaml: Path) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val = root / cfg["val"]
    files = sorted(p for p in val.rglob("*") if p.suffix.lower() in IMG_EXT) if val.is_dir() else []
    if not files:
        raise SystemExit(f"No validation images found under {val}")
    return files


def _bench(model, frames, imgsz: int, runs: int, warmup: int) -> dict:
    cyc = itertools.cycle(frames)
    for _ in range(warmup):
        model.predict(next(cyc), imgsz=imgsz, device="cpu", verbose=False)
    e2e, infer = [], []
    for _ in range(runs):
        f = next(cyc)
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=imgsz, device="cpu", verbose=False)[0]
        e2e.append((time.perf_counter() - t0) * 1000)
        infer.append(r.speed["inference"])
    e2e_sorted = sorted(e2e)
    mean = statistics.fmean(e2e)
    return {
        "latency_mean_ms": round(mean, 2),
        "latency_p50_ms": round(statistics.median(e2e), 2),
        "latency_p95_ms": round(e2e_sorted[int(0.95 * (len(e2e_sorted) - 1))], 2),
        "model_only_ms": round(statistics.fmean(infer), 2),
        "fps": round(1000.0 / mean, 2),
    }


def _plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, marker in (("pytorch", "o"), ("onnx", "s")):
        pts = [r for r in rows if r["backend"] == backend]
        ax.plot([r["latency_mean_ms"] for r in pts], [r["map50_95"] for r in pts], marker=marker, label=backend)
        for r in pts:
            ax.annotate(str(r["imgsz"]), (r["latency_mean_ms"], r["map50_95"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Latency per image on CPU (ms)  <- faster")
    ax.set_ylabel("mAP@0.5:0.95  (higher = better)")
    ax.set_title("Latency vs accuracy (labels = input size)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained .pt file")
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 416, 640])
    ap.add_argument("--runs", type=int, default=100, help="timed predictions per variant")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from ultralytics import YOLO


    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    frames = [cv2.imread(str(p)) for p in _val_images(data)[: max(args.runs, 20)]]
    frames = [f for f in frames if f is not None]

    pt_model = YOLO(args.weights)
    rows: list[dict] = []
    for sz in args.sizes:
        onnx_path = models_dir / f"best_{sz}.onnx"
        if not onnx_path.exists():
            exported = pt_model.export(format="onnx", imgsz=sz, opset=12, simplify=True, dynamic=False)
            onnx_path.write_bytes(Path(exported).read_bytes())

        for backend, path in (("pytorch", args.weights), ("onnx", str(onnx_path))):
            print(f"[bench] {backend} @ {sz}")
            m = YOLO(path, task="detect")
            metrics = m.val(data=str(data), imgsz=sz, device="cpu", batch=1, plots=False, verbose=False)
            speed = _bench(m, frames, sz, args.runs, args.warmup)
            rows.append(
                {
                    "backend": backend,
                    "imgsz": sz,
                    "map50": round(float(metrics.box.map50), 4),
                    "map50_95": round(float(metrics.box.map), 4),
                    **speed,
                    "size_mb": round(Path(path).stat().st_size / 1e6, 2),
                }
            )

    cols = list(rows[0].keys())
    with open(out / "tradeoff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "tradeoff.md").write_text("\n".join(md) + "\n")
    (out / "hardware.txt").write_text(f"{platform.platform()}\n{platform.processor()}\npython {platform.python_version()}\n")
    _plot(rows, out / "tradeoff.png")
    print("\n".join(md))
    print(f"\n[done] wrote {out}/tradeoff.csv, tradeoff.md, tradeoff.png")


if __name__ == "__main__":
    main()
