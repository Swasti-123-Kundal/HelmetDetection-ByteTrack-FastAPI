"""Inference building blocks shared by the CLI, the API and the benchmark."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .config import Settings
from .violations import ViolationCounter, normalize

GREEN = (80, 200, 80)
RED = (60, 60, 230)
YELLOW = (0, 200, 230)


def load_model(path: str | Path):
    """Load a .pt or .onnx model. Imported lazily so tests don't need ultralytics."""
    from ultralytics import YOLO

    return YOLO(str(path), task="detect")


# --------------------------------------------------------------------------- #
# Single image
# --------------------------------------------------------------------------- #
def detect_image(
    model,
    image_bgr: np.ndarray,
    settings: Settings,
    violation_classes: Optional[Iterable[str]] = None,
) -> dict:
    """Run detection on one BGR image and return a JSON-friendly dict."""
    viol = {normalize(c) for c in (violation_classes or settings.violation_classes)}
    t0 = time.perf_counter()
    r = model.predict(
        image_bgr,
        conf=settings.conf,
        iou=settings.iou,
        imgsz=settings.imgsz,
        verbose=False,
    )[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for box in r.boxes:
        cls_name = normalize(r.names[int(box.cls[0])])
        detections.append(
            {
                "class": cls_name,
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
                "is_violation": cls_name in viol,
            }
        )
    n_viol = sum(d["is_violation"] for d in detections)
    return {
        "num_detections": len(detections),
        "num_violations": n_viol,
        "has_violation": n_viol > 0,
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
    }


def draw_detections(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Draw boxes from a `detect_image` result on a copy of the image."""
    out = image_bgr.copy()
    for d in result["detections"]:
        x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
        color = RED if d["is_violation"] else GREEN
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _label(out, f'{d["class"]} {d["confidence"]:.2f}', x1, y1, color)
    return out


# --------------------------------------------------------------------------- #
# Video + tracking
# --------------------------------------------------------------------------- #
def _label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(y, th + 6)
    cv2.rectangle(img, (x, y0 - th - 6), (x + tw + 4, y0 + bl - 4), color, -1)
    cv2.putText(img, text, (x + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def track_video(
    model,
    source: str | Path,
    out_path: str | Path,
    settings: Settings,
    violation_classes: Optional[Iterable[str]] = None,
    snapshot_dir: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
) -> dict:
    """Detect + track every frame, count UNIQUE helmet violations, write an annotated video.

    Uses Ultralytics' built-in ByteTrack (tracker=bytetrack.yaml). `persist=False`
    is correct here because we pass the whole video as `source`: the tracker is
    re-initialised for each new video and keeps its state across frames of this one.
    """
    source, out_path = Path(source), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_dir:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height)
    )
    counter = ViolationCounter(
        violation_classes or settings.violation_classes,
        min_frames=settings.min_violation_frames,
        window=settings.smoothing_window,
    )

    t0 = time.perf_counter()
    n_frames = 0
    stream = model.track(
        source=str(source),
        tracker=settings.tracker,
        persist=False,
        stream=True,
        conf=settings.conf,
        iou=settings.iou,
        imgsz=settings.imgsz,
        verbose=False,
    )
    for i, r in enumerate(stream):
        if max_frames and i >= max_frames:
            break
        frame = r.orig_img.copy()
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxys = boxes.xyxy.tolist()
            for tid, c, cf, bb in zip(ids, clss, confs, xyxys):
                newly = counter.update(i, tid, r.names[c])
                shown = counter.smoothed_label(tid) or normalize(r.names[c])
                is_v = counter.is_violation_label(shown)
                color = RED if (is_v or counter.is_confirmed(tid)) else GREEN
                x1, y1, x2, y2 = (int(v) for v in bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                _label(frame, f"#{tid} {shown} {cf:.2f}", x1, y1, color)
                if newly and snapshot_dir:
                    crop = r.orig_img[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size:
                        cv2.imwrite(str(Path(snapshot_dir) / f"violation_id{tid}_f{i}.jpg"), crop)

        elapsed = max(time.perf_counter() - t0, 1e-6)
        hud = f"frame {i} | violations: {len(counter.violation_ids)} | {(i + 1) / elapsed:.1f} FPS"
        _label(frame, hud, 8, 24, YELLOW)
        writer.write(frame)
        n_frames = i + 1

    writer.release()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    return {
        "frames_processed": n_frames,
        "processing_fps": round(n_frames / elapsed, 2),
        "video_fps": round(fps_in, 2),
        **counter.summary(),
        "output_video": str(out_path),
    }
