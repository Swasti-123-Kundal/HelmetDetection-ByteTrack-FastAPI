"""FastAPI service: image detection + video tracking.

Run:
    MODEL_PATH=models/best_640.onnx uvicorn saferide.api:app --port 8000
Docs (Swagger UI): http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from . import pipeline
from .config import load_settings

log = logging.getLogger("saferide.api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_640.onnx"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))

state: dict = {"model": None, "settings": None}
_infer_lock = threading.Lock()  # the model / tracker are not thread-safe


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["settings"] = load_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        state["model"] = pipeline.load_model(MODEL_PATH)
        log.info("Loaded model %s", MODEL_PATH)
    else:
        log.warning("Model file not found: %s (API will return 503 until it exists)", MODEL_PATH)
    yield
    state["model"] = None


app = FastAPI(
    title="SafeRide - Helmet Violation Detection",
    version="0.1.0",
    description="YOLOv8 + ByteTrack, served from an ONNX model.",
    lifespan=lifespan,
)


def _require_model():
    if state.get("model") is None:
        raise HTTPException(503, f"Model not loaded. Put a model at {MODEL_PATH} (or set MODEL_PATH).")
    return state["model"]


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the file as an image (use jpg/png).")
    return img


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state.get("model") is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    """Detect helmets / violations in one image. Returns JSON."""
    model = _require_model()
    img = _decode_image(await file.read())
    with _infer_lock:
        return pipeline.detect_image(model, img, state["settings"])


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...)) -> Response:
    """Same as /detect but returns the image with boxes drawn (JPEG)."""
    model = _require_model()
    img = _decode_image(await file.read())
    with _infer_lock:
        result = pipeline.detect_image(model, img, state["settings"])
    ok, buf = cv2.imencode(".jpg", pipeline.draw_detections(img, result))
    if not ok:
        raise HTTPException(500, "Failed to encode output image")
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Num-Violations": str(result["num_violations"])})


@app.post("/track")
def track(file: UploadFile = File(...), max_frames: int | None = None) -> dict:
    """Upload a video; get unique-violation counts + a link to the annotated video."""
    model = _require_model()
    job = uuid.uuid4().hex[:12]
    src = OUTPUT_DIR / f"{job}_input{Path(file.filename or 'video.mp4').suffix or '.mp4'}"
    dst = OUTPUT_DIR / f"{job}_tracked.mp4"
    snaps = OUTPUT_DIR / f"{job}_violators"

    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if src.stat().st_size > MAX_UPLOAD_MB * 1_000_000:
        src.unlink(missing_ok=True)
        raise HTTPException(413, f"Video larger than {MAX_UPLOAD_MB} MB")

    try:
        with _infer_lock:
            summary = pipeline.track_video(
                model, src, dst, state["settings"], snapshot_dir=snaps, max_frames=max_frames
            )
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        src.unlink(missing_ok=True)

    summary["output_video"] = f"/outputs/{dst.name}"
    return summary


@app.get("/outputs/{name}")
def get_output(name: str) -> FileResponse:
    path = OUTPUT_DIR / Path(name).name  # strip any path components (no traversal)
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)
