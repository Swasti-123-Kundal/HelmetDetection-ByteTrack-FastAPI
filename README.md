# SafeRide - Real-time Helmet Violation Detection & Tracking

End-to-end computer-vision system: **YOLOv8 fine-tuning -> ByteTrack multi-object tracking -> ONNX export with a latency/accuracy study -> FastAPI service -> Docker**.

It does not just detect "no helmet" in single frames. It tracks every rider with a persistent ID and counts **unique violators**, confirming a violation only after it is seen in N frames. That avoids inflated counts from per-frame flicker.

```
 video / image
      |
      v
 YOLOv8 (helmet / no-helmet)  --ONNX-->  onnxruntime
      |
      v
 ByteTrack (persistent track IDs)
      |
      v
 ViolationCounter  (per-ID frame voting -> unique confirmed violations)
      |
      +--> annotated video + violator crops
      +--> FastAPI  (/detect, /detect/annotated, /track, /health)  -->  Docker
```

## Project layout

| Path | What it does |
|---|---|
| `scripts/voc_to_yolo.py` | Convert the Kaggle VOC dataset to YOLO format + `data.yaml` |
| `src/saferide/train.py` | Fine-tune YOLOv8 |
| `src/saferide/export.py` | Export ONNX at several input sizes |
| `src/saferide/tradeoff.py` | Benchmark PyTorch vs ONNX: mAP, latency (mean/p50/p95), FPS; writes CSV + plot |
| `src/saferide/violations.py` | Track-level violation logic (pure Python, unit-tested) |
| `src/saferide/pipeline.py` | Image detection, video tracking, drawing |
| `src/saferide/track.py` | CLI for video tracking |
| `src/saferide/api.py` | FastAPI app |
| `Dockerfile` | CPU image for the API |
| `notebooks/train_colab.ipynb` | Train on a free Colab GPU |
| `docs/EXPLAIN.md` | Concepts + interview Q&A (Hinglish) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                              # logic + API tests (no weights needed)
```

### 1. Data
```bash
pip install kaggle                  # needs ~/.kaggle/kaggle.json (Kaggle > Settings > API)
kaggle datasets download -d andrewmvd/helmet-detection -p raw --unzip
python scripts/voc_to_yolo.py --src raw --out datasets/helmet
```
Check the printed class names. The "no helmet" class must appear in `violation_classes` in `configs/project.yaml`.

### 2. Train (use Colab if you have no GPU)
```bash
python -m saferide.train --data datasets/helmet/data.yaml --model yolov8n.pt --epochs 50 --name helmet_v8n
```

### 3. Export + trade-off study
```bash
python -m saferide.export   --weights runs/helmet_v8n/weights/best.pt --sizes 320 416 640
python -m saferide.tradeoff --weights runs/helmet_v8n/weights/best.pt --data datasets/helmet/data.yaml --sizes 320 416 640
```
Outputs `results/tradeoff.csv`, `tradeoff.md`, `tradeoff.png`, `hardware.txt`.

### 4. Track a video
```bash
python -m saferide.track --model models/best_640.onnx --source samples/traffic.mp4 \
    --out outputs/tracked.mp4 --snapshots outputs/violators
```
Use a road video with two-wheelers (record your own, or take a free clip from Pexels / Pixabay).

### 5. API
```bash
MODEL_PATH=models/best_640.onnx uvicorn saferide.api:app --port 8000
# open http://localhost:8000/docs
curl -F "file=@test.jpg" http://localhost:8000/detect
curl -F "file=@samples/traffic.mp4" http://localhost:8000/track
```

### 6. Docker
```bash
docker build -t saferide:latest .
docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models saferide:latest
```

## Results (fill in from YOUR runs)

Detection quality (validation split):

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| YOLOv8n | 640 | _TODO_ | _TODO_ |

Latency vs accuracy (CPU: _your CPU name_). Paste `results/tradeoff.md` here and embed `results/tradeoff.png`.

Tracking: _TODO: unique violators found in your test video vs. manual count._

## Known limitations (be upfront about these in interviews)
- The Kaggle dataset is small; accuracy on real CCTV footage will be lower than on its own validation split.
- ByteTrack can switch IDs under heavy occlusion, which can over/under-count. A longer `min_violation_frames` trades recall for precision.
- Violations are counted per detected head/helmet box, not yet associated with a motorcycle/rider. That is the natural next step.
- Output video uses the `mp4v` codec: it plays in VLC and most desktop players, but not always in browsers.
- Ultralytics is AGPL-3.0. Fine for a portfolio project; check licensing before any commercial use.

## Resume bullets (replace X with your measured numbers)
- Built a real-time helmet-violation detection system: fine-tuned **YOLOv8** on a custom helmet dataset (mAP@0.5 = X) and added **ByteTrack** multi-object tracking with per-ID voting to count unique violators and reduce false counts.
- Exported the model to **ONNX** and benchmarked PyTorch vs ONNX at 3 input sizes on CPU, reducing latency from X ms to Y ms (Z% faster) for a W-point change in mAP@0.5:0.95.
- Deployed as a **FastAPI** service (image + video endpoints) in a **Docker** container with health checks and unit tests.
