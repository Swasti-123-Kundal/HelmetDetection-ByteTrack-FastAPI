# SafeRide - Real-time Helmet Violation Detection & Tracking

End-to-end computer-vision system: **YOLOv8 fine-tuning -> ByteTrack multi-object tracking -> ONNX export with a CPU latency/accuracy study -> FastAPI service -> Docker**.

It does not just detect "no helmet" in single frames. It tracks every rider with a persistent ID and counts **unique violators**, confirming a violation only after the same track is seen without a helmet in N frames. That avoids inflated counts from per-frame flicker.

```
 video / image
      |
      v
 YOLOv8s (with_helmet / without_helmet)  --ONNX-->  onnxruntime
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

## Results

**Setup:** Kaggle *Helmet Detection* dataset (764 images, 2 classes: `with_helmet`, `without_helmet`), split 533 / 152 / 76 images (train / val / test). YOLOv8s fine-tuned at 800 px on a Tesla T4 with early stopping (best epoch 31 of 46, patience 15).

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| Validation | 0.794 | 0.462 | 0.710 | 0.785 |
| Test (held-out) | 0.815 | 0.508 | 0.822 | 0.783 |

The best checkpoint was selected on the validation split; the test split was never used for training or checkpoint selection.

Per class on the test split:

| Class | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| with_helmet | 0.932 | 0.618 |
| without_helmet | 0.698 | 0.398 |

The violation class (`without_helmet`) is the weaker one, which is why track-level voting is used to filter single-frame false alarms.



### CPU latency vs accuracy (PyTorch vs ONNX Runtime)

Measured on a Colab Intel Xeon CPU @ 2.00GHz, one image per call, 100 timed runs after warmup. mAP is on the validation split.

| backend | imgsz | mAP@0.5 | mAP@0.5:0.95 | mean latency (ms) | FPS |
|---|---|---|---|---|---|
| pytorch | 320 | 0.662 | 0.368 | 73.3 | 13.6 |
| onnx | 320 | 0.663 | 0.370 | 77.6 | 12.9 |
| pytorch | 416 | 0.746 | 0.422 | 111.1 | 9.0 |
| onnx | 416 | 0.747 | 0.428 | 129.6 | 7.7 |
| pytorch | 640 | 0.790 | 0.454 | 234.0 | 4.3 |
| onnx | 640 | 0.787 | 0.456 | 301.4 | 3.3 |



**Findings**
- Reducing the input size from 640 to 416 cut mean latency by about 2.1x (234 ms to 111 ms) for a 3.2-point drop in mAP@0.5:0.95. 416 is the best trade-off here; 320 is faster but loses 8.6 points.
- ONNX preserved accuracy (within about 0.005 mAP) but was **not faster than PyTorch on this CPU**. It did show a smaller gap between median and p95 latency, i.e. more consistent response times. The cause was not investigated (possible factors: thread settings, PyTorch's optimized CPU kernels).

## Tracking demo

Video tracking is run with `saferide.track` (see below). It writes an annotated video and a cropped image of each confirmed violator.

## Project layout

| Path | What it does |
|---|---|
| `scripts/voc_to_yolo.py` | Convert the Kaggle VOC dataset to YOLO format + `data.yaml` (train/val/test split) |
| `src/saferide/train.py` | Fine-tune YOLOv8 |
| `src/saferide/export.py` | Export ONNX at several input sizes |
| `src/saferide/tradeoff.py` | Benchmark PyTorch vs ONNX: mAP, latency (mean/p50/p95), FPS; writes CSV + plot |
| `src/saferide/violations.py` | Track-level violation logic (pure Python, unit-tested) |
| `src/saferide/pipeline.py` | Image detection, video tracking, drawing |
| `src/saferide/track.py` | CLI for video tracking |
| `src/saferide/api.py` | FastAPI app |
| `configs/project.yaml` | Violation classes, thresholds, tracker, `min_violation_frames` |
| `Dockerfile` | CPU image for the API |
| `notebooks/train_colab.ipynb` | Train on a free Colab GPU |
| `docs/EXPLAIN.md` | Concepts and interview Q&A (Hinglish) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                              # 17 tests, no weights needed
```

### 1. Data
Dataset: [andrewmvd/helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) on Kaggle (CC0), by Larxel.
```bash
kaggle datasets download -d andrewmvd/helmet-detection -p raw --unzip   # needs ~/.kaggle/kaggle.json
python scripts/voc_to_yolo.py --src raw --out datasets/helmet
```
Check the printed class names: the "no helmet" class must appear in `violation_classes` in `configs/project.yaml`.

### 2. Train (a free Colab T4 GPU is enough, see `notebooks/train_colab.ipynb`)
```bash
python -m saferide.train --data datasets/helmet/data.yaml --model yolov8s.pt \
    --epochs 100 --imgsz 800 --name helmet_v8s_highacc
```

### 3. Export and trade-off study
```bash
python -m saferide.export   --weights <path/to/best.pt> --sizes 320 416 640
python -m saferide.tradeoff --weights <path/to/best.pt> --data datasets/helmet/data.yaml --sizes 320 416 640
```
Outputs `results/tradeoff.csv`, `tradeoff.md`, `tradeoff.png`, `hardware.txt`. Ultralytics saves weights under `runs/detect/runs/<name>/weights/best.pt`.

### 4. Track a video
ONNX models have a fixed input size, so `--imgsz` must match the exported file (and `imgsz` in `configs/project.yaml`).
```bash
python -m saferide.track --model models/best_416.onnx --source samples/traffic.mp4 --imgsz 416 \
    --out outputs/tracked.mp4 --snapshots outputs/violators
```

### 5. API
```bash
# bash
MODEL_PATH=models/best_416.onnx uvicorn saferide.api:app --port 8000
# PowerShell
$env:MODEL_PATH="models/best_416.onnx"; uvicorn saferide.api:app --port 8000
```
Open http://localhost:8000/docs, or:
```bash
curl -F "file=@test.jpg" http://localhost:8000/detect
curl -F "file=@samples/traffic.mp4" http://localhost:8000/track
```

### 6. Docker
```bash
docker build -t saferide:latest .
docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models saferide:latest
```

## Known limitations
- The dataset is small (764 images, mostly close-up photos of riders and cyclists); accuracy on real CCTV footage will be lower than on its own test split. The test split is also small (44 `without_helmet` objects), so metrics are noisy.
- ByteTrack can switch IDs under heavy occlusion, which can over- or under-count. A larger `min_violation_frames` trades recall for precision.
- Violations are counted per detected head/helmet box and are not yet associated with a motorcycle or rider. That is the natural next step.
- The output video uses the `mp4v` codec: it plays in VLC and most desktop players, but not always in browsers.
- Latency numbers are from one Colab CPU; results on other hardware will differ.
- Ultralytics is licensed AGPL-3.0; check licensing before any commercial use.
