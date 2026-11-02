PY ?= python
DATA ?= datasets/helmet/data.yaml
WEIGHTS ?= runs/helmet_v8n/weights/best.pt
VIDEO ?= samples/traffic.mp4

setup:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

train:
	$(PY) -m saferide.train --data $(DATA) --model yolov8n.pt --epochs 50 --name helmet_v8n

export:
	$(PY) -m saferide.export --weights $(WEIGHTS) --sizes 320 416 640

tradeoff:
	$(PY) -m saferide.tradeoff --weights $(WEIGHTS) --data $(DATA) --sizes 320 416 640

track:
	$(PY) -m saferide.track --model models/best_640.onnx --source $(VIDEO) --out outputs/tracked.mp4

serve:
	MODEL_PATH=models/best_640.onnx uvicorn saferide.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t saferide:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $$(pwd)/models:/app/models saferide:latest
