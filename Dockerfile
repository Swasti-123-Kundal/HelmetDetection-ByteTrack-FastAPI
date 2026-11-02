FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best_640.onnx \
    OUTPUT_DIR=/app/outputs

# libgl/libglib are needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install -e .

# Model is copied if present at build time, or mount it: -v $(pwd)/models:/app/models
COPY models ./models

RUN useradd -m appuser && mkdir -p /app/outputs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "saferide.api:app", "--host", "0.0.0.0", "--port", "8000"]
