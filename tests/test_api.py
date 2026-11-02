"""API tests use a fake model, so they need no weights and no GPU."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from saferide import api, pipeline


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path / "out")
    with TestClient(api.app) as c:
        yield c


def _png_bytes():
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_detect_503_without_model(client):
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 503


def test_detect_ok_with_fake_model(client, monkeypatch):
    api.state["model"] = object()
    fake = {
        "num_detections": 1, "num_violations": 1, "has_violation": True, "latency_ms": 1.0,
        "detections": [{"class": "without_helmet", "confidence": 0.9, "bbox_xyxy": [1, 1, 30, 30], "is_violation": True}],
    }
    monkeypatch.setattr(pipeline, "detect_image", lambda *a, **k: fake)
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["has_violation"] is True

    r2 = client.post("/detect/annotated", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r2.status_code == 200 and r2.headers["content-type"] == "image/jpeg"
    assert r2.headers["x-num-violations"] == "1"


def test_detect_rejects_non_image(client):
    api.state["model"] = object()
    r = client.post("/detect", files={"file": ("a.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_outputs_blocks_path_traversal(client):
    assert client.get("/outputs/..%2F..%2Fetc%2Fpasswd").status_code == 404
