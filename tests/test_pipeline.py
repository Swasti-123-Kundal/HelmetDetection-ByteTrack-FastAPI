"""Pipeline tests with a fake Ultralytics-like model (no weights needed)."""
from pathlib import Path

import cv2
import numpy as np

from saferide import pipeline
from saferide.config import Settings


class Vec(list):
    def tolist(self):
        return list(self)

    def int(self):
        return Vec(int(x) for x in self)


class Box:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = [cls], [conf], [Vec(xyxy)]


class Boxes:
    def __init__(self, items, ids=None):
        self.items = items
        self.id = Vec(ids) if ids is not None else None
        self.cls = Vec(b.cls[0] for b in items)
        self.conf = Vec(b.conf[0] for b in items)
        self.xyxy = Vec(b.xyxy[0] for b in items)

    def __iter__(self):
        return iter(self.items)


class Result:
    names = {0: "with_helmet", 1: "without_helmet"}

    def __init__(self, boxes, frame):
        self.boxes, self.orig_img = boxes, frame


class FakeModel:
    def __init__(self, per_frame):
        self.per_frame = per_frame

    def predict(self, img, **kw):
        return [Result(Boxes([Box(1, 0.9, [5, 5, 30, 30]), Box(0, 0.8, [35, 5, 60, 30])]), img)]

    def track(self, source, stream=True, **kw):
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            items = [Box(c, 0.9, [5, 5, 30, 30]) for c in self.per_frame[i]]
            ids = list(range(1, len(items) + 1))
            yield Result(Boxes(items, ids), frame)
            i += 1
        cap.release()


def _make_video(path: Path, n=12):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(n):
        w.write(np.zeros((64, 64, 3), np.uint8))
    w.release()


def test_detect_image_flags_violation():
    res = pipeline.detect_image(FakeModel([]), np.zeros((64, 64, 3), np.uint8), Settings())
    assert res["num_detections"] == 2 and res["num_violations"] == 1 and res["has_violation"]
    out = pipeline.draw_detections(np.zeros((64, 64, 3), np.uint8), res)
    assert out.shape == (64, 64, 3) and out.any()  # something was drawn


def test_track_video_counts_unique_violations(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out" / "o.mp4"
    _make_video(src, 12)
    # track 1: no helmet in all frames (violation); track 2: helmet in all frames (safe)
    per_frame = [[1, 0] for _ in range(12)]
    s = Settings(min_violation_frames=5)
    summary = pipeline.track_video(FakeModel(per_frame), src, dst, s, snapshot_dir=tmp_path / "snaps")
    assert summary["frames_processed"] == 12
    assert summary["unique_tracks"] == 2
    assert summary["confirmed_violations"] == 1
    assert summary["violation_track_ids"] == [1]
    assert dst.is_file() and dst.stat().st_size > 0
    assert len(list((tmp_path / "snaps").glob("*.jpg"))) == 1  # one crop per confirmed violator


def test_track_video_short_flicker_not_counted(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "o.mp4"
    _make_video(src, 10)
    per_frame = [[1] if i < 3 else [0] for i in range(10)]
    summary = pipeline.track_video(FakeModel(per_frame), src, dst, Settings(min_violation_frames=5))
    assert summary["confirmed_violations"] == 0


def test_track_video_bad_source(tmp_path):
    try:
        pipeline.track_video(FakeModel([]), tmp_path / "nope.mp4", tmp_path / "o.mp4", Settings())
    except ValueError:
        return
    raise AssertionError("expected ValueError")
