from saferide.violations import ViolationCounter, normalize


def test_normalize():
    assert normalize("Without Helmet") == "without_helmet"
    assert normalize(" no-helmet ") == "no_helmet"


def test_confirms_after_min_frames_only_once():
    c = ViolationCounter(["without_helmet"], min_frames=3)
    assert c.update(0, 1, "without_helmet") is False
    assert c.update(1, 1, "without_helmet") is False
    assert c.update(2, 1, "without_helmet") is True   # 3rd violating frame -> confirmed
    assert c.update(3, 1, "without_helmet") is False  # never fires twice
    assert c.violation_ids == [1]


def test_flicker_is_not_a_violation():
    c = ViolationCounter(["without_helmet"], min_frames=5)
    for f in range(4):
        c.update(f, 7, "without_helmet")
    for f in range(4, 40):
        c.update(f, 7, "with_helmet")
    assert c.violation_ids == []


def test_unique_ids_counted_once():
    c = ViolationCounter(["no_helmet"], min_frames=2)
    for f in range(10):
        c.update(f, 1, "No Helmet")
        c.update(f, 2, "no_helmet")
        c.update(f, 3, "helmet")
    s = c.summary()
    assert s["unique_tracks"] == 3
    assert s["confirmed_violations"] == 2
    assert s["violation_track_ids"] == [1, 2]


def test_smoothed_label_is_majority():
    c = ViolationCounter(["without_helmet"], min_frames=99, window=5)
    for f, lab in enumerate(["with_helmet", "without_helmet", "with_helmet", "with_helmet", "without_helmet"]):
        c.update(f, 1, lab)
    assert c.smoothed_label(1) == "with_helmet"
    assert c.smoothed_label(999) is None


def test_window_forgets_old_labels():
    c = ViolationCounter(["without_helmet"], min_frames=99, window=3)
    for f in range(10):
        c.update(f, 1, "without_helmet")
    for f in range(10, 13):
        c.update(f, 1, "with_helmet")
    assert c.smoothed_label(1) == "with_helmet"


def test_invalid_min_frames():
    try:
        ViolationCounter(["x"], min_frames=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
