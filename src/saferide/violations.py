"""Track-level violation logic (pure Python, no ML dependencies, fully unit-tested).

Why this exists
---------------
A detector works frame by frame. If it says "no helmet" in frame 17 for one
rider and "helmet" in frame 18, counting raw detections would give wrong,
inflated numbers. A tracker (ByteTrack) gives every rider a persistent ID, so we can
count UNIQUE riders and only confirm a violation after it has been seen in
`min_frames` frames.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional


def normalize(name: str) -> str:
    """'No Helmet' -> 'no_helmet' so config and dataset names always match."""
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_")


@dataclass
class TrackState:
    window: Deque[str]
    total_frames: int = 0
    violation_frames: int = 0
    confirmed: bool = False
    first_frame: Optional[int] = None
    confirmed_frame: Optional[int] = None


class ViolationCounter:
    def __init__(
        self,
        violation_classes: Iterable[str],
        min_frames: int = 5,
        window: int = 15,
    ) -> None:
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        self.violation_classes = {normalize(c) for c in violation_classes}
        self.min_frames = min_frames
        self.window = window
        self.tracks: Dict[int, TrackState] = {}

    def is_violation_label(self, class_name: str) -> bool:
        return normalize(class_name) in self.violation_classes

    def update(self, frame_idx: int, track_id: int, class_name: str) -> bool:
        """Register one detection of `track_id` in `frame_idx`.

        Returns True exactly once per track: on the frame where the track
        becomes a confirmed violation.
        """
        label = normalize(class_name)
        st = self.tracks.get(track_id)
        if st is None:
            st = TrackState(window=deque(maxlen=self.window), first_frame=frame_idx)
            self.tracks[track_id] = st

        st.window.append(label)
        st.total_frames += 1
        if label in self.violation_classes:
            st.violation_frames += 1

        if not st.confirmed and st.violation_frames >= self.min_frames:
            st.confirmed = True
            st.confirmed_frame = frame_idx
            return True
        return False

    def smoothed_label(self, track_id: int) -> Optional[str]:
        """Majority vote over the last `window` labels (stable label for drawing)."""
        st = self.tracks.get(track_id)
        if not st or not st.window:
            return None
        return Counter(st.window).most_common(1)[0][0]

    def is_confirmed(self, track_id: int) -> bool:
        st = self.tracks.get(track_id)
        return bool(st and st.confirmed)

    @property
    def violation_ids(self) -> List[int]:
        return sorted(t for t, s in self.tracks.items() if s.confirmed)

    def summary(self) -> dict:
        return {
            "unique_tracks": len(self.tracks),
            "confirmed_violations": len(self.violation_ids),
            "violation_track_ids": self.violation_ids,
        }
