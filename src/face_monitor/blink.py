from __future__ import annotations

import math
from typing import Sequence

from face_monitor.models import FaceState, Point


LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)


def _distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def compute_eye_aspect_ratio(landmarks_px: Sequence[Point], indices: Sequence[int]) -> float:
    """Compute the standard eye aspect ratio from six eye landmarks."""
    p1, p2, p3, p4, p5, p6 = [landmarks_px[index] for index in indices]
    horizontal = _distance(p1, p4)
    if horizontal == 0:
        return 0.0

    vertical_a = _distance(p2, p6)
    vertical_b = _distance(p3, p5)
    return (vertical_a + vertical_b) / (2.0 * horizontal)


class BlinkDetector:
    """Counts blinks when the eye stays closed for a few consecutive frames."""

    def __init__(self, threshold: float = 0.23, min_closed_frames: int = 2) -> None:
        self.threshold = threshold
        self.min_closed_frames = min_closed_frames

    def update(self, state: FaceState, ear: float, frame_index: int) -> bool:
        if ear < self.threshold:
            state.closed_frames += 1
            state.is_closed = True
            return False

        blink_detected = state.is_closed and state.closed_frames >= self.min_closed_frames
        if blink_detected:
            state.blink_count += 1
            state.last_blink_frame = frame_index

        state.closed_frames = 0
        state.is_closed = False
        return blink_detected
