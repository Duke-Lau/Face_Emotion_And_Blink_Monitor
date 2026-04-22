from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from face_monitor.models import BBox, EmotionResult


class SessionLogger:
    """Writes experiment-friendly frame summaries to CSV."""

    def __init__(
        self,
        path: Path,
        emotion_labels: Sequence[str],
        action_unit_labels: Sequence[str],
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.emotion_labels = list(emotion_labels)
        self.action_unit_labels = list(action_unit_labels)
        self.fieldnames = [
            "timestamp_s",
            "frame_index",
            "face_id",
            "blink_count",
            "ear",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "emotion_label",
            "emotion_confidence",
            "emotion_backend",
        ]
        self.fieldnames.extend("emotion_{0}".format(label) for label in self.emotion_labels)
        self.fieldnames.extend("au_{0}".format(label) for label in self.action_unit_labels)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log_face(
        self,
        *,
        timestamp_s: float,
        frame_index: int,
        face_id: int,
        blink_count: int,
        ear: float,
        bbox: BBox,
        emotion: EmotionResult,
    ) -> None:
        x1, y1, x2, y2 = bbox
        row = {
            "timestamp_s": round(timestamp_s, 3),
            "frame_index": frame_index,
            "face_id": face_id,
            "blink_count": blink_count,
            "ear": round(ear, 4),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "emotion_label": emotion.label,
            "emotion_confidence": round(emotion.confidence, 6),
            "emotion_backend": emotion.backend,
        }
        for label in self.emotion_labels:
            row["emotion_{0}".format(label)] = round(float(emotion.scores.get(label, 0.0)), 6)
        for label in self.action_unit_labels:
            row["au_{0}".format(label)] = round(float(emotion.action_units.get(label, 0.0)), 6)

        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
