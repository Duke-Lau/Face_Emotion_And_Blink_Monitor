from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class EmotionResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    backend: str
    action_units: Dict[str, float] = field(default_factory=dict)


@dataclass
class FaceDetection:
    landmarks_px: List[Point]
    bbox: BBox
    center: Point
    left_ear: float
    right_ear: float


@dataclass
class FaceState:
    face_id: int
    center: Point
    bbox: BBox
    blink_count: int = 0
    closed_frames: int = 0
    is_closed: bool = False
    last_blink_frame: int = -9999
    last_emotion: Optional[EmotionResult] = None
    emotion_updated_at: int = -9999
    last_seen_frame: int = 0
