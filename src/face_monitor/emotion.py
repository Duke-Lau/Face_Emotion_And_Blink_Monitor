from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from face_monitor.models import BBox, EmotionResult, Point

# Keep matplotlib cache out of protected home directories.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "face_monitor_mpl"))


def _prepend_env_path(name: str, path: Path) -> None:
    current = os.environ.get(name, "")
    parts = [item for item in current.split(":") if item]
    path_str = str(path)
    if path_str not in parts:
        os.environ[name] = "{0}:{1}".format(path_str, current) if current else path_str


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE_PACKAGES = PROJECT_ROOT / ".venv" / "lib" / "python3.9" / "site-packages"
for library_dir in (
    VENV_SITE_PACKAGES / "cmeel.prefix" / "lib",
    VENV_SITE_PACKAGES / "torch" / ".dylibs",
    VENV_SITE_PACKAGES / "sklearn" / ".dylibs",
):
    if library_dir.exists():
        _prepend_env_path("DYLD_LIBRARY_PATH", library_dir)
        _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", library_dir)

try:
    from feat import Detector
except Exception:  # pragma: no cover - optional dependency during local verification
    Detector = None


DEFAULT_EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def _distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values()) or 1.0
    return {label: value / total for label, value in scores.items()}


class GeometryEmotionRecognizer:
    """Fallback emotion estimator based on facial geometry from landmarks."""

    backend_name = "geometry"
    emotion_labels = DEFAULT_EMOTION_LABELS
    action_unit_labels: Sequence[str] = ()

    def analyze(
        self,
        face_bgr,
        landmarks_px: List[Point],
        bbox: BBox,
    ) -> EmotionResult:
        del face_bgr  # The geometry backend only needs landmarks.

        face_width = max(_distance(landmarks_px[234], landmarks_px[454]), 1.0)
        mouth_width = _distance(landmarks_px[61], landmarks_px[291]) / face_width
        mouth_open = _distance(landmarks_px[13], landmarks_px[14]) / face_width
        eye_open = (
            _distance(landmarks_px[159], landmarks_px[145]) +
            _distance(landmarks_px[386], landmarks_px[374])
        ) / (2.0 * face_width)
        brow_eye_gap = (
            _distance(landmarks_px[105], landmarks_px[159]) +
            _distance(landmarks_px[334], landmarks_px[386])
        ) / (2.0 * face_width)

        happy = _clamp((mouth_width - 0.32) / 0.18 + (mouth_open - 0.012) / 0.05)
        surprise = _clamp((mouth_open - 0.03) / 0.08 + (eye_open - 0.015) / 0.03)
        sad = _clamp((0.055 - brow_eye_gap) / 0.03 + (0.02 - mouth_open) / 0.02)
        angry = _clamp((0.045 - brow_eye_gap) / 0.02 + (0.018 - mouth_open) / 0.02)
        fear = _clamp(surprise * 0.7 + sad * 0.2)
        disgust = _clamp(angry * 0.4)
        neutral = _clamp(0.45 - max(happy, surprise, sad, angry) * 0.3, 0.05, 0.6)

        scores = _normalize_scores(
            {
                "anger": angry + 0.01,
                "disgust": disgust + 0.01,
                "fear": fear + 0.01,
                "happiness": happy + 0.01,
                "sadness": sad + 0.01,
                "surprise": surprise + 0.01,
                "neutral": neutral,
            }
        )

        label = max(scores, key=scores.get)
        return EmotionResult(
            label=label,
            confidence=scores[label],
            scores=scores,
            backend=self.backend_name,
        )


class PyFeatEmotionRecognizer:
    """Emotion and Action Unit recognizer backed by py-feat."""

    backend_name = "py-feat"

    def __init__(self) -> None:
        if Detector is None:
            raise RuntimeError("py-feat is not installed")

        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            facepose_model="img2pose",
            identity_model="facenet",
            device="cpu",
            verbose=False,
        )
        self.emotion_labels = list(self.detector.info["emotion_model_columns"])
        # We keep py-feat's AU model available for compatibility, but skip AU
        # inference at runtime so the webcam loop only pays for emotion detection.
        self.action_unit_labels: Sequence[str] = ()

    def analyze_frame(self, frame_bgr, max_side: int = 640) -> List[Dict[str, object]]:
        import cv2

        scale = 1.0
        frame_for_model = frame_bgr
        frame_height, frame_width = frame_bgr.shape[:2]
        longest_side = max(frame_height, frame_width)
        if max_side and longest_side > max_side:
            scale = max_side / float(longest_side)
            resized_width = max(1, int(round(frame_width * scale)))
            resized_height = max(1, int(round(frame_height * scale)))
            frame_for_model = cv2.resize(frame_bgr, (resized_width, resized_height))

        rgb_frame = cv2.cvtColor(frame_for_model, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame, threshold=0.5)
        landmarks = self.detector.detect_landmarks(rgb_frame, faces)
        emotions = self.detector.detect_emotions(rgb_frame, faces, landmarks)
        frame_faces = faces[0] if faces else []
        frame_emotions = emotions[0] if emotions else []
        outputs: List[Dict[str, object]] = []

        for index, face in enumerate(frame_faces):
            x1, y1, x2, y2 = [int(round(float(value) / scale)) for value in face[:4]]
            emotion_values = frame_emotions[index] if index < len(frame_emotions) else []
            scores = _normalize_scores(
                {
                    label: max(float(value), 0.0)
                    for label, value in zip(self.emotion_labels, emotion_values)
                }
            )
            if not scores:
                scores = {"neutral": 1.0}

            label = max(scores, key=scores.get)
            outputs.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "emotion": EmotionResult(
                        label=label,
                        confidence=float(scores[label]),
                        scores=scores,
                        backend=self.backend_name,
                        action_units={},
                    ),
                }
            )

        return outputs


class EmotionAnalyzer:
    """Chooses the best available backend and exposes a single analyze API."""

    def __init__(self, preferred_backend: str = "auto") -> None:
        self.notice: Optional[str] = None

        if preferred_backend not in ("auto", "py-feat", "geometry"):
            raise ValueError("emotion backend must be one of: auto, py-feat, geometry")

        if preferred_backend in ("auto", "py-feat"):
            try:
                self.backend = PyFeatEmotionRecognizer()
            except Exception:
                self.backend = GeometryEmotionRecognizer()
                if preferred_backend == "py-feat":
                    self.notice = "py-feat backend unavailable, falling back to geometry-based emotion estimation."
                else:
                    self.notice = "py-feat backend not available, using geometry-based fallback for emotion estimation."
        else:
            self.backend = GeometryEmotionRecognizer()

    @property
    def backend_name(self) -> str:
        return self.backend.backend_name

    @property
    def emotion_labels(self) -> Sequence[str]:
        return getattr(self.backend, "emotion_labels", DEFAULT_EMOTION_LABELS)

    @property
    def action_unit_labels(self) -> Sequence[str]:
        return getattr(self.backend, "action_unit_labels", ())

    @property
    def supports_frame_analysis(self) -> bool:
        return hasattr(self.backend, "analyze_frame")

    def analyze(
        self,
        face_bgr,
        landmarks_px: List[Point],
        bbox: BBox,
    ) -> EmotionResult:
        return self.backend.analyze(face_bgr, landmarks_px, bbox)

    def analyze_frame(self, frame_bgr, max_side: int = 640) -> List[Dict[str, object]]:
        if not self.supports_frame_analysis:
            return []
        return self.backend.analyze_frame(frame_bgr, max_side=max_side)
