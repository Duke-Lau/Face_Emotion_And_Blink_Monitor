from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, List, Optional, Tuple


@dataclass
class EmotionFrameResult:
    frame_index: int
    analyzed_faces: List[Dict[str, object]]
    face_emotions: Dict[int, object]
    error: Optional[str] = None


class AsyncEmotionWorker:
    """Run heavyweight emotion inference off the camera/UI thread."""

    def __init__(self, analyzer, max_side: int = 640) -> None:
        self.analyzer = analyzer
        self.max_side = max_side
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._pending_frame = None
        self._pending_frame_index = -1
        self._pending_faces: List[Tuple[int, Tuple[int, int]]] = []
        self._latest_result: Optional[EmotionFrameResult] = None
        self._thread: Optional[threading.Thread] = None
        self._is_processing = False
        self._last_error: Optional[str] = None

        if self.analyzer.supports_frame_analysis:
            self._thread = threading.Thread(target=self._run, name="emotion-worker", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._thread is not None

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return "disabled"
        with self._lock:
            if self._last_error:
                return "error"
            if self._is_processing:
                return "analyzing"
            if self._latest_result is not None:
                return "ready"
        return "idle"

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def submit(self, frame, frame_index: int, tracked_faces) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._pending_frame = frame.copy()
            self._pending_frame_index = frame_index
            self._pending_faces = [
                (state.face_id, detection.center)
                for detection, state in tracked_faces
            ]
        self._wake_event.set()

    def get_latest(self) -> Optional[EmotionFrameResult]:
        with self._lock:
            result = self._latest_result
            self._latest_result = None
        return result

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        self._wake_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.2)
            self._wake_event.clear()
            if self._stop_event.is_set():
                return

            with self._lock:
                frame = self._pending_frame
                frame_index = self._pending_frame_index
                pending_faces = list(self._pending_faces)
                self._pending_frame = None
                self._pending_faces = []
                self._is_processing = frame is not None

            if frame is None:
                continue

            try:
                analyzed_faces = self.analyzer.analyze_frame(frame, max_side=self.max_side)
                result = EmotionFrameResult(
                    frame_index=frame_index,
                    analyzed_faces=analyzed_faces,
                    face_emotions=self._match_face_ids(pending_faces, analyzed_faces),
                )
            except Exception as exc:  # pragma: no cover
                result = EmotionFrameResult(
                    frame_index=frame_index,
                    analyzed_faces=[],
                    face_emotions={},
                    error=str(exc),
                )

            with self._lock:
                self._is_processing = False
                self._last_error = result.error
                self._latest_result = result

    @staticmethod
    def _match_face_ids(
        tracked_faces: List[Tuple[int, Tuple[int, int]]],
        analyzed_faces: List[Dict[str, object]],
    ) -> Dict[int, object]:
        matches: Dict[int, object] = {}
        used_indices = set()
        candidate_pairs = []

        for face_id, center in tracked_faces:
            for index, analyzed_face in enumerate(analyzed_faces):
                analyzed_center = analyzed_face["center"]
                dx = center[0] - analyzed_center[0]
                dy = center[1] - analyzed_center[1]
                distance_sq = dx * dx + dy * dy
                candidate_pairs.append((distance_sq, face_id, index))

        for distance_sq, face_id, analyzed_index in sorted(candidate_pairs, key=lambda item: item[0]):
            if analyzed_index in used_indices or face_id in matches:
                continue
            if distance_sq > 180 * 180:
                continue
            matches[face_id] = analyzed_faces[analyzed_index]["emotion"]
            used_indices.add(analyzed_index)

        if len(tracked_faces) == 1 and len(analyzed_faces) == 1:
            face_id, _ = tracked_faces[0]
            matches.setdefault(face_id, analyzed_faces[0]["emotion"])

        return matches
