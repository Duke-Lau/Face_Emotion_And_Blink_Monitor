from __future__ import annotations

import math
from typing import Dict, List, Tuple

from face_monitor.models import FaceDetection, FaceState


class FaceTracker:
    """Simple centroid tracker so blink counts survive across frames."""

    def __init__(self, max_distance: float = 120.0, max_missing_frames: int = 15) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.states: Dict[int, FaceState] = {}
        self.next_face_id = 1

    def update(
        self,
        detections: List[FaceDetection],
        frame_index: int,
    ) -> List[Tuple[FaceDetection, FaceState]]:
        matches: List[Tuple[int, FaceDetection, FaceState]] = []
        unmatched_states = set(self.states.keys())
        unmatched_detections = set(range(len(detections)))

        candidate_pairs = []
        for face_id, state in self.states.items():
            for detection_index, detection in enumerate(detections):
                distance = math.hypot(
                    state.center[0] - detection.center[0],
                    state.center[1] - detection.center[1],
                )
                if distance <= self.max_distance:
                    candidate_pairs.append((distance, face_id, detection_index))

        for _, face_id, detection_index in sorted(candidate_pairs, key=lambda item: item[0]):
            if face_id not in unmatched_states or detection_index not in unmatched_detections:
                continue

            state = self.states[face_id]
            detection = detections[detection_index]
            state.center = detection.center
            state.bbox = detection.bbox
            state.last_seen_frame = frame_index
            matches.append((detection_index, detection, state))
            unmatched_states.remove(face_id)
            unmatched_detections.remove(detection_index)

        for detection_index in list(unmatched_detections):
            detection = detections[detection_index]
            state = FaceState(
                face_id=self.next_face_id,
                center=detection.center,
                bbox=detection.bbox,
                last_seen_frame=frame_index,
            )
            self.states[state.face_id] = state
            self.next_face_id += 1
            matches.append((detection_index, detection, state))

        stale_face_ids = [
            face_id
            for face_id, state in self.states.items()
            if frame_index - state.last_seen_frame > self.max_missing_frames
        ]
        for face_id in stale_face_ids:
            del self.states[face_id]

        matches.sort(key=lambda item: item[0])
        return [(detection, state) for _, detection, state in matches]
