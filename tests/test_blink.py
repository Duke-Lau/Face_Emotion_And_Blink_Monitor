from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_monitor.blink import BlinkDetector, compute_eye_aspect_ratio
from face_monitor.models import FaceState


def test_eye_aspect_ratio_is_higher_for_open_eye():
    open_eye = [
        (0, 0),
        (2, 3),
        (6, 3),
        (10, 0),
        (6, -3),
        (2, -3),
    ]
    closed_eye = [
        (0, 0),
        (2, 1),
        (6, 1),
        (10, 0),
        (6, -1),
        (2, -1),
    ]

    open_ear = compute_eye_aspect_ratio(open_eye, range(6))
    closed_ear = compute_eye_aspect_ratio(closed_eye, range(6))

    assert open_ear > closed_ear


def test_blink_detector_counts_single_blink():
    detector = BlinkDetector(threshold=0.23, min_closed_frames=2)
    state = FaceState(face_id=1, center=(0, 0), bbox=(0, 0, 10, 10))

    ears = [0.30, 0.29, 0.18, 0.16, 0.28, 0.29]
    detections = [detector.update(state, ear, frame_index) for frame_index, ear in enumerate(ears)]

    assert detections.count(True) == 1
    assert state.blink_count == 1
