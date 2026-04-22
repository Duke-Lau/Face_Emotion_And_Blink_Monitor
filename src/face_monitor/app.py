from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import List, Optional, Sequence

from face_monitor.blink import (
    BlinkDetector,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    compute_eye_aspect_ratio,
)
from face_monitor.emotion import EmotionAnalyzer
from face_monitor.emotion_worker import AsyncEmotionWorker
from face_monitor.models import BBox, FaceDetection, Point
from face_monitor.session_log import SessionLogger
from face_monitor.tracking import FaceTracker


WINDOW_NAME = "Face Emotion & Blink Monitor"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime webcam face analysis focused on emotion and blink detection.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open. Default: 0")
    parser.add_argument("--width", type=int, default=1280, help="Camera width. Default: 1280")
    parser.add_argument("--height", type=int, default=720, help="Camera height. Default: 720")
    parser.add_argument("--max-faces", type=int, default=2, help="Maximum number of faces to track. Default: 2")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to MediaPipe face landmarker model bundle (.task).",
    )
    parser.add_argument(
        "--emotion-backend",
        choices=("auto", "py-feat", "geometry"),
        default="auto",
        help="Emotion recognition backend. Default: auto",
    )
    parser.add_argument(
        "--emotion-interval",
        type=int,
        default=6,
        help="Re-run emotion inference every N frames per face. Default: 6",
    )
    parser.add_argument(
        "--blink-threshold",
        type=float,
        default=0.23,
        help="EAR threshold below which the eye is considered closed. Default: 0.23",
    )
    parser.add_argument(
        "--min-closed-frames",
        type=int,
        default=2,
        help="Minimum consecutive closed-eye frames to count as a blink. Default: 2",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the webcam preview so it behaves like a selfie camera.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Display FPS in the top-left corner.",
    )
    parser.add_argument(
        "--emotion-log",
        default="",
        help="Optional CSV log path. Defaults to logs/session_<timestamp>.csv",
    )
    parser.add_argument(
        "--emotion-max-side",
        type=int,
        default=640,
        help="Resize frames sent to the emotion backend so their longest side is at most this value. Default: 640",
    )
    return parser


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _bbox_with_padding(bbox: BBox, frame_shape: Sequence[int], padding_ratio: float = 0.18) -> BBox:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    return (
        _clamp(x1 - pad_x, 0, width - 1),
        _clamp(y1 - pad_y, 0, height - 1),
        _clamp(x2 + pad_x, 0, width - 1),
        _clamp(y2 + pad_y, 0, height - 1),
    )


def _landmarks_to_pixels(face_landmarks, frame_width: int, frame_height: int) -> List[Point]:
    points: List[Point] = []
    landmark_items = getattr(face_landmarks, "landmark", face_landmarks)
    for landmark in landmark_items:
        x = _clamp(int(landmark.x * frame_width), 0, frame_width - 1)
        y = _clamp(int(landmark.y * frame_height), 0, frame_height - 1)
        points.append((x, y))
    return points


def _build_detection(landmarks_px: List[Point]) -> FaceDetection:
    xs = [point[0] for point in landmarks_px]
    ys = [point[1] for point in landmarks_px]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    left_ear = compute_eye_aspect_ratio(landmarks_px, LEFT_EYE_INDICES)
    right_ear = compute_eye_aspect_ratio(landmarks_px, RIGHT_EYE_INDICES)
    return FaceDetection(
        landmarks_px=landmarks_px,
        bbox=bbox,
        center=center,
        left_ear=left_ear,
        right_ear=right_ear,
    )


def _crop_face(frame, bbox: BBox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _draw_face_overlay(
    frame,
    state,
    detection: FaceDetection,
    frame_index: int,
    emotion_status: str,
) -> None:
    import cv2

    x1, y1, x2, y2 = detection.bbox
    avg_ear = (detection.left_ear + detection.right_ear) / 2.0
    color = (60, 200, 120)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    top_lines = [
        "Face #{0}".format(state.face_id),
        "Blinks: {0}".format(state.blink_count),
        "EAR: {0:.3f}".format(avg_ear),
    ]

    if state.last_emotion is not None:
        top_lines.append(
            "Emotion: {0} {1:.0%}".format(
                state.last_emotion.label,
                state.last_emotion.confidence,
            )
        )
        top_lines.append("Backend: {0}".format(state.last_emotion.backend))
        if state.last_emotion.action_units:
            top_aus = sorted(
                state.last_emotion.action_units.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:2]
            top_lines.append(
                "AUs: {0}".format(
                    ", ".join("{0}:{1:.2f}".format(label, value) for label, value in top_aus)
                )
            )
    elif emotion_status in ("idle", "analyzing", "ready"):
        top_lines.append("Emotion: analyzing...")

    if frame_index - state.last_blink_frame <= 4:
        top_lines.append("Blink detected")

    text_y = max(26, y1 - 12)
    for line in top_lines:
        cv2.putText(
            frame,
            line,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        text_y += 20


def _draw_hud(
    frame,
    face_count: int,
    fps: Optional[float],
    analyzer: EmotionAnalyzer,
    emotion_status: str,
    emotion_error: Optional[str],
) -> None:
    import cv2

    hud_lines = [
        "Faces: {0}".format(face_count),
        "Emotion backend: {0}".format(analyzer.backend_name),
        "Emotion status: {0}".format(emotion_status),
        "Keys: q quit | r reset blinks".format(),
    ]

    if fps is not None:
        hud_lines.insert(1, "FPS: {0:.1f}".format(fps))

    if emotion_error:
        hud_lines.append("Emotion error: {0}".format(emotion_error[:70]))

    y = 28
    for line in hud_lines:
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


def _create_landmarker(mp, model_path: Path, max_faces: int):
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _create_face_runtime(mp, args, model_path: Path):
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        try:
            runtime = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=args.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return "solutions", runtime
        except Exception as exc:
            print("MediaPipe FaceMesh init failed: {0}".format(exc))
            print("Falling back to the Tasks API if a .task model is available.")

    if not model_path.exists():
        print("Face landmarker model not found: {0}".format(model_path))
        print("Download it from:")
        print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("Then rerun with --model-path or place it at {0}".format(DEFAULT_MODEL_PATH))
        return None, None

    try:
        runtime = _create_landmarker(mp, model_path=model_path, max_faces=args.max_faces)
    except Exception as exc:
        print("Failed to initialize MediaPipe Face Landmarker: {0}".format(exc))
        return None, None

    return "tasks", runtime


def _match_emotion_results(tracked_faces, analyzed_faces):
    matches = {}
    used_indices = set()
    candidate_pairs = []

    for detection, state in tracked_faces:
        for index, analyzed_face in enumerate(analyzed_faces):
            center = analyzed_face["center"]
            distance = math.hypot(state.center[0] - center[0], state.center[1] - center[1])
            candidate_pairs.append((distance, state.face_id, index))

    for distance, face_id, analyzed_index in sorted(candidate_pairs, key=lambda item: item[0]):
        if analyzed_index in used_indices or face_id in matches:
            continue
        if distance > 180:
            continue
        matches[face_id] = analyzed_faces[analyzed_index]["emotion"]
        used_indices.add(analyzed_index)

    return matches


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        print("Missing dependency: {0}".format(exc))
        print("Install requirements first: python3 -m pip install -r requirements.txt")
        return 1

    parser = _build_parser()
    args = parser.parse_args(argv)
    model_path = Path(args.model_path).expanduser().resolve()
    log_path = Path(args.emotion_log).expanduser().resolve() if args.emotion_log else (
        DEFAULT_LOG_DIR / "session_{0}.csv".format(time.strftime("%Y%m%d_%H%M%S"))
    )

    analyzer = EmotionAnalyzer(preferred_backend=args.emotion_backend)
    emotion_worker = AsyncEmotionWorker(analyzer, max_side=args.emotion_max_side)
    tracker = FaceTracker()
    blink_detector = BlinkDetector(
        threshold=args.blink_threshold,
        min_closed_frames=args.min_closed_frames,
    )
    session_logger = SessionLogger(
        path=log_path,
        emotion_labels=analyzer.emotion_labels,
        action_unit_labels=analyzer.action_unit_labels,
    )

    if analyzer.notice:
        print(analyzer.notice)
    print("Emotion log: {0}".format(log_path))

    runtime_kind, face_runtime = _create_face_runtime(mp, args, model_path)
    if face_runtime is None:
        return 1

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Cannot open camera {0}".format(args.camera))
        return 1

    frame_index = 0
    total_blinks = 0
    fps = None
    last_fps_timestamp = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed, stopping session.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_index += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections: List[FaceDetection] = []
            if runtime_kind == "solutions":
                mesh_result = face_runtime.process(rgb_frame)
                if mesh_result.multi_face_landmarks:
                    for face_landmarks in mesh_result.multi_face_landmarks:
                        points = _landmarks_to_pixels(face_landmarks, frame.shape[1], frame.shape[0])
                        detections.append(_build_detection(points))
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                mesh_result = face_runtime.detect_for_video(mp_image, timestamp_ms)
                if mesh_result.face_landmarks:
                    for face_landmarks in mesh_result.face_landmarks:
                        points = _landmarks_to_pixels(face_landmarks, frame.shape[1], frame.shape[0])
                        detections.append(_build_detection(points))

            tracked_faces = tracker.update(detections, frame_index)
            refreshed_results = {}
            latest_async_result = emotion_worker.get_latest()
            if latest_async_result is not None:
                if latest_async_result.error:
                    print("Emotion worker error: {0}".format(latest_async_result.error))
                refreshed_results = latest_async_result.face_emotions
            if emotion_worker.enabled and tracked_faces and (
                frame_index == 1 or frame_index % args.emotion_interval == 0
            ):
                emotion_worker.submit(frame, frame_index, tracked_faces)

            for detection, state in tracked_faces:
                avg_ear = (detection.left_ear + detection.right_ear) / 2.0
                blinked = blink_detector.update(state, avg_ear, frame_index)
                if blinked:
                    total_blinks += 1

                if state.face_id in refreshed_results:
                    state.last_emotion = refreshed_results[state.face_id]
                    state.emotion_updated_at = frame_index
                    session_logger.log_face(
                        timestamp_s=time.time(),
                        frame_index=frame_index,
                        face_id=state.face_id,
                        blink_count=state.blink_count,
                        ear=avg_ear,
                        bbox=detection.bbox,
                        emotion=state.last_emotion,
                    )
                elif (
                    not emotion_worker.enabled and
                    (
                        state.last_emotion is None or
                        frame_index - state.emotion_updated_at >= args.emotion_interval
                    )
                ):
                    padded_bbox = _bbox_with_padding(detection.bbox, frame.shape)
                    face_crop = _crop_face(frame, padded_bbox)
                    if face_crop is not None and face_crop.size:
                        state.last_emotion = analyzer.analyze(
                            face_bgr=face_crop,
                            landmarks_px=detection.landmarks_px,
                            bbox=detection.bbox,
                        )
                        state.emotion_updated_at = frame_index
                        session_logger.log_face(
                            timestamp_s=time.time(),
                            frame_index=frame_index,
                            face_id=state.face_id,
                            blink_count=state.blink_count,
                            ear=avg_ear,
                            bbox=detection.bbox,
                            emotion=state.last_emotion,
                        )

                _draw_face_overlay(frame, state, detection, frame_index, emotion_worker.status_text)

            if args.show_fps:
                now = time.time()
                frame_delta = now - last_fps_timestamp
                if frame_delta > 0:
                    fps = 1.0 / frame_delta
                last_fps_timestamp = now

            _draw_hud(
                frame,
                len(tracked_faces),
                fps if args.show_fps else None,
                analyzer,
                emotion_worker.status_text,
                emotion_worker.last_error,
            )

            cv2.imshow(WINDOW_NAME, frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord("q"):
                break
            if pressed_key == ord("r"):
                total_blinks = 0
                for state in tracker.states.values():
                    state.blink_count = 0

    finally:
        emotion_worker.close()
        session_logger.close()
        face_runtime.close()
        cap.release()
        cv2.destroyAllWindows()

    print("--- Session Summary ---")
    print("Frames processed: {0}".format(frame_index))
    print("Unique faces seen: {0}".format(tracker.next_face_id - 1))
    print("Total blinks counted: {0}".format(total_blinks))
    print("Emotion backend used: {0}".format(analyzer.backend_name))
    return 0
