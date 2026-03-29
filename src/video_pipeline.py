import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from src.ear import compute_ear, average_ear
from src.blink_detector import calibrate_threshold, detect_blinks_from_ear_signal
from src.feature_extraction import compute_blink_metrics
from src.dbsp_classifier import classify_dbsp
from src.visualization import generate_ear_plot

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def validate_video(size_bytes: int, duration_s: Optional[float], total_frames: Optional[int]) -> list[str]:
    errors = []
    if size_bytes > 500_000_000:
        errors.append("File exceeds the 500MB limit.")
    if duration_s is not None and duration_s < 10:
        errors.append("Video is too short. Please upload a video of at least 10 seconds.")
    if duration_s is not None and duration_s > 300:
        errors.append("Video exceeds the 5-minute limit.")
    return errors


def _extract_eye_landmarks(face_landmarks, indices, w, h):
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        points.append([lm.x * w, lm.y * h])
    return np.array(points)


def process_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"errors": ["Could not read this video file. Please check the file and try again."]}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    validation_errors = validate_video(size_bytes=0, duration_s=duration_s, total_frames=total_frames)
    if validation_errors:
        cap.release()
        return {"errors": validation_errors}

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    ear_signal = []
    frames_with_face = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            left = _extract_eye_landmarks(face, LEFT_EYE, w, h)
            right = _extract_eye_landmarks(face, RIGHT_EYE, w, h)
            ear = average_ear(left, right)
            ear_signal.append(ear)
            frames_with_face += 1
        else:
            ear_signal.append(None)
        frame_index += 1

    cap.release()
    face_mesh.close()

    valid_ears = [e for e in ear_signal if e is not None]

    if frames_with_face < 10:
        return {"errors": [
            "Could not detect a face in this video. "
            "Please ensure the face is clearly visible and facing the camera."
        ]}

    threshold, ear_open = calibrate_threshold(valid_ears)

    continuous_signal = []
    last_ear = valid_ears[0] if valid_ears else 0.3
    for e in ear_signal:
        if e is not None:
            last_ear = e
        continuous_signal.append(last_ear)

    blinks = detect_blinks_from_ear_signal(continuous_signal, threshold, fps)
    metrics = compute_blink_metrics(blinks, fps, len(continuous_signal), ear_open)
    dbsp_class = classify_dbsp(metrics["blink_rate"], metrics["incomplete_blink_pct"])
    ear_plot_png = generate_ear_plot(continuous_signal, blinks, threshold, fps)

    sample_frames = []
    cap = cv2.VideoCapture(video_path)
    frames_to_capture = set()

    if valid_ears:
        open_frame_idx = 15
        frames_to_capture.add(("open", open_frame_idx))

    for i, b in enumerate(blinks[:3]):
        mid = (b["start_frame"] + b["end_frame"]) // 2
        frames_to_capture.add(("blink", mid))

    target_indices = {idx for _, idx in frames_to_capture}
    label_map = {idx: label for label, idx in frames_to_capture}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in target_indices:
            _, jpeg = cv2.imencode(".jpg", frame)
            sample_frames.append({"label": label_map[fi], "frame_index": fi, "jpeg_bytes": jpeg.tobytes()})
        fi += 1
    cap.release()

    return {
        "metrics": metrics, "dbsp_class": dbsp_class,
        "ear_plot_png": ear_plot_png, "sample_frames": sample_frames, "errors": [],
    }
