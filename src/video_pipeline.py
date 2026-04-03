import os
import sys
import bz2
import cv2
import numpy as np
import dlib
from typing import Optional
from src.ear import compute_ear, average_ear
from src.blink_detector import calibrate_threshold, detect_blinks_from_ear_signal
from src.feature_extraction import compute_blink_metrics
from src.dbsp_classifier import classify_dbsp
from src.visualization import generate_ear_plot

# dlib 68-landmark eye indices (0-indexed)
# Each eye has 6 points in the order: [corner_left, upper_left, upper_right, corner_right, lower_right, lower_left]
RIGHT_EYE = [36, 37, 38, 39, 40, 41]
LEFT_EYE = [42, 43, 44, 45, 46, 47]

# Shape predictor model path
_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"


def _get_model_path() -> str:
    """Find the shape predictor model file.

    Looks in: same dir as this file, project root, bundled PyInstaller path.
    Downloads automatically if not found.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), _MODEL_FILENAME),
        os.path.join(os.path.dirname(__file__), "..", _MODEL_FILENAME),
        os.path.join(os.path.dirname(__file__), "..", "models", _MODEL_FILENAME),
    ]

    # PyInstaller bundle path
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, _MODEL_FILENAME))
        candidates.append(os.path.join(os.path.dirname(sys.executable), _MODEL_FILENAME))

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    # Auto-download if not found
    download_dir = os.path.join(os.path.dirname(__file__), "..")
    return _download_model(download_dir)


def _download_model(dest_dir: str) -> str:
    """Download and extract the dlib shape predictor model."""
    import urllib.request

    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = os.path.join(dest_dir, _MODEL_FILENAME + ".bz2")
    dat_path = os.path.join(dest_dir, _MODEL_FILENAME)

    print(f"Downloading face landmark model (~100MB)... ", end="", flush=True)
    urllib.request.urlretrieve(url, bz2_path)
    print("done. Extracting... ", end="", flush=True)

    with bz2.BZ2File(bz2_path, "rb") as src, open(dat_path, "wb") as dst:
        dst.write(src.read())

    os.remove(bz2_path)
    print("done.")
    return dat_path


# Lazy-loaded globals
_detector = None
_predictor = None


def _get_detector_and_predictor():
    """Lazy-load dlib face detector and shape predictor."""
    global _detector, _predictor
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
        model_path = _get_model_path()
        _predictor = dlib.shape_predictor(model_path)
    return _detector, _predictor


def validate_video(size_bytes: int, duration_s: Optional[float], total_frames: Optional[int]) -> list[str]:
    errors = []
    if size_bytes > 500_000_000:
        errors.append("File exceeds the 500MB limit.")
    if duration_s is not None and duration_s < 10:
        errors.append("Video is too short. Please upload a video of at least 10 seconds.")
    if duration_s is not None and duration_s > 300:
        errors.append("Video exceeds the 5-minute limit.")
    return errors


def _extract_eye_landmarks(shape, indices):
    """Extract 6 eye landmarks as numpy array of (x, y) pixel coords."""
    points = []
    for idx in indices:
        p = shape.part(idx)
        points.append([p.x, p.y])
    return np.array(points, dtype=np.float64)


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

    detector, predictor = _get_detector_and_predictor()

    ear_signal = []
    frames_with_face = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        if len(faces) > 0:
            shape = predictor(gray, faces[0])
            left = _extract_eye_landmarks(shape, LEFT_EYE)
            right = _extract_eye_landmarks(shape, RIGHT_EYE)
            ear = average_ear(left, right)
            ear_signal.append(ear)
            frames_with_face += 1
        else:
            ear_signal.append(None)

        frame_index += 1

    cap.release()

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

    # Capture sample blink frames
    sample_frames = []
    cap = cv2.VideoCapture(video_path)
    frames_to_capture = set()

    if valid_ears:
        frames_to_capture.add(("open", 15))

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
