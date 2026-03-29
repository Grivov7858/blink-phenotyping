import numpy as np
from typing import Optional

def calibrate_threshold(ear_values: list[float], min_frames: int = 10) -> tuple[float, Optional[float]]:
    calibration = ear_values[:60]
    if len(calibration) < min_frames:
        return 0.21, None
    ear_open = float(np.median(calibration))
    threshold = ear_open * 0.75
    return threshold, ear_open

def detect_blinks_from_ear_signal(signal: list[float], threshold: float, fps: float) -> list[dict]:
    min_frames = max(3, round(3 * fps / 30))
    blinks = []
    in_blink = False
    start = 0
    ear_min = float("inf")

    for i, ear in enumerate(signal):
        if ear < threshold:
            if not in_blink:
                in_blink = True
                start = i
                ear_min = ear
            else:
                ear_min = min(ear_min, ear)
        else:
            if in_blink:
                duration = i - start
                if duration >= min_frames:
                    blinks.append({"start_frame": start, "end_frame": i - 1, "ear_min": ear_min})
                in_blink = False
                ear_min = float("inf")

    if in_blink:
        duration = len(signal) - start
        if duration >= min_frames:
            blinks.append({"start_frame": start, "end_frame": len(signal) - 1, "ear_min": ear_min})

    return blinks
