import numpy as np

def compute_ear(landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 eye landmarks.
    landmarks: shape (6, 2) — [p1, p2, p3, p4, p5, p6]
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = landmarks
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def average_ear(left_landmarks: np.ndarray, right_landmarks: np.ndarray) -> float:
    return (compute_ear(left_landmarks) + compute_ear(right_landmarks)) / 2.0
