import numpy as np
from src.ear import compute_ear

def test_ear_open_eye():
    landmarks = np.array([
        [0.0, 0.5], [0.2, 0.8], [0.5, 0.8],
        [0.7, 0.5], [0.5, 0.2], [0.2, 0.2],
    ])
    ear = compute_ear(landmarks)
    assert 0.2 < ear < 1.0, f"Open eye EAR should be moderate, got {ear}"

def test_ear_closed_eye():
    landmarks = np.array([
        [0.0, 0.5], [0.2, 0.51], [0.5, 0.51],
        [0.7, 0.5], [0.5, 0.49], [0.2, 0.49],
    ])
    ear = compute_ear(landmarks)
    assert ear < 0.1, f"Closed eye EAR should be near 0, got {ear}"

def test_ear_average_both_eyes():
    from src.ear import average_ear
    left = np.array([
        [0.0, 0.5], [0.2, 0.8], [0.5, 0.8],
        [0.7, 0.5], [0.5, 0.2], [0.2, 0.2],
    ])
    right = np.array([
        [0.0, 0.5], [0.2, 0.7], [0.5, 0.7],
        [0.7, 0.5], [0.5, 0.3], [0.2, 0.3],
    ])
    avg = average_ear(left, right)
    expected = (compute_ear(left) + compute_ear(right)) / 2
    assert abs(avg - expected) < 1e-6
