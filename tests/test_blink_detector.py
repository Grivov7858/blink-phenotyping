from src.blink_detector import detect_blinks_from_ear_signal, calibrate_threshold

def test_calibrate_threshold_normal():
    ear_values = [0.30, 0.32, 0.28, 0.31, 0.29] * 12  # 60 values
    threshold, ear_open = calibrate_threshold(ear_values)
    assert 0.20 < threshold < 0.25
    assert 0.28 < ear_open < 0.32

def test_calibrate_threshold_fallback():
    ear_values = [0.30] * 5
    threshold, ear_open = calibrate_threshold(ear_values)
    assert threshold == 0.21
    assert ear_open is None

def test_detect_single_blink_30fps():
    signal = [0.30] * 20 + [0.10, 0.08, 0.09, 0.10] + [0.30] * 20
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=30)
    assert len(blinks) == 1
    assert blinks[0]["start_frame"] == 20
    assert blinks[0]["end_frame"] == 23
    assert blinks[0]["ear_min"] == 0.08

def test_detect_no_blink_too_short():
    signal = [0.30] * 20 + [0.10, 0.10] + [0.30] * 20
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=30)
    assert len(blinks) == 0

def test_detect_blink_60fps():
    signal = [0.30] * 20 + [0.10] * 6 + [0.30] * 20
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=60)
    assert len(blinks) == 1
    signal_short = [0.30] * 20 + [0.10] * 5 + [0.30] * 20
    blinks_short = detect_blinks_from_ear_signal(signal_short, threshold=0.225, fps=60)
    assert len(blinks_short) == 0

def test_detect_multiple_blinks():
    signal = [0.30] * 10 + [0.10] * 4 + [0.30] * 10 + [0.10] * 5 + [0.30] * 10
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=30)
    assert len(blinks) == 2
