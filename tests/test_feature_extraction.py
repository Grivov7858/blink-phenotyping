from src.feature_extraction import compute_blink_metrics

def test_metrics_normal():
    blinks = [
        {"start_frame": 30, "end_frame": 34, "ear_min": 0.05},
        {"start_frame": 150, "end_frame": 155, "ear_min": 0.15},
        {"start_frame": 300, "end_frame": 305, "ear_min": 0.04},
    ]
    fps = 30.0
    total_frames = 1800
    ear_open = 0.30
    metrics = compute_blink_metrics(blinks, fps, total_frames, ear_open)
    assert metrics["blink_count"] == 3
    assert metrics["blink_rate"] == 3.0
    assert abs(metrics["incomplete_blink_pct"] - 33.33) < 1.0
    assert 4.0 < metrics["mean_interblink_interval"] < 4.5

def test_metrics_zero_blinks():
    metrics = compute_blink_metrics([], 30.0, 1800, 0.30)
    assert metrics["blink_count"] == 0
    assert metrics["blink_rate"] == 0.0
    assert metrics["incomplete_blink_pct"] is None
    assert metrics["mean_interblink_interval"] is None

def test_metrics_single_blink():
    blinks = [{"start_frame": 30, "end_frame": 34, "ear_min": 0.05}]
    metrics = compute_blink_metrics(blinks, 30.0, 1800, 0.30)
    assert metrics["blink_count"] == 1
    assert metrics["mean_interblink_interval"] is None
