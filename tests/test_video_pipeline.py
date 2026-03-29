from src.video_pipeline import validate_video

def test_validate_video_too_large():
    errors = validate_video(size_bytes=600_000_000, duration_s=None, total_frames=None)
    assert any("500MB" in e for e in errors)

def test_validate_video_too_short():
    errors = validate_video(size_bytes=1_000_000, duration_s=5.0, total_frames=150)
    assert any("10 seconds" in e for e in errors)

def test_validate_video_too_long():
    errors = validate_video(size_bytes=1_000_000, duration_s=360.0, total_frames=10800)
    assert any("5-minute" in e for e in errors)

def test_validate_video_ok():
    errors = validate_video(size_bytes=50_000_000, duration_s=60.0, total_frames=1800)
    assert errors == []
