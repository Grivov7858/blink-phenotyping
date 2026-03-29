from src.dbsp_classifier import classify_dbsp

def test_adaptive():
    result = classify_dbsp(blink_rate=15.0, incomplete_blink_pct=10.0)
    assert result == "adaptive"

def test_strong():
    result = classify_dbsp(blink_rate=5.0, incomplete_blink_pct=50.0)
    assert result == "strong"

def test_moderate_mixed():
    result = classify_dbsp(blink_rate=10.0, incomplete_blink_pct=30.0)
    assert result == "moderate"

def test_moderate_boundary():
    result = classify_dbsp(blink_rate=5.0, incomplete_blink_pct=10.0)
    assert result == "moderate"

def test_strong_both_max():
    result = classify_dbsp(blink_rate=3.0, incomplete_blink_pct=60.0)
    assert result == "strong"

def test_insufficient_data():
    result = classify_dbsp(blink_rate=10.0, incomplete_blink_pct=None)
    assert result == "insufficient_data"
