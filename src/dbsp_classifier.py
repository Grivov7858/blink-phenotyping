from typing import Optional

def classify_dbsp(blink_rate: float, incomplete_blink_pct: Optional[float]) -> str:
    if incomplete_blink_pct is None:
        return "insufficient_data"

    if blink_rate > 12:
        rate_score = 0
    elif blink_rate >= 8:
        rate_score = 1
    else:
        rate_score = 2

    if incomplete_blink_pct < 20:
        incomplete_score = 0
    elif incomplete_blink_pct <= 40:
        incomplete_score = 1
    else:
        incomplete_score = 2

    total = rate_score + incomplete_score
    if total == 0:
        return "adaptive"
    elif total <= 2:
        return "moderate"
    else:
        return "strong"
