from typing import Optional


def compute_blink_metrics(blinks: list[dict], fps: float, total_frames: int, ear_open: Optional[float]) -> dict:
    blink_count = len(blinks)
    duration_minutes = total_frames / fps / 60.0
    blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0.0

    if blink_count == 0:
        return {"blink_count": 0, "blink_rate": 0.0, "incomplete_blink_pct": None, "mean_interblink_interval": None}

    if ear_open is not None:
        incomplete_threshold = 0.25 * ear_open
        incomplete_count = sum(1 for b in blinks if b["ear_min"] >= incomplete_threshold)
        incomplete_blink_pct = round(incomplete_count / blink_count * 100, 2)
    else:
        incomplete_blink_pct = None

    if blink_count >= 2:
        intervals = []
        for i in range(1, len(blinks)):
            gap_frames = blinks[i]["start_frame"] - blinks[i - 1]["end_frame"]
            intervals.append(gap_frames / fps)
        mean_ibi = round(sum(intervals) / len(intervals), 2)
    else:
        mean_ibi = None

    return {"blink_count": blink_count, "blink_rate": round(blink_rate, 2), "incomplete_blink_pct": incomplete_blink_pct, "mean_interblink_interval": mean_ibi}
