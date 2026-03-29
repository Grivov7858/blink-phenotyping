# Blink Phenotyping Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask web app where clinicians upload face videos and clinical data to get blink metrics, DBSP classification, and statistical analysis.

**Architecture:** Stateless Flask app with Python backend. Video processing via MediaPipe FaceMesh + OpenCV. No database. Two pages: Analyze Video and Run Statistics.

**Tech Stack:** Python, Flask, OpenCV, MediaPipe, NumPy, Pandas, scikit-learn, Matplotlib

**Spec:** `docs/superpowers/specs/2026-03-24-blink-phenotyping-design.md`

---

## File Structure

```
doctor/
  app.py                        # Flask entry point, routes, form validation
  requirements.txt              # Python dependencies
  src/
    __init__.py                 # Package init
    ear.py                      # EAR calculation from landmarks
    blink_detector.py           # Frame-by-frame blink detection pipeline
    feature_extraction.py       # Compute blink metrics from raw detections
    dbsp_classifier.py          # DBSP scoring and classification
    statistics.py               # Logistic regression, ROC, confusion matrix
    visualization.py            # EAR signal plot, blink frame captures
  templates/
    base.html                   # Shared HTML layout
    analyze.html                # Video analysis page
    statistics.html             # Statistics page
  static/
    style.css                   # Styling
  tests/
    __init__.py
    test_ear.py                 # EAR calculation tests
    test_blink_detector.py      # Blink detection tests
    test_feature_extraction.py  # Metrics computation tests
    test_dbsp_classifier.py     # DBSP classification tests
    test_statistics.py          # Statistics module tests
    test_app.py                 # Flask route tests
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/dgrivov/Documents/doctor
git init
```

- [ ] **Step 2: Create requirements.txt**

```
flask==3.1.*
opencv-python==4.11.*
mediapipe==0.10.*
numpy==2.*
pandas==2.*
scikit-learn==1.6.*
matplotlib==3.*
pytest==8.*
```

- [ ] **Step 3: Create package init files**

Create `src/__init__.py` — empty file.
Create `tests/__init__.py` — empty file.

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 5: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.mp4
*.avi
*.mov
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py .gitignore
git commit -m "chore: project setup with dependencies"
```

---

### Task 2: EAR Calculation

**Files:**
- Create: `src/ear.py`
- Create: `tests/test_ear.py`

- [ ] **Step 1: Write failing tests for EAR calculation**

```python
# tests/test_ear.py
import numpy as np
from src.ear import compute_ear

def test_ear_open_eye():
    """Open eye landmarks should give EAR close to 0.3-0.4."""
    # Simulated open eye: vertical distances are significant
    # Using 6 landmark points: [p1, p2, p3, p4, p5, p6]
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    landmarks = np.array([
        [0.0, 0.5],   # p1 - left corner
        [0.2, 0.8],   # p2 - upper-left
        [0.5, 0.8],   # p3 - upper-right
        [0.7, 0.5],   # p4 - right corner
        [0.5, 0.2],   # p5 - lower-right
        [0.2, 0.2],   # p6 - lower-left
    ])
    ear = compute_ear(landmarks)
    assert 0.2 < ear < 1.0, f"Open eye EAR should be moderate, got {ear}"


def test_ear_closed_eye():
    """Closed eye landmarks should give EAR close to 0."""
    landmarks = np.array([
        [0.0, 0.5],   # p1
        [0.2, 0.51],  # p2 - almost same height as p6
        [0.5, 0.51],  # p3 - almost same height as p5
        [0.7, 0.5],   # p4
        [0.5, 0.49],  # p5
        [0.2, 0.49],  # p6
    ])
    ear = compute_ear(landmarks)
    assert ear < 0.1, f"Closed eye EAR should be near 0, got {ear}"


def test_ear_average_both_eyes():
    """average_ear should return mean of left and right EAR."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ear.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.ear'`

- [ ] **Step 3: Implement EAR calculation**

```python
# src/ear.py
import numpy as np


def compute_ear(landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 eye landmarks.

    landmarks: shape (6, 2) — [p1, p2, p3, p4, p5, p6]
    p1, p4 = horizontal corners
    p2, p3 = upper lid
    p5, p6 = lower lid

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
    """Return the average EAR of both eyes."""
    return (compute_ear(left_landmarks) + compute_ear(right_landmarks)) / 2.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ear.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/ear.py tests/test_ear.py
git commit -m "feat: EAR calculation from eye landmarks"
```

---

### Task 3: Blink Detector

**Files:**
- Create: `src/blink_detector.py`
- Create: `tests/test_blink_detector.py`

The blink detector takes a sequence of EAR values (one per frame) and detects blink events. It handles calibration, fps-normalized minimum frame count, and records blink start/end/EAR_min.

- [ ] **Step 1: Write failing tests for blink detection**

```python
# tests/test_blink_detector.py
from src.blink_detector import detect_blinks_from_ear_signal, calibrate_threshold


def test_calibrate_threshold_normal():
    """Median of open-eye EAR * 0.75."""
    ear_values = [0.30, 0.32, 0.28, 0.31, 0.29] * 12  # 60 values
    threshold, ear_open = calibrate_threshold(ear_values)
    # median ~ 0.30, threshold = 0.30 * 0.75 = 0.225
    assert 0.20 < threshold < 0.25
    assert 0.28 < ear_open < 0.32


def test_calibrate_threshold_fallback():
    """Too few usable frames → fallback threshold 0.21."""
    ear_values = [0.30] * 5  # Only 5 values, below 10 minimum
    threshold, ear_open = calibrate_threshold(ear_values)
    assert threshold == 0.21
    assert ear_open is None


def test_detect_single_blink_30fps():
    """A single dip below threshold for 4 frames = 1 blink at 30fps."""
    # Build signal: 20 frames open, 4 frames closed, 20 frames open
    ear_open = 0.30
    signal = [0.30] * 20 + [0.10, 0.08, 0.09, 0.10] + [0.30] * 20
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=30)
    assert len(blinks) == 1
    assert blinks[0]["start_frame"] == 20
    assert blinks[0]["end_frame"] == 23
    assert blinks[0]["ear_min"] == 0.08


def test_detect_no_blink_too_short():
    """A dip of only 2 frames at 30fps is NOT a blink (need 3+)."""
    signal = [0.30] * 20 + [0.10, 0.10] + [0.30] * 20
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=30)
    assert len(blinks) == 0


def test_detect_blink_60fps():
    """At 60fps, need 6+ frames below threshold."""
    signal = [0.30] * 20 + [0.10] * 6 + [0.30] * 20
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=60)
    assert len(blinks) == 1

    # 5 frames at 60fps should NOT count
    signal_short = [0.30] * 20 + [0.10] * 5 + [0.30] * 20
    blinks_short = detect_blinks_from_ear_signal(signal_short, threshold=0.225, fps=60)
    assert len(blinks_short) == 0


def test_detect_multiple_blinks():
    """Two separate dips = 2 blinks."""
    signal = (
        [0.30] * 10
        + [0.10] * 4
        + [0.30] * 10
        + [0.10] * 5
        + [0.30] * 10
    )
    blinks = detect_blinks_from_ear_signal(signal, threshold=0.225, fps=30)
    assert len(blinks) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_blink_detector.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement blink detector**

```python
# src/blink_detector.py
import numpy as np
from typing import Optional


def calibrate_threshold(
    ear_values: list[float], min_frames: int = 10
) -> tuple[float, Optional[float]]:
    """Calibrate blink threshold from open-eye EAR values.

    Takes up to 60 EAR values from calibration window.
    Returns (threshold, ear_open). If fewer than min_frames,
    falls back to (0.21, None).
    """
    calibration = ear_values[:60]
    if len(calibration) < min_frames:
        return 0.21, None
    ear_open = float(np.median(calibration))
    threshold = ear_open * 0.75
    return threshold, ear_open


def detect_blinks_from_ear_signal(
    signal: list[float], threshold: float, fps: float
) -> list[dict]:
    """Detect blinks from a sequence of EAR values.

    A blink is a contiguous run of frames where EAR < threshold,
    lasting at least min_frames (3 at 30fps, scaled by fps).

    Returns list of dicts: {start_frame, end_frame, ear_min}
    """
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
                    blinks.append({
                        "start_frame": start,
                        "end_frame": i - 1,
                        "ear_min": ear_min,
                    })
                in_blink = False
                ear_min = float("inf")

    # Handle blink at end of signal
    if in_blink:
        duration = len(signal) - start
        if duration >= min_frames:
            blinks.append({
                "start_frame": start,
                "end_frame": len(signal) - 1,
                "ear_min": ear_min,
            })

    return blinks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_blink_detector.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/blink_detector.py tests/test_blink_detector.py
git commit -m "feat: blink detection from EAR signal with fps normalization"
```

---

### Task 4: Feature Extraction

**Files:**
- Create: `src/feature_extraction.py`
- Create: `tests/test_feature_extraction.py`

Computes blink metrics from raw blink detections.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_feature_extraction.py
from src.feature_extraction import compute_blink_metrics


def test_metrics_normal():
    """Normal case with multiple blinks."""
    blinks = [
        {"start_frame": 30, "end_frame": 34, "ear_min": 0.05},
        {"start_frame": 150, "end_frame": 155, "ear_min": 0.15},
        {"start_frame": 300, "end_frame": 305, "ear_min": 0.04},
    ]
    fps = 30.0
    total_frames = 1800  # 60 seconds
    ear_open = 0.30

    metrics = compute_blink_metrics(blinks, fps, total_frames, ear_open)

    assert metrics["blink_count"] == 3
    assert metrics["blink_rate"] == 3.0  # 3 blinks / 1 minute
    # ear_min 0.05 < 0.075 (0.25*0.30) → complete
    # ear_min 0.15 >= 0.075 → incomplete
    # ear_min 0.04 < 0.075 → complete
    assert abs(metrics["incomplete_blink_pct"] - 33.33) < 1.0
    # interblink intervals: (150-34)/30=3.87s, (300-155)/30=4.83s
    assert 4.0 < metrics["mean_interblink_interval"] < 4.5


def test_metrics_zero_blinks():
    """Zero blinks should return safe defaults."""
    metrics = compute_blink_metrics([], 30.0, 1800, 0.30)
    assert metrics["blink_count"] == 0
    assert metrics["blink_rate"] == 0.0
    assert metrics["incomplete_blink_pct"] is None
    assert metrics["mean_interblink_interval"] is None


def test_metrics_single_blink():
    """Single blink: no interblink interval."""
    blinks = [{"start_frame": 30, "end_frame": 34, "ear_min": 0.05}]
    metrics = compute_blink_metrics(blinks, 30.0, 1800, 0.30)
    assert metrics["blink_count"] == 1
    assert metrics["mean_interblink_interval"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_feature_extraction.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement feature extraction**

```python
# src/feature_extraction.py
from typing import Optional


def compute_blink_metrics(
    blinks: list[dict],
    fps: float,
    total_frames: int,
    ear_open: Optional[float],
) -> dict:
    """Compute blink metrics from detected blinks.

    Returns dict with: blink_count, blink_rate, incomplete_blink_pct,
    mean_interblink_interval.
    """
    blink_count = len(blinks)
    duration_minutes = total_frames / fps / 60.0
    blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0.0

    if blink_count == 0:
        return {
            "blink_count": 0,
            "blink_rate": 0.0,
            "incomplete_blink_pct": None,
            "mean_interblink_interval": None,
        }

    # Incomplete blink classification
    if ear_open is not None:
        incomplete_threshold = 0.25 * ear_open
        incomplete_count = sum(
            1 for b in blinks if b["ear_min"] >= incomplete_threshold
        )
        incomplete_blink_pct = round(incomplete_count / blink_count * 100, 2)
    else:
        incomplete_blink_pct = None

    # Interblink intervals
    if blink_count >= 2:
        intervals = []
        for i in range(1, len(blinks)):
            gap_frames = blinks[i]["start_frame"] - blinks[i - 1]["end_frame"]
            intervals.append(gap_frames / fps)
        mean_ibi = round(sum(intervals) / len(intervals), 2)
    else:
        mean_ibi = None

    return {
        "blink_count": blink_count,
        "blink_rate": round(blink_rate, 2),
        "incomplete_blink_pct": incomplete_blink_pct,
        "mean_interblink_interval": mean_ibi,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_feature_extraction.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/feature_extraction.py tests/test_feature_extraction.py
git commit -m "feat: blink metrics computation from raw detections"
```

---

### Task 5: DBSP Classifier

**Files:**
- Create: `src/dbsp_classifier.py`
- Create: `tests/test_dbsp_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dbsp_classifier.py
from src.dbsp_classifier import classify_dbsp


def test_adaptive():
    """High blink rate, low incomplete → adaptive."""
    result = classify_dbsp(blink_rate=15.0, incomplete_blink_pct=10.0)
    assert result == "adaptive"


def test_strong():
    """Low blink rate, high incomplete → strong."""
    result = classify_dbsp(blink_rate=5.0, incomplete_blink_pct=50.0)
    assert result == "strong"


def test_moderate_mixed():
    """Mid-range values → moderate."""
    result = classify_dbsp(blink_rate=10.0, incomplete_blink_pct=30.0)
    assert result == "moderate"


def test_moderate_boundary():
    """One high, one low score → moderate (score 2)."""
    result = classify_dbsp(blink_rate=5.0, incomplete_blink_pct=10.0)
    assert result == "moderate"


def test_strong_both_max():
    """Both worst scores → strong (score 4)."""
    result = classify_dbsp(blink_rate=3.0, incomplete_blink_pct=60.0)
    assert result == "strong"


def test_insufficient_data():
    """None for incomplete_blink_pct → insufficient data."""
    result = classify_dbsp(blink_rate=10.0, incomplete_blink_pct=None)
    assert result == "insufficient_data"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dbsp_classifier.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement DBSP classifier**

```python
# src/dbsp_classifier.py
from typing import Optional


def classify_dbsp(
    blink_rate: float, incomplete_blink_pct: Optional[float]
) -> str:
    """Classify Digital Blink Suppression Phenotype using scoring.

    Blink rate score: >12 → 0, 8-12 → 1, <8 → 2
    Incomplete blink score: <20% → 0, 20-40% → 1, >40% → 2
    Total: 0 → adaptive, 1-2 → moderate, 3-4 → strong
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dbsp_classifier.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/dbsp_classifier.py tests/test_dbsp_classifier.py
git commit -m "feat: DBSP phenotype classification with scoring"
```

---

### Task 6: Visualization

**Files:**
- Create: `src/visualization.py`
- Create: `tests/test_visualization.py`

Generates EAR signal plot and captures blink frames from the video.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_visualization.py
import io
from src.visualization import generate_ear_plot, generate_stats_plots


def test_generate_ear_plot_returns_png():
    """EAR plot should return PNG bytes."""
    ear_signal = [0.30] * 50 + [0.10] * 5 + [0.30] * 50
    blinks = [{"start_frame": 50, "end_frame": 54, "ear_min": 0.10}]
    threshold = 0.225
    fps = 30.0

    png_bytes = generate_ear_plot(ear_signal, blinks, threshold, fps)

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_generate_stats_plots_returns_dict():
    """Stats plots should return dict of name → PNG bytes."""
    # Minimal fake data for logistic regression
    import numpy as np
    np.random.seed(42)
    y_true = [0] * 15 + [1] * 15
    y_prob = list(np.random.uniform(0, 0.4, 15)) + list(np.random.uniform(0.6, 1.0, 15))

    plots = generate_stats_plots(y_true, y_prob)

    assert "roc_curve.png" in plots
    assert "confusion_matrix.png" in plots
    assert plots["roc_curve.png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert plots["confusion_matrix.png"][:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_visualization.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement visualization**

```python
# src/visualization.py
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def generate_ear_plot(
    ear_signal: list[float],
    blinks: list[dict],
    threshold: float,
    fps: float,
) -> bytes:
    """Generate EAR signal plot with blink markers. Returns PNG bytes."""
    times = [i / fps for i in range(len(ear_signal))]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, ear_signal, color="#2196F3", linewidth=0.8)
    ax.axhline(y=threshold, color="#F44336", linestyle="--", linewidth=0.8, label="Threshold")

    for b in blinks:
        start_t = b["start_frame"] / fps
        end_t = b["end_frame"] / fps
        ax.axvspan(start_t, end_t, alpha=0.2, color="#F44336")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EAR")
    ax.set_title("Eye Aspect Ratio Over Time")
    ax.legend()
    ax.set_ylim(0, max(ear_signal) * 1.2 if ear_signal else 0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_stats_plots(
    y_true: list[int], y_prob: list[float]
) -> dict[str, bytes]:
    """Generate ROC curve and confusion matrix plots. Returns dict of name → PNG bytes."""
    plots = {}

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    plots["roc_curve.png"] = buf.read()

    # Confusion matrix
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["NIBUT >= 7", "NIBUT < 7"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    plots["confusion_matrix.png"] = buf.read()

    return plots
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_visualization.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/visualization.py tests/test_visualization.py
git commit -m "feat: EAR signal and statistics plot generation"
```

---

### Task 7: Statistics Module

**Files:**
- Create: `src/statistics.py`
- Create: `tests/test_statistics.py`

Logistic regression with validation, returns metrics and predictions for plotting.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_statistics.py
import pandas as pd
import numpy as np
from src.statistics import validate_csv, run_logistic_regression


def test_validate_csv_valid():
    """Valid CSV passes validation."""
    df = pd.DataFrame({
        "subject_id": range(25),
        "age": [10] * 25,
        "screen_time_h": [3.0] * 25,
        "symptom_score": [20] * 25,
        "nibut_s": [7.5] * 25,
        "blink_count": [15] * 25,
        "blink_rate": [10.0] * 25,
        "incomplete_blink_pct": [25.0] * 25,
        "mean_interblink_interval": [4.0] * 25,
        "dbsp_class": ["moderate"] * 25,
    })
    errors = validate_csv(df)
    assert errors == []


def test_validate_csv_missing_columns():
    """Missing columns should be reported."""
    df = pd.DataFrame({"subject_id": [1], "age": [10]})
    errors = validate_csv(df)
    assert len(errors) == 1
    assert "missing required columns" in errors[0].lower()


def test_validate_csv_too_few_rows():
    """Fewer than 20 rows should be reported."""
    df = pd.DataFrame({
        "subject_id": range(5),
        "age": [10] * 5,
        "screen_time_h": [3.0] * 5,
        "symptom_score": [20] * 5,
        "nibut_s": [7.5] * 5,
        "blink_count": [15] * 5,
        "blink_rate": [10.0] * 5,
        "incomplete_blink_pct": [25.0] * 5,
        "mean_interblink_interval": [4.0] * 5,
        "dbsp_class": ["moderate"] * 5,
    })
    errors = validate_csv(df)
    assert len(errors) == 1
    assert "20" in errors[0]


def test_run_logistic_regression():
    """Logistic regression should return metrics and predictions."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "screen_time_h": np.random.uniform(1, 8, n),
        "symptom_score": np.random.randint(0, 50, n),
        "blink_rate": np.random.uniform(5, 18, n),
        "incomplete_blink_pct": np.random.uniform(5, 60, n),
        "nibut_s": np.random.uniform(3, 15, n),
    })
    result = run_logistic_regression(df)
    assert "auc" in result
    assert "sensitivity" in result
    assert "specificity" in result
    assert "y_true" in result
    assert "y_prob" in result
    assert 0.0 <= result["auc"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_statistics.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement statistics module**

```python
# src/statistics.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_predict


REQUIRED_COLUMNS = [
    "subject_id", "age", "screen_time_h", "symptom_score", "nibut_s",
    "blink_count", "blink_rate", "incomplete_blink_pct",
    "mean_interblink_interval", "dbsp_class",
]

PREDICTOR_COLUMNS = ["incomplete_blink_pct", "blink_rate", "screen_time_h", "symptom_score"]


def validate_csv(df: pd.DataFrame) -> list[str]:
    """Validate uploaded CSV for statistics. Returns list of error messages."""
    errors = []
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"CSV is missing required columns: {sorted(missing)}")
        return errors  # Can't check rows if columns are wrong
    if len(df) < 20:
        errors.append(
            f"At least 20 patient records are needed to run statistical analysis. "
            f"Current file has {len(df)} rows."
        )
    return errors


def run_logistic_regression(df: pd.DataFrame) -> dict:
    """Run logistic regression: target = NIBUT < 7.

    Returns dict with: auc, sensitivity, specificity, y_true, y_prob.
    """
    X = df[PREDICTOR_COLUMNS].values
    y = (df["nibut_s"] < 7).astype(int).values

    model = LogisticRegression(max_iter=1000, random_state=42)

    # Use cross-validated predictions for honest evaluation
    y_prob = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_score = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": round(auc_score, 3),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "y_true": y.tolist(),
        "y_prob": y_prob.tolist(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_statistics.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/statistics.py tests/test_statistics.py
git commit -m "feat: logistic regression with CSV validation"
```

---

### Task 8: Video Processing Pipeline

**Files:**
- Create: `src/video_pipeline.py`
- Create: `tests/test_video_pipeline.py`

This is the integration layer: takes a video file path, runs MediaPipe FaceMesh frame by frame, extracts EAR signal, runs blink detection, computes metrics, classifies DBSP, generates visualizations.

- [ ] **Step 1: Write failing test**

```python
# tests/test_video_pipeline.py
from src.video_pipeline import validate_video


def test_validate_video_too_large():
    """Files over 500MB should be rejected."""
    errors = validate_video(size_bytes=600_000_000, duration_s=None, total_frames=None)
    assert any("500MB" in e for e in errors)


def test_validate_video_too_short():
    """Videos under 10 seconds should be rejected."""
    errors = validate_video(size_bytes=1_000_000, duration_s=5.0, total_frames=150)
    assert any("10 seconds" in e for e in errors)


def test_validate_video_too_long():
    """Videos over 5 minutes should be rejected."""
    errors = validate_video(size_bytes=1_000_000, duration_s=360.0, total_frames=10800)
    assert any("5-minute" in e for e in errors)


def test_validate_video_ok():
    """Valid video passes."""
    errors = validate_video(size_bytes=50_000_000, duration_s=60.0, total_frames=1800)
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_video_pipeline.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement video pipeline**

```python
# src/video_pipeline.py
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from src.ear import compute_ear, average_ear
from src.blink_detector import calibrate_threshold, detect_blinks_from_ear_signal
from src.feature_extraction import compute_blink_metrics
from src.dbsp_classifier import classify_dbsp
from src.visualization import generate_ear_plot

# MediaPipe FaceMesh eye landmark indices (for 6-point EAR)
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def validate_video(
    size_bytes: int,
    duration_s: Optional[float],
    total_frames: Optional[int],
) -> list[str]:
    """Validate video constraints. Returns list of error messages."""
    errors = []
    if size_bytes > 500_000_000:
        errors.append("File exceeds the 500MB limit.")
    if duration_s is not None and duration_s < 10:
        errors.append("Video is too short. Please upload a video of at least 10 seconds.")
    if duration_s is not None and duration_s > 300:
        errors.append("Video exceeds the 5-minute limit.")
    return errors


def _extract_eye_landmarks(face_landmarks, indices, w, h):
    """Extract 6 eye landmarks as numpy array of (x, y) pixel coords."""
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        points.append([lm.x * w, lm.y * h])
    return np.array(points)


def process_video(video_path: str) -> dict:
    """Process a video file end-to-end.

    Returns dict with:
        metrics: blink metrics dict
        dbsp_class: str
        ear_plot_png: bytes
        blink_frames: list of (frame_index, jpeg_bytes) for sample blinks
        errors: list of str (empty if OK)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"errors": ["Could not read this video file. Please check the file and try again."]}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    validation_errors = validate_video(
        size_bytes=0,  # Already checked by Flask before this point
        duration_s=duration_s,
        total_frames=total_frames,
    )
    if validation_errors:
        cap.release()
        return {"errors": validation_errors}

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ear_signal = []
    frames_with_face = 0
    blink_sample_frames = {}  # frame_index → jpeg bytes (captured later)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            left = _extract_eye_landmarks(face, LEFT_EYE, w, h)
            right = _extract_eye_landmarks(face, RIGHT_EYE, w, h)
            ear = average_ear(left, right)
            ear_signal.append(ear)
            frames_with_face += 1
        else:
            ear_signal.append(None)

        frame_index += 1

    cap.release()
    face_mesh.close()

    # Filter out None values for calibration (use only frames with detected face)
    valid_ears = [e for e in ear_signal if e is not None]

    if frames_with_face < 10:
        return {"errors": [
            "Could not detect a face in this video. "
            "Please ensure the face is clearly visible and facing the camera."
        ]}

    # Calibrate and detect
    threshold, ear_open = calibrate_threshold(valid_ears)

    # Build continuous signal: replace None with last known EAR for detection
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

    # Capture sample blink frames (first 3 blinks + one open-eye frame)
    sample_frames = []
    cap = cv2.VideoCapture(video_path)
    frames_to_capture = set()

    # One open-eye frame from early in the video
    if valid_ears:
        open_frame_idx = 15  # Early frame likely to have open eyes
        frames_to_capture.add(("open", open_frame_idx))

    # First 3 blink midpoints
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
            sample_frames.append({
                "label": label_map[fi],
                "frame_index": fi,
                "jpeg_bytes": jpeg.tobytes(),
            })
        fi += 1
    cap.release()

    return {
        "metrics": metrics,
        "dbsp_class": dbsp_class,
        "ear_plot_png": ear_plot_png,
        "sample_frames": sample_frames,
        "errors": [],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_video_pipeline.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/video_pipeline.py tests/test_video_pipeline.py
git commit -m "feat: video processing pipeline with MediaPipe integration"
```

---

### Task 9: Flask App — Routes and Form Validation

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write failing tests for routes**

```python
# tests/test_app.py
import io
import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_redirects_to_analyze(client):
    resp = client.get("/")
    assert resp.status_code == 302 or resp.status_code == 200


def test_analyze_page_loads(client):
    resp = client.get("/analyze")
    assert resp.status_code == 200
    assert b"Analyze" in resp.data


def test_statistics_page_loads(client):
    resp = client.get("/statistics")
    assert resp.status_code == 200
    assert b"Statistics" in resp.data


def test_analyze_missing_fields(client):
    """POST without required fields returns validation error."""
    resp = client.post("/analyze", data={})
    assert resp.status_code == 200
    assert b"required" in resp.data.lower() or b"error" in resp.data.lower()


def test_analyze_invalid_age(client):
    """Age outside 1-18 returns error."""
    data = {
        "subject_id": "001",
        "age": "25",
        "screen_time_h": "3.0",
        "symptom_score": "20",
        "nibut_s": "7.5",
    }
    video = (io.BytesIO(b"fake"), "test.mp4")
    resp = client.post("/analyze", data={**data, "video": video})
    assert b"1" in resp.data and b"18" in resp.data


def test_statistics_no_file(client):
    """POST without file returns error."""
    resp = client.post("/statistics", data={})
    assert resp.status_code == 200
    assert b"error" in resp.data.lower() or b"required" in resp.data.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_app.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement Flask app**

```python
# app.py
import io
import os
import tempfile
import zipfile
import base64
from flask import (
    Flask, render_template, request, redirect, url_for, send_file
)
import pandas as pd
from src.video_pipeline import process_video, validate_video
from src.statistics import validate_csv, run_logistic_regression
from src.visualization import generate_stats_plots


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

    @app.template_filter("b64encode")
    def b64encode_filter(s):
        return base64.b64encode(s.encode("utf-8")).decode("utf-8")

    @app.route("/")
    def index():
        return redirect(url_for("analyze"))

    @app.route("/analyze", methods=["GET", "POST"])
    def analyze():
        if request.method == "GET":
            return render_template("analyze.html")

        # Validate form fields
        errors = []
        subject_id = request.form.get("subject_id", "").strip()
        if not subject_id:
            errors.append("Subject ID is required.")

        age = request.form.get("age", "").strip()
        try:
            age_val = int(age)
            if age_val < 1 or age_val > 18:
                errors.append("Age must be between 1 and 18.")
        except ValueError:
            errors.append("Age must be a whole number between 1 and 18.")
            age_val = None

        screen_time = request.form.get("screen_time_h", "").strip()
        try:
            screen_time_val = float(screen_time)
            if screen_time_val < 0 or screen_time_val > 24:
                errors.append("Screen time must be between 0 and 24 hours.")
        except ValueError:
            errors.append("Screen time must be a number between 0 and 24.")
            screen_time_val = None

        symptom_score = request.form.get("symptom_score", "").strip()
        try:
            symptom_val = int(symptom_score)
            if symptom_val < 0 or symptom_val > 100:
                errors.append("Symptom score must be between 0 and 100.")
        except ValueError:
            errors.append("Symptom score must be a whole number between 0 and 100.")
            symptom_val = None

        nibut = request.form.get("nibut_s", "").strip()
        try:
            nibut_val = float(nibut)
            if nibut_val < 0 or nibut_val > 30:
                errors.append("NIBUT must be between 0 and 30 seconds.")
        except ValueError:
            errors.append("NIBUT must be a number between 0 and 30.")
            nibut_val = None

        video = request.files.get("video")
        if not video or video.filename == "":
            errors.append("Video file is required.")

        if errors:
            return render_template("analyze.html", errors=errors, form=request.form)

        # Save video to temp file for processing
        suffix = os.path.splitext(video.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            video.save(tmp)
            tmp_path = tmp.name

        try:
            result = process_video(tmp_path)
        finally:
            os.unlink(tmp_path)

        if result["errors"]:
            return render_template("analyze.html", errors=result["errors"], form=request.form)

        # Build CSV row
        csv_data = {
            "subject_id": subject_id,
            "age": age_val,
            "screen_time_h": screen_time_val,
            "symptom_score": symptom_val,
            "nibut_s": nibut_val,
            "blink_count": result["metrics"]["blink_count"],
            "blink_rate": result["metrics"]["blink_rate"],
            "incomplete_blink_pct": result["metrics"]["incomplete_blink_pct"],
            "mean_interblink_interval": result["metrics"]["mean_interblink_interval"],
            "dbsp_class": result["dbsp_class"],
        }

        # Encode images as base64 for embedding in HTML
        ear_plot_b64 = base64.b64encode(result["ear_plot_png"]).decode("utf-8")
        sample_frames_b64 = [
            {
                "label": f["label"],
                "data": base64.b64encode(f["jpeg_bytes"]).decode("utf-8"),
            }
            for f in result["sample_frames"]
        ]

        # Build CSV string for download
        csv_df = pd.DataFrame([csv_data])
        csv_string = csv_df.to_csv(index=False)

        return render_template(
            "analyze.html",
            results=result["metrics"],
            dbsp_class=result["dbsp_class"],
            ear_plot_b64=ear_plot_b64,
            sample_frames=sample_frames_b64,
            csv_string=csv_string,
            form=request.form,
        )

    @app.route("/statistics", methods=["GET", "POST"])
    def statistics():
        if request.method == "GET":
            return render_template("statistics.html")

        csv_file = request.files.get("csv_file")
        if not csv_file or csv_file.filename == "":
            return render_template("statistics.html", errors=["CSV file is required."])

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            return render_template("statistics.html", errors=[
                "Could not read CSV file. Please check the format."
            ])

        errors = validate_csv(df)
        if errors:
            return render_template("statistics.html", errors=errors)

        result = run_logistic_regression(df)
        plots = generate_stats_plots(result["y_true"], result["y_prob"])

        # Encode plots as base64
        plots_b64 = {
            name: base64.b64encode(data).decode("utf-8")
            for name, data in plots.items()
        }

        # Build ZIP for download
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for name, data in plots.items():
                zf.writestr(name, data)
        zip_buf.seek(0)
        zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

        return render_template(
            "statistics.html",
            stats=result,
            plots=plots_b64,
            zip_b64=zip_b64,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_app.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: Flask app with analyze and statistics routes"
```

---

### Task 10: HTML Templates and Styling

**Files:**
- Create: `templates/base.html`
- Create: `templates/analyze.html`
- Create: `templates/statistics.html`
- Create: `static/style.css`

No tests for this task — it's pure markup/styling.

- [ ] **Step 1: Create base.html**

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blink Phenotyping Tool</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <nav>
        <div class="nav-inner">
            <span class="logo">Blink Phenotyping</span>
            <div class="nav-links">
                <a href="{{ url_for('analyze') }}" class="{{ 'active' if request.endpoint == 'analyze' }}">Analyze Video</a>
                <a href="{{ url_for('statistics') }}" class="{{ 'active' if request.endpoint == 'statistics' }}">Statistics</a>
            </div>
        </div>
    </nav>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

- [ ] **Step 2: Create analyze.html**

```html
<!-- templates/analyze.html -->
{% extends "base.html" %}
{% block content %}
<h1>Analyze Video</h1>

{% if errors %}
<div class="error-box">
    {% for e in errors %}
    <p>{{ e }}</p>
    {% endfor %}
</div>
{% endif %}

{% if not results %}
<form method="POST" enctype="multipart/form-data" id="analyze-form">
    <div class="form-group">
        <label for="subject_id">Subject ID</label>
        <input type="text" id="subject_id" name="subject_id" required
               value="{{ form.get('subject_id', '') if form else '' }}">
    </div>
    <div class="form-group">
        <label for="age">Age (years, 1-18)</label>
        <input type="number" id="age" name="age" min="1" max="18" required
               value="{{ form.get('age', '') if form else '' }}">
    </div>
    <div class="form-group">
        <label for="screen_time_h">Screen Time (hours/day)</label>
        <input type="number" id="screen_time_h" name="screen_time_h"
               min="0" max="24" step="0.1" required
               value="{{ form.get('screen_time_h', '') if form else '' }}">
    </div>
    <div class="form-group">
        <label for="symptom_score">Symptom Score (OSDI, 0-100)</label>
        <input type="number" id="symptom_score" name="symptom_score"
               min="0" max="100" required
               value="{{ form.get('symptom_score', '') if form else '' }}">
    </div>
    <div class="form-group">
        <label for="nibut_s">NIBUT (seconds, 0-30)</label>
        <input type="number" id="nibut_s" name="nibut_s"
               min="0" max="30" step="0.1" required
               value="{{ form.get('nibut_s', '') if form else '' }}">
    </div>
    <div class="form-group">
        <label for="video">Video File (.mp4, .avi, .mov)</label>
        <input type="file" id="video" name="video"
               accept=".mp4,.avi,.mov" required>
    </div>
    <button type="submit" id="submit-btn">Analyze</button>
    <div id="spinner" class="spinner hidden">Processing video... This may take up to a minute.</div>
</form>

<script>
document.getElementById("analyze-form").addEventListener("submit", function() {
    document.getElementById("submit-btn").disabled = true;
    document.getElementById("spinner").classList.remove("hidden");
});
</script>
{% endif %}

{% if results %}
<div class="results-section">
    <h2>Blink Metrics</h2>
    <table class="metrics-table">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Blink Count</td><td>{{ results.blink_count }}</td></tr>
        <tr><td>Blink Rate (per min)</td><td>{{ results.blink_rate }}</td></tr>
        <tr>
            <td>Incomplete Blink %</td>
            <td>{{ results.incomplete_blink_pct if results.incomplete_blink_pct is not none else 'N/A' }}</td>
        </tr>
        <tr>
            <td>Mean Interblink Interval (s)</td>
            <td>{{ results.mean_interblink_interval if results.mean_interblink_interval is not none else 'N/A' }}</td>
        </tr>
    </table>

    <h2>DBSP Classification</h2>
    <div class="dbsp-badge dbsp-{{ dbsp_class }}">
        {{ dbsp_class | upper }}
    </div>

    <h2>EAR Signal</h2>
    <img src="data:image/png;base64,{{ ear_plot_b64 }}" alt="EAR Signal" class="plot-img">

    {% if sample_frames %}
    <h2>Sample Frames</h2>
    <div class="frame-grid">
        {% for f in sample_frames %}
        <div class="frame-card">
            <img src="data:image/jpeg;base64,{{ f.data }}" alt="{{ f.label }}">
            <span class="frame-label">{{ f.label | upper }}</span>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <h2>Download</h2>
    <a href="data:text/csv;base64,{{ csv_string | b64encode }}"
       download="blink_data.csv" class="btn">Download CSV</a>
    <a href="{{ url_for('analyze') }}" class="btn btn-secondary">Analyze Another</a>
</div>
{% endif %}
{% endblock %}
```

Note: The `b64encode` filter needs to be registered in `app.py`. Add this inside `create_app()`:

```python
@app.template_filter("b64encode")
def b64encode_filter(s):
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")
```

- [ ] **Step 3: Create statistics.html**

```html
<!-- templates/statistics.html -->
{% extends "base.html" %}
{% block content %}
<h1>Statistical Analysis</h1>

{% if errors %}
<div class="error-box">
    {% for e in errors %}
    <p>{{ e }}</p>
    {% endfor %}
</div>
{% endif %}

{% if not stats %}
<p class="instructions">
    Upload a CSV file containing accumulated patient data to run logistic regression analysis.
    The CSV must match the format from the "Download CSV" button on the Analyze page.
</p>
<form method="POST" enctype="multipart/form-data">
    <div class="form-group">
        <label for="csv_file">Patient Data CSV</label>
        <input type="file" id="csv_file" name="csv_file" accept=".csv" required>
    </div>
    <button type="submit">Run Analysis</button>
</form>
{% endif %}

{% if stats %}
<div class="results-section">
    <h2>Model Performance</h2>
    <table class="metrics-table">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>AUC</td><td>{{ stats.auc }}</td></tr>
        <tr><td>Sensitivity</td><td>{{ stats.sensitivity }}</td></tr>
        <tr><td>Specificity</td><td>{{ stats.specificity }}</td></tr>
    </table>

    <h2>ROC Curve</h2>
    <img src="data:image/png;base64,{{ plots['roc_curve.png'] }}" alt="ROC Curve" class="plot-img">

    <h2>Confusion Matrix</h2>
    <img src="data:image/png;base64,{{ plots['confusion_matrix.png'] }}" alt="Confusion Matrix" class="plot-img">

    <h2>Download</h2>
    <a href="data:application/zip;base64,{{ zip_b64 }}"
       download="statistics_plots.zip" class="btn">Download All Plots (ZIP)</a>
    <a href="{{ url_for('statistics') }}" class="btn btn-secondary">Run Another Analysis</a>
</div>
{% endif %}
{% endblock %}
```

- [ ] **Step 4: Create style.css**

```css
/* static/style.css */
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

nav {
    background: #fff;
    border-bottom: 2px solid #2196F3;
    padding: 0 2rem;
}

.nav-inner {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
}

.logo {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2196F3;
}

.nav-links a {
    text-decoration: none;
    color: #666;
    margin-left: 2rem;
    padding: 0.5rem 0;
    border-bottom: 2px solid transparent;
}

.nav-links a.active {
    color: #2196F3;
    border-bottom-color: #2196F3;
}

main {
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 2rem;
}

h1 { margin-bottom: 1.5rem; color: #1a1a1a; }
h2 { margin: 2rem 0 1rem; color: #1a1a1a; }

.form-group {
    margin-bottom: 1rem;
}

.form-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.3rem;
    color: #555;
}

.form-group input {
    width: 100%;
    max-width: 400px;
    padding: 0.6rem 0.8rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 1rem;
}

button, .btn {
    display: inline-block;
    padding: 0.7rem 2rem;
    background: #2196F3;
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    text-decoration: none;
    margin-top: 1rem;
    margin-right: 0.5rem;
}

button:hover, .btn:hover { background: #1976D2; }
button:disabled { background: #aaa; cursor: not-allowed; }

.btn-secondary {
    background: #757575;
}
.btn-secondary:hover { background: #616161; }

.error-box {
    background: #ffebee;
    border: 1px solid #ef9a9a;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    color: #c62828;
}

.metrics-table {
    border-collapse: collapse;
    width: 100%;
    max-width: 500px;
}

.metrics-table th, .metrics-table td {
    border: 1px solid #ddd;
    padding: 0.6rem 1rem;
    text-align: left;
}

.metrics-table th { background: #f5f5f5; font-weight: 600; }

.dbsp-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
}

.dbsp-adaptive { background: #4CAF50; }
.dbsp-moderate { background: #FF9800; }
.dbsp-strong { background: #F44336; }
.dbsp-insufficient_data { background: #9E9E9E; }

.plot-img {
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 0.5rem 0;
}

.frame-grid {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.frame-card {
    text-align: center;
}

.frame-card img {
    width: 200px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.frame-label {
    display: block;
    margin-top: 0.3rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: #555;
}

.spinner {
    margin-top: 1rem;
    color: #2196F3;
    font-weight: 600;
}

.hidden { display: none; }

.instructions {
    margin-bottom: 1.5rem;
    color: #666;
}
```

- [ ] **Step 5: Verify the app starts and pages load**

```bash
cd /home/dgrivov/Documents/doctor && python -c "from app import create_app; app = create_app(); print('App created OK')"
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add templates/ static/ app.py
git commit -m "feat: HTML templates and CSS styling for web interface"
```

---

### Task 11: Integration Test and Final Verification

**Files:**
- Modify: `app.py` (add b64encode filter if not yet added)

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 2: Manual smoke test**

Start the app and verify both pages load in a browser:

```bash
python app.py
```

Visit `http://localhost:5000/analyze` and `http://localhost:5000/statistics`.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final integration and cleanup"
```
