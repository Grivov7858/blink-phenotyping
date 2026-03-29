# AI-Assisted Blink Phenotyping and Digital Blink Suppression Phenotype (DBSP)

## Purpose

A web-based tool for ophthalmological research on pediatric patients. Clinicians upload a face video recorded during a digital task (tablet/smartphone use) along with clinical data, and the system returns objective blink metrics and a Digital Blink Suppression Phenotype (DBSP) classification. A separate statistics page accepts accumulated patient data and produces a predictive diagnostic model with visualizations suitable for a research poster.

## Users

Clinicians with minimal technical experience, running the app locally on one machine. Single-user; concurrent requests are not supported. Interface in English.

## Architecture

Stateless Flask web application. No database, no persistent storage. Videos are processed in memory and discarded. Results are returned immediately and can be downloaded as CSV/PNG.

### Flow: Analyze Video

1. Clinician fills form with validated inputs:
   - Subject ID: free text, required
   - Age: integer, 1-18 years, required
   - Screen Time: decimal, 0-24 hours, required
   - Symptom Score: integer, 0-100 (OSDI scale), required
   - NIBUT: decimal, 0-30 seconds, required
2. Clinician uploads a video file (.mp4, .avi, .mov) — max 500MB, max 5 minutes duration
3. System processes the video using MediaPipe FaceMesh + EAR algorithm
4. System returns:
   - Blink metrics table (blink count, blink rate, incomplete blink %, mean interblink interval)
   - DBSP classification (adaptive / moderate / strong) with color-coded badge
   - EAR signal plot (PNG)
   - Sample blink frames (open eye vs blink)
5. Clinician can download a CSV row containing all clinical + blink data

### Flow: Run Statistics

1. Clinician uploads a CSV file matching the download schema exactly (columns: subject_id, age, screen_time_h, symptom_score, nibut_s, blink_count, blink_rate, incomplete_blink_pct, mean_interblink_interval, dbsp_class). System validates column names and shows a clear error if they don't match.
2. System requires minimum 20 rows to run logistic regression (fewer → error message explaining why).
3. System runs logistic regression with target: NIBUT < 7
4. Predictors: incomplete_blink_pct, blink_rate, screen_time_h, symptom_score
5. System returns: ROC curve, AUC, sensitivity, specificity, confusion matrix
6. Download plots as a ZIP file containing all PNGs

## Blink Detection Engine

### Technology

MediaPipe FaceMesh — 468 facial landmarks, runs in Python, no GPU required.

### Algorithm

1. Read video frame by frame using OpenCV
2. For each frame, detect face and extract eye landmarks via MediaPipe
3. Calculate Eye Aspect Ratio (EAR) for both eyes, average them
4. Calibrate EAR threshold per video: take median EAR from first 60 frames where a face is detected. Blink threshold = 75% of this calibrated EAR_open. If fewer than 10 usable frames in calibration window, fall back to fixed threshold of 0.21.
5. Detect blinks: EAR drops below threshold for 3+ consecutive frames (normalized to 30fps — if video is 60fps, require 6+ frames)
6. Mark blink start (EAR falls below threshold) and end (EAR returns above threshold)
7. For each blink, record EAR_min

### Incomplete Blink Classification

- Complete blink: EAR_min < 0.25 * EAR_open
- Incomplete blink: EAR_min >= 0.25 * EAR_open

### Metrics Computed Per Video

| Metric | Formula |
|--------|---------|
| blink_count | Total number of detected blinks |
| blink_rate | blink_count / video_duration_minutes |
| incomplete_blink_pct | incomplete_blinks / total_blinks * 100 |
| mean_interblink_interval | Mean time (seconds) between consecutive blinks |

### DBSP Classification

Classification uses a scoring approach to ensure every patient is classified. Compute a suppression score from blink_rate and incomplete_blink_pct, then classify:

**Blink rate score:**
- blink_rate > 12: 0 points
- 8 <= blink_rate <= 12: 1 point
- blink_rate < 8: 2 points

**Incomplete blink score:**
- incomplete_blink_pct < 20%: 0 points
- 20% <= incomplete_blink_pct <= 40%: 1 point
- incomplete_blink_pct > 40%: 2 points

**Total score → phenotype:**
- 0: Adaptive blinker
- 1-2: Moderate suppression
- 3-4: Strong DBSP

## Web Interface

### Page 1: Analyze Video

- Form fields: Subject ID, Age, Screen Time (hours), Symptom Score, NIBUT (seconds)
- File upload accepting .mp4, .avi, .mov
- "Analyze" button with loading spinner (processing takes 30-60 seconds)
- Results section: metrics table, DBSP badge (green/yellow/red), EAR plot, sample blink frames
- "Download CSV" button for a single row of all data

### Page 2: Run Statistics

- CSV file upload
- "Run Analysis" button
- Results: ROC curve plot, AUC score, sensitivity, specificity, confusion matrix plot
- Download all plots as a ZIP file containing all PNGs

### Styling

Clean, minimal, medical-looking UI. White background, simple layout, large buttons.

## Error Handling

| Scenario | User sees |
|----------|-----------|
| No face detected in video | "Could not detect a face in this video. Please ensure the face is clearly visible and facing the camera." |
| Zero blinks detected | Metrics returned with blink_count=0, blink_rate=0, incomplete_blink_pct=N/A, mean_interblink_interval=N/A, dbsp_class="Insufficient data" |
| Video too short (< 10 seconds) | "Video is too short. Please upload a video of at least 10 seconds." |
| Video exceeds 500MB | Rejected before reading into memory: "File exceeds the 500MB limit." |
| Video exceeds 5 minutes | Checked after opening with OpenCV but before blink detection: "Video exceeds the 5-minute limit." |
| Corrupted video file | "Could not read this video file. Please check the file and try again." |
| Statistics CSV missing columns | "CSV is missing required columns: [list]. Please use the exact format from the Download CSV button on the Analyze page." |
| Statistics CSV < 20 rows | "At least 20 patient records are needed to run statistical analysis. Current file has N rows." |

## Project Structure

```
doctor/
  app.py                     # Flask entry point + routes
  requirements.txt           # Python dependencies
  templates/
    base.html                # Shared layout
    analyze.html             # Video analysis page
    statistics.html          # Statistics page
  static/
    style.css                # Styling
  src/
    blink_detector.py        # MediaPipe + EAR blink detection
    feature_extraction.py    # Compute metrics from raw blink data
    dbsp_classifier.py       # DBSP phenotype classification
    statistics.py            # Logistic regression, ROC, confusion matrix
    visualization.py         # EAR plots, blink frame captures
```

## Dependencies

- flask
- opencv-python
- mediapipe
- numpy
- pandas
- scikit-learn
- matplotlib

## How to Run

```bash
pip install -r requirements.txt
python app.py
# Opens on http://localhost:5000
```

## Output Formats

### CSV Row (per patient)

```
subject_id,age,screen_time_h,symptom_score,nibut_s,blink_count,blink_rate,incomplete_blink_pct,mean_interblink_interval,dbsp_class
```

### Plots

- EAR signal over time (ear_signal.png)
- Sample blink frames with annotations
- ROC curve (roc_curve.png)
- Confusion matrix (confusion_matrix.png)
