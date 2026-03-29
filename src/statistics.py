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
    errors = []
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"CSV is missing required columns: {sorted(missing)}")
        return errors
    if len(df) < 20:
        errors.append(f"At least 20 patient records are needed to run statistical analysis. Current file has {len(df)} rows.")
    return errors

def run_logistic_regression(df: pd.DataFrame) -> dict:
    X = df[PREDICTOR_COLUMNS].values
    y = (df["nibut_s"] < 7).astype(int).values
    model = LogisticRegression(max_iter=1000, random_state=42)
    y_prob = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc_score = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": round(auc_score, 3), "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3), "y_true": y.tolist(), "y_prob": y_prob.tolist(),
    }
