import pandas as pd
import numpy as np
from src.statistics import validate_csv, run_logistic_regression

def test_validate_csv_valid():
    df = pd.DataFrame({
        "subject_id": range(25), "age": [10] * 25, "screen_time_h": [3.0] * 25,
        "symptom_score": [20] * 25, "nibut_s": [7.5] * 25, "blink_count": [15] * 25,
        "blink_rate": [10.0] * 25, "incomplete_blink_pct": [25.0] * 25,
        "mean_interblink_interval": [4.0] * 25, "dbsp_class": ["moderate"] * 25,
    })
    errors = validate_csv(df)
    assert errors == []

def test_validate_csv_missing_columns():
    df = pd.DataFrame({"subject_id": [1], "age": [10]})
    errors = validate_csv(df)
    assert len(errors) == 1
    assert "missing required columns" in errors[0].lower()

def test_validate_csv_too_few_rows():
    df = pd.DataFrame({
        "subject_id": range(5), "age": [10] * 5, "screen_time_h": [3.0] * 5,
        "symptom_score": [20] * 5, "nibut_s": [7.5] * 5, "blink_count": [15] * 5,
        "blink_rate": [10.0] * 5, "incomplete_blink_pct": [25.0] * 5,
        "mean_interblink_interval": [4.0] * 5, "dbsp_class": ["moderate"] * 5,
    })
    errors = validate_csv(df)
    assert len(errors) == 1
    assert "20" in errors[0]

def test_run_logistic_regression():
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
