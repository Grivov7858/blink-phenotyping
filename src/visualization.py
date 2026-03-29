import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def generate_ear_plot(ear_signal: list[float], blinks: list[dict], threshold: float, fps: float) -> bytes:
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

def generate_stats_plots(y_true: list[int], y_prob: list[float]) -> dict[str, bytes]:
    plots = {}
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
