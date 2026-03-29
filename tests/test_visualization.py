import io
from src.visualization import generate_ear_plot, generate_stats_plots

def test_generate_ear_plot_returns_png():
    ear_signal = [0.30] * 50 + [0.10] * 5 + [0.30] * 50
    blinks = [{"start_frame": 50, "end_frame": 54, "ear_min": 0.10}]
    threshold = 0.225
    fps = 30.0
    png_bytes = generate_ear_plot(ear_signal, blinks, threshold, fps)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

def test_generate_stats_plots_returns_dict():
    import numpy as np
    np.random.seed(42)
    y_true = [0] * 15 + [1] * 15
    y_prob = list(np.random.uniform(0, 0.4, 15)) + list(np.random.uniform(0.6, 1.0, 15))
    plots = generate_stats_plots(y_true, y_prob)
    assert "roc_curve.png" in plots
    assert "confusion_matrix.png" in plots
    assert plots["roc_curve.png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert plots["confusion_matrix.png"][:8] == b"\x89PNG\r\n\x1a\n"
