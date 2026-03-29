import io
import os
import tempfile
import zipfile
import base64
from flask import Flask, render_template, request, redirect, url_for, send_file
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

        csv_data = {
            "subject_id": subject_id, "age": age_val, "screen_time_h": screen_time_val,
            "symptom_score": symptom_val, "nibut_s": nibut_val,
            "blink_count": result["metrics"]["blink_count"],
            "blink_rate": result["metrics"]["blink_rate"],
            "incomplete_blink_pct": result["metrics"]["incomplete_blink_pct"],
            "mean_interblink_interval": result["metrics"]["mean_interblink_interval"],
            "dbsp_class": result["dbsp_class"],
        }

        ear_plot_b64 = base64.b64encode(result["ear_plot_png"]).decode("utf-8")
        sample_frames_b64 = [
            {"label": f["label"], "data": base64.b64encode(f["jpeg_bytes"]).decode("utf-8")}
            for f in result["sample_frames"]
        ]

        csv_df = pd.DataFrame([csv_data])
        csv_string = csv_df.to_csv(index=False)

        return render_template(
            "analyze.html", results=result["metrics"], dbsp_class=result["dbsp_class"],
            ear_plot_b64=ear_plot_b64, sample_frames=sample_frames_b64,
            csv_string=csv_string, form=request.form,
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
            return render_template("statistics.html", errors=["Could not read CSV file. Please check the format."])

        errors = validate_csv(df)
        if errors:
            return render_template("statistics.html", errors=errors)

        result = run_logistic_regression(df)
        plots = generate_stats_plots(result["y_true"], result["y_prob"])

        plots_b64 = {name: base64.b64encode(data).decode("utf-8") for name, data in plots.items()}

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for name, data in plots.items():
                zf.writestr(name, data)
        zip_buf.seek(0)
        zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

        return render_template("statistics.html", stats=result, plots=plots_b64, zip_b64=zip_b64)

    return app


if __name__ == "__main__":
    import webbrowser
    import threading

    app = create_app()
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(host="127.0.0.1", port=5000)
