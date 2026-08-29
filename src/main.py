"""Flask web interface and JSON API."""
import json
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from .currency import inr_per_usd, inr_to_usd
from .features import prediction_frame

ROOT = Path(__file__).resolve().parents[1]; ARTIFACTS = ROOT / "artifacts"

FORM_DEFAULTS = {
    "Company": "Dell", "TypeName": "Notebook", "Cpu_family": "Intel Core i5",
    "Gpu_vendor": "Intel", "OS": "Windows", "Ram": "8", "Weight": "1.8",
    "PPI": "141", "Cpu_speed_ghz": "2.5", "HDD": "0", "SSD": "512",
    "Flash": "0", "Hybrid": "0", "Touchscreen": "0", "IPS": "1",
    "Gpu_dedicated": "0",
}

def create_app(test_config=None):
    app = Flask(__name__, static_folder=str(ROOT / "static"), template_folder=str(ROOT / "templates")); app.config.update(test_config or {})
    def assets():
        if not (ARTIFACTS / "model.joblib").exists(): raise RuntimeError("Model not trained. Run: python train_model.py")
        return joblib.load(ARTIFACTS / "model.joblib"), joblib.load(ARTIFACTS / "options.joblib"), json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    def calculate(model, metadata, payload):
        log_price = float(model.predict(prediction_frame(payload))[0]); price_inr = float(np.exp(log_price)); q = metadata["test_metrics"]["interval_log_half_width"]; rate = inr_per_usd()
        return {"price_inr": price_inr, "lower_inr": float(np.exp(log_price-q)), "upper_inr": float(np.exp(log_price+q)), "price_usd": inr_to_usd(price_inr, rate), "rate": rate}
    @app.get("/")
    def home(): return render_template("index.html")
    @app.get("/health")
    def health():
        """Container/orchestrator health endpoint that also validates model assets."""
        try:
            _, _, metadata = assets()
            return jsonify({
                "status": "healthy",
                "model": metadata.get("selected_model"),
                "dataset_rows": metadata.get("dataset_rows"),
            })
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"status": "unhealthy", "error": str(exc)}), 503
    @app.route("/predict", methods=["GET", "POST"])
    def predict_page():
        model, options, metadata = assets(); result = error = None
        form_data = FORM_DEFAULTS.copy()
        if request.method == "POST":
            # Preserve the submitted values whether prediction succeeds or fails.
            form_data.update(request.form.to_dict())
            try: result = calculate(model, metadata, form_data)
            except (ValueError, TypeError) as exc: error = str(exc)
        return render_template(
            "prediction.html", options=options, result=result, error=error,
            form_data=form_data,
        )
    @app.post("/api/predict")
    def predict_api():
        model, _, metadata = assets()
        try:
            result = calculate(model, metadata, request.get_json(force=True))
            return jsonify({"predicted_price_inr": round(result["price_inr"], 2), "predicted_price_usd": round(result["price_usd"], 2), "prediction_interval_90_inr": [round(result["lower_inr"], 2), round(result["upper_inr"], 2)], "inr_per_usd": result["rate"], "currency_note": "USD is converted from the INR prediction; it is not a US-market model."})
        except (ValueError, TypeError) as exc: return jsonify({"error": str(exc)}), 400
    @app.get("/api/metrics")
    def metrics_api(): return jsonify(assets()[2])
    @app.get("/metrics")
    def metrics_page(): return render_template("metrics.html", metadata=assets()[2])
    @app.get("/methodology")
    def methodology(): return render_template("methodology.html", metadata=assets()[2])
    return app

app = create_app()
