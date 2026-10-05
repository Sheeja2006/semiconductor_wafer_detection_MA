"""
app.py
======
Minimal Flask application exposing the trained wafer defect
prediction model as a REST API. Loads best_model.pkl, scaler.pkl and
label_encoder.pkl from models/ using joblib.load().

Run with:
    python app.py

Endpoints:
    GET  /              -> health check
    POST /predict       -> JSON body: {"features": [f1, f2, ..., fN]}
                            returns predicted class + probability
"""

import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_DIR = os.path.join(BASE_DIR, "training")

# Make training/ importable so we can reuse explain_model.py and
# stage_config.py without duplicating logic in the Flask app.
sys.path.insert(0, TRAINING_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Load artifacts once at startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    logger.info("Model, scaler, and label encoder loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model artifacts: {e}")
    model, scaler, label_encoder = None, None, None

# Feature column order must match training. Derived from the dataset
# at startup so /explain can label SHAP values with real feature names.
try:
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "processed_dataset.csv")
    _df_cols = pd.read_csv(DATASET_PATH, nrows=1)
    _drop_cols = [c for c in ["Wafer_ID", "Final_Result", "Root_Cause_Stage",
                               "Freeze_Wafer", "Rework_Required"] if c in _df_cols.columns]
    FEATURE_NAMES = _df_cols.drop(columns=_drop_cols).columns.tolist()
except Exception as e:
    logger.exception(f"Failed to derive feature names: {e}")
    FEATURE_NAMES = None

# Build the SHAP explainer once at startup (expensive to rebuild per request)
try:
    from explain_model import build_explainer, explain_single_wafer
    explainer = build_explainer(model) if model is not None else None
    logger.info("SHAP explainer built successfully.")
except Exception as e:
    logger.exception(f"Failed to build SHAP explainer: {e}")
    explainer = None



@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    status = "ready" if model is not None else "model not loaded"
    return jsonify({"status": status})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict wafer PASS/FAIL from a JSON payload of feature values.

    Expected JSON body:
        {"features": [v1, v2, ..., vN]}   # same order/columns as training

    Returns
    -------
    JSON: {"prediction": "Pass"/"Fail", "probability": float}
    """
    try:
        if model is None or scaler is None or label_encoder is None:
            return jsonify({"error": "Model artifacts not loaded on server."}), 500

        payload = request.get_json(force=True)
        features = payload.get("features")

        if features is None:
            return jsonify({"error": "'features' key missing from request body."}), 400

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred_encoded = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_scaled)[0, 1])

        result_label = "Pass" if pred_label == 1 else "Fail"

        return jsonify({"prediction": result_label, "probability": proba})

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Predict wafer PASS/FAIL AND return an explanation + recommended
    action, using SHAP feature attribution mapped to fabrication
    stages (see training/stage_config.py).

    Expected JSON body:
        {"features": [v1, v2, ..., vN]}   # same order/columns as training

    Returns
    -------
    JSON:
        {
            "prediction": "Pass"/"Fail",
            "probability_pass": float,
            "confidence": float,
            "top_features": [
                {"feature": str, "stage": str, "shap_value": float, "direction": str},
                ...
            ],
            "recommendation": {
                "action": str,
                "disposition": str,
                "confidence_band": str,
                "root_cause_stage": str or null,
                "precaution": str
            }
        }
    """
    try:
        if model is None or scaler is None or label_encoder is None:
            return jsonify({"error": "Model artifacts not loaded on server."}), 500
        if explainer is None or FEATURE_NAMES is None:
            return jsonify({"error": "Explainability engine not available on server."}), 500

        payload = request.get_json(force=True)
        features = payload.get("features")

        if features is None:
            return jsonify({"error": "'features' key missing from request body."}), 400
        if len(features) != len(FEATURE_NAMES):
            return jsonify({
                "error": f"Expected {len(FEATURE_NAMES)} features, got {len(features)}."
            }), 400

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)[0]

        result = explain_single_wafer(
            model, explainer, X_scaled, FEATURE_NAMES, label_encoder
        )

        return jsonify(result)

    except Exception as e:
        logger.exception(f"Explanation failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
