"""
save_model.py
==============
Persists the final trained artifacts (best model, scaler, label
encoder) to disk using joblib, ready to be loaded directly inside a
Flask application via joblib.load().

Author: ML Engineering Team
"""

import os
import joblib

from utils import MODELS_DIR, get_logger

logger = get_logger(__name__)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


def save_artifacts(model, scaler, label_encoder) -> None:
    """
    Save the trained model, feature scaler, and target label encoder
    to the models/ directory.

    Parameters
    ----------
    model : fitted estimator
        The final (tuned) best model.
    scaler : fitted sklearn.preprocessing.StandardScaler
        Scaler fit on the training features.
    label_encoder : fitted sklearn.preprocessing.LabelEncoder
        Encoder fit on the target column.
    """
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        joblib.dump(model, BEST_MODEL_PATH)
        logger.info(f"Saved best model to {BEST_MODEL_PATH}")

        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Saved scaler to {SCALER_PATH}")

        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        logger.info(f"Saved label encoder to {LABEL_ENCODER_PATH}")

    except Exception as e:
        logger.exception(f"Failed to save model artifacts: {e}")
        raise


def load_artifacts():
    """
    Convenience loader (mirrors what the Flask app will do) to load
    back the saved model, scaler, and label encoder.

    Returns
    -------
    model, scaler, label_encoder
    """
    try:
        model = joblib.load(BEST_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("Loaded model, scaler, and label encoder successfully.")
        return model, scaler, label_encoder

    except Exception as e:
        logger.exception(f"Failed to load model artifacts: {e}")
        raise
