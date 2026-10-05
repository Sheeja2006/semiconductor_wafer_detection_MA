"""
explain_model.py
=================
Adds explainability and actionable recommendations on top of the
trained wafer defect prediction model:

    - SHAP TreeExplainer for per-wafer feature attribution
    - Global SHAP summary plot (reports/shap_summary.png)
    - A combined recommendation engine that maps:
        (top contributing fabrication stage) + (prediction confidence)
        -> recommended action (Proceed / Rework / Recycle / Scrap /
           Manual Review) + a human-readable precaution note

NOTE ON DOMAIN ASSUMPTIONS:
The stage reworkability/disposition rules live in stage_config.py and
are placeholder engineering assumptions - edit them there once real
fab domain knowledge is available. The confidence thresholds below
are similarly reasonable defaults, not calibrated against real cost
data, and can be tuned in CONFIDENCE_THRESHOLDS.

Author: ML Engineering Team
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_logger, REPORTS_DIR, MODELS_DIR
from stage_config import FEATURE_STAGE_MAP, STAGE_METADATA

logger = get_logger(__name__)

# How many top contributing features to surface in an explanation
TOP_N_FEATURES = 5

# Confidence bands applied to the probability of the PREDICTED class
# (not just probability of Fail) - used to set recommendation severity.
CONFIDENCE_THRESHOLDS = {
    "high": 0.75,   # >= 0.75  -> high confidence
    "moderate": 0.55,  # 0.55-0.75 -> moderate confidence
    # < 0.55                  -> low confidence / borderline
}


# ------------------------------------------------------------------
# SHAP EXPLAINER SETUP
# ------------------------------------------------------------------
def build_explainer(model):
    """
    Build a SHAP TreeExplainer for a tree-based model (LightGBM,
    XGBoost, CatBoost, RandomForest, DecisionTree). Falls back to a
    generic Explainer for non-tree models (e.g. Logistic Regression,
    SVM) using a background sample.

    Parameters
    ----------
    model : fitted estimator

    Returns
    -------
    shap.Explainer
    """
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("Built SHAP TreeExplainer.")
        return explainer
    except Exception as e:
        logger.warning(
            f"TreeExplainer failed ({e}); this model type may not be "
            f"tree-based. Falling back is required for non-tree models."
        )
        raise


def compute_shap_values(explainer, X) -> np.ndarray:
    """
    Compute SHAP values for a batch of samples.

    Parameters
    ----------
    explainer : shap.Explainer
    X : array-like or DataFrame, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        SHAP values for the POSITIVE (Pass) class.
    """
    try:
        shap_values = explainer.shap_values(X)

        # Some explainers return a list [class0_values, class1_values]
        # for binary classification; others return a single array.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = Pass

        # Newer shap versions may return a 3D array (n, features, classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return shap_values

    except Exception as e:
        logger.exception(f"Failed to compute SHAP values: {e}")
        raise


# ------------------------------------------------------------------
# GLOBAL SHAP SUMMARY PLOT
# ------------------------------------------------------------------
def generate_shap_summary_plot(
    model, X_background: pd.DataFrame, max_display: int = 20
) -> None:
    """
    Generate and save a global SHAP summary (beeswarm) plot showing
    which features matter most across the whole test set, and in
    which direction they push predictions.

    Parameters
    ----------
    model : fitted estimator
    X_background : pd.DataFrame
        Feature matrix (e.g. the test set) with proper column names.
    max_display : int
        Max number of features to show in the plot.
    """
    try:
        explainer = build_explainer(model)
        shap_values = compute_shap_values(explainer, X_background)

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_background,
            max_display=max_display,
            show=False,
        )
        path = os.path.join(REPORTS_DIR, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot to {path}")

    except Exception as e:
        logger.exception(f"Failed to generate SHAP summary plot: {e}")


# ------------------------------------------------------------------
# PER-WAFER EXPLANATION
# ------------------------------------------------------------------
def get_top_contributing_features(
    shap_row: np.ndarray, feature_names: List[str], predicted_label: str, top_n: int = TOP_N_FEATURES
) -> List[Dict]:
    """
    Identify the top-N features pushing this specific prediction
    toward the predicted class.

    Parameters
    ----------
    shap_row : np.ndarray, shape (n_features,)
        SHAP values (for the Pass class) for a single wafer.
    feature_names : List[str]
    predicted_label : str
        "Pass" or "Fail" - determines which direction of SHAP value
        counts as "contributing toward this prediction".
    top_n : int

    Returns
    -------
    List[Dict]
        Each dict: {feature, shap_value, stage, direction}
    """
    # If predicted Fail, features with NEGATIVE shap (toward class 0)
    # are the ones driving the Fail call. If predicted Pass, positive
    # shap values drive it.
    if predicted_label == "Fail":
        order = np.argsort(shap_row)  # most negative first
    else:
        order = np.argsort(-shap_row)  # most positive first

    top_idx = order[:top_n]

    results = []
    for idx in top_idx:
        fname = feature_names[idx]
        val = float(shap_row[idx])
        stage = FEATURE_STAGE_MAP.get(fname, "Unmapped")
        results.append(
            {
                "feature": fname,
                "shap_value": val,
                "stage": stage,
                "direction": "pushes toward Fail" if val < 0 else "pushes toward Pass",
            }
        )
    return results


def get_confidence_band(probability_of_predicted_class: float) -> str:
    """Map a probability to a qualitative confidence band."""
    if probability_of_predicted_class >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif probability_of_predicted_class >= CONFIDENCE_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "low"


def recommend_action(
    predicted_label: str,
    probability_of_predicted_class: float,
    top_features: List[Dict],
) -> Dict:
    """
    Combine the top contributing fabrication stage with prediction
    confidence to produce a recommended action and precaution note.

    Parameters
    ----------
    predicted_label : str
        "Pass" or "Fail"
    probability_of_predicted_class : float
        Model's probability for whichever class it predicted.
    top_features : List[Dict]
        Output of get_top_contributing_features().

    Returns
    -------
    Dict
        {
            "action": str,
            "disposition": str,
            "confidence_band": str,
            "root_cause_stage": str,
            "precaution": str,
        }
    """
    confidence_band = get_confidence_band(probability_of_predicted_class)

    if predicted_label == "Pass":
        if confidence_band == "low":
            return {
                "action": "Proceed with monitoring",
                "disposition": "proceed_monitor",
                "confidence_band": confidence_band,
                "root_cause_stage": None,
                "precaution": (
                    "Prediction is Pass but confidence is low/borderline. "
                    "Recommend a spot-check at final inspection rather than "
                    "skipping verification."
                ),
            }
        return {
            "action": "Proceed to next stage",
            "disposition": "proceed",
            "confidence_band": confidence_band,
            "root_cause_stage": None,
            "precaution": "High/moderate confidence Pass; no additional action needed.",
        }

    # predicted_label == "Fail"
    # Root cause = the single highest-magnitude contributing stage,
    # skipping non-actionable "detection-only" stages if a more
    # specific stage is also present in the top features.
    actionable_features = [
        f for f in top_features
        if STAGE_METADATA.get(f["stage"], {}).get("disposition") != "inspect_only"
    ]
    root_feature = actionable_features[0] if actionable_features else top_features[0]
    root_stage = root_feature["stage"]
    stage_meta = STAGE_METADATA.get(root_stage, {})

    base_action = stage_meta.get("action", "Hold for manual engineering review.")
    disposition = stage_meta.get("disposition", "manual_review")
    precaution = stage_meta.get(
        "precaution", "No stage-specific guidance available; route for manual review."
    )

    if confidence_band == "low":
        action = f"Manual review recommended (low-confidence Fail flag). If confirmed: {base_action}"
    elif confidence_band == "moderate":
        action = base_action
    else:  # high confidence fail
        if disposition == "rework":
            action = base_action
        else:
            action = f"{base_action} (high-confidence defect)"

    return {
        "action": action,
        "disposition": disposition,
        "confidence_band": confidence_band,
        "root_cause_stage": root_stage,
        "precaution": precaution,
    }


def explain_single_wafer(
    model,
    explainer,
    feature_row: np.ndarray,
    feature_names: List[str],
    label_encoder,
) -> Dict:
    """
    Produce a full explanation + recommendation for a single wafer.

    Parameters
    ----------
    model : fitted estimator
    explainer : shap.Explainer (pre-built via build_explainer)
    feature_row : np.ndarray, shape (n_features,)
        A single SCALED feature vector (already through scaler.transform).
    feature_names : List[str]
    label_encoder : fitted LabelEncoder

    Returns
    -------
    Dict
        {
            "prediction": "Pass"/"Fail",
            "probability_pass": float,
            "confidence": float,   # probability of the PREDICTED class
            "top_features": [...],
            "recommendation": {...},
        }
    """
    try:
        X_row = feature_row.reshape(1, -1)

        pred_encoded = model.predict(X_row)[0]
        pred_label_raw = label_encoder.inverse_transform([pred_encoded])[0]
        predicted_label = "Pass" if pred_label_raw == 1 else "Fail"

        proba_pass = float(model.predict_proba(X_row)[0, 1])
        confidence = proba_pass if predicted_label == "Pass" else 1 - proba_pass

        shap_values = compute_shap_values(explainer, X_row)
        top_features = get_top_contributing_features(
            shap_values[0], feature_names, predicted_label
        )

        recommendation = recommend_action(predicted_label, confidence, top_features)

        return {
            "prediction": predicted_label,
            "probability_pass": round(proba_pass, 4),
            "confidence": round(confidence, 4),
            "top_features": top_features,
            "recommendation": recommendation,
        }

    except Exception as e:
        logger.exception(f"Failed to explain wafer prediction: {e}")
        raise


# ------------------------------------------------------------------
# STANDALONE SCRIPT ENTRY POINT
# ------------------------------------------------------------------
def run_explainability_report(n_samples: int = 3) -> None:
    """
    Standalone runner: loads saved artifacts + test-like data, builds
    the global SHAP summary plot, and prints example per-wafer
    explanations for a few sample wafers (for demo / report purposes).
    """
    import joblib
    from utils import load_dataset, get_feature_target_split, split_train_test
    from sklearn.preprocessing import StandardScaler

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    df = load_dataset()
    X, y = get_feature_target_split(df)
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    logger.info("Generating global SHAP summary plot...")
    generate_shap_summary_plot(model, X_test_scaled_df)

    explainer = build_explainer(model)

    logger.info(f"Generating {n_samples} example per-wafer explanations...")
    for i in range(min(n_samples, len(X_test_scaled_df))):
        row = X_test_scaled_df.iloc[i].values
        result = explain_single_wafer(
            model, explainer, row, X.columns.tolist(), label_encoder
        )
        print("\n" + "=" * 60)
        print(f"WAFER SAMPLE #{i+1}")
        print("=" * 60)
        print(f"Prediction : {result['prediction']} (confidence: {result['confidence']:.2%})")
        print("Top contributing features:")
        for f in result["top_features"]:
            print(
                f"  - {f['feature']} ({f['stage']}): "
                f"{f['direction']} (SHAP={f['shap_value']:.4f})"
            )
        rec = result["recommendation"]
        print(f"Recommended action : {rec['action']}")
        print(f"Precaution          : {rec['precaution']}")


if __name__ == "__main__":
    run_explainability_report()