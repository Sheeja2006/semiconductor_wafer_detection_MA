"""
threshold_tuning.py
====================
Cost-sensitive decision threshold analysis for the wafer defect
predictor. Even though the classes are balanced (~50/50), the
COST of a false negative (predicting Pass on a wafer that actually
Fails - i.e. a defective wafer slips through) is typically higher
than the cost of a false positive (predicting Fail on a wafer that
actually Passes - i.e. an unnecessary rework/scrap). The default
0.5 probability threshold does not account for this asymmetry.

This script sweeps candidate thresholds and reports precision/recall
trade-offs so a threshold can be chosen deliberately based on
business cost, rather than defaulting to 0.5.

Author: ML Engineering Team
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_logger, load_dataset, get_feature_target_split, split_train_test, REPORTS_DIR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def sweep_thresholds(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """
    Evaluate precision/recall/F1 and the confusion matrix breakdown
    across a range of candidate decision thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels (1 = Pass, 0 = Fail).
    y_proba : array-like
        Predicted probability of the Pass class.
    thresholds : array-like, optional
        Candidate thresholds to evaluate. Defaults to 0.30-0.70 in
        steps of 0.05.

    Returns
    -------
    pd.DataFrame
        One row per threshold with precision/recall/F1 for the Pass
        class, plus the count of missed defects (false Pass) and
        unnecessary rework/scrap (false Fail).
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # fn here = predicted Fail(0) when actually Pass(1) -> unnecessary rework/scrap
        # fp here = predicted Pass(1) when actually Fail(0) -> MISSED DEFECT (costly)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append({
            "Threshold": round(t, 2),
            "Precision (Pass)": round(precision, 4),
            "Recall (Pass)": round(recall, 4),
            "F1 (Pass)": round(f1, 4),
            "Missed_Defects_FalsePass": int(fp),   # defective wafer predicted Pass - costly
            "Unnecessary_Rework_FalseFail": int(fn),  # good wafer predicted Fail - wasteful but safe
        })

    return pd.DataFrame(rows)


def plot_threshold_tradeoff(sweep_df: pd.DataFrame, save_path: str) -> None:
    """Plot missed-defects vs. unnecessary-rework counts across thresholds."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(sweep_df["Threshold"], sweep_df["Missed_Defects_FalsePass"],
              "o-", color="crimson", label="Missed Defects (False Pass)")
    ax1.plot(sweep_df["Threshold"], sweep_df["Unnecessary_Rework_FalseFail"],
              "o-", color="steelblue", label="Unnecessary Rework (False Fail)")
    ax1.set_xlabel("Decision Threshold (probability of Pass)")
    ax1.set_ylabel("Count on Test Set")
    ax1.set_title("Cost Trade-off Across Decision Thresholds")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved threshold trade-off plot to {save_path}")


def recommend_threshold(sweep_df: pd.DataFrame, cost_ratio: float = 3.0) -> dict:
    """
    Recommend a threshold that minimizes total weighted cost, where
    missing a real defect (False Pass) is assumed to cost
    `cost_ratio`x more than an unnecessary rework/scrap (False Fail).

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of sweep_thresholds().
    cost_ratio : float
        How many times costlier a missed defect is vs. unnecessary
        rework. Default 3.0 is a placeholder assumption - adjust
        based on real fab cost data.

    Returns
    -------
    dict
        The recommended row (as a dict) plus the assumed cost ratio.
    """
    sweep_df = sweep_df.copy()
    sweep_df["Weighted_Cost"] = (
        sweep_df["Missed_Defects_FalsePass"] * cost_ratio
        + sweep_df["Unnecessary_Rework_FalseFail"] * 1.0
    )
    best_row = sweep_df.loc[sweep_df["Weighted_Cost"].idxmin()]
    return {
        "recommended_threshold": float(best_row["Threshold"]),
        "cost_ratio_assumed": cost_ratio,
        "missed_defects": int(best_row["Missed_Defects_FalsePass"]),
        "unnecessary_rework": int(best_row["Unnecessary_Rework_FalseFail"]),
        "precision": float(best_row["Precision (Pass)"]),
        "recall": float(best_row["Recall (Pass)"]),
    }


def run_threshold_analysis() -> None:
    """Standalone runner: loads the saved model and reports the full analysis."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    df = load_dataset()
    X, y = get_feature_target_split(df)
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)
    X_test_scaled = scaler.transform(X_test)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    sweep_df = sweep_thresholds(y_test, y_proba)
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (default classifier threshold = 0.50)")
    print("=" * 70)
    print(sweep_df.to_string(index=False))

    sweep_path = os.path.join(REPORTS_DIR, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    logger.info(f"Saved threshold sweep table to {sweep_path}")

    plot_path = os.path.join(REPORTS_DIR, "threshold_tradeoff.png")
    plot_threshold_tradeoff(sweep_df, plot_path)

    recommendation = recommend_threshold(sweep_df, cost_ratio=3.0)
    print("\n" + "=" * 70)
    print(f"RECOMMENDATION (assuming missed defects cost {recommendation['cost_ratio_assumed']}x "
          f"more than unnecessary rework)")
    print("=" * 70)
    for k, v in recommendation.items():
        print(f"{k}: {v}")

    with open(os.path.join(REPORTS_DIR, "threshold_recommendation.txt"), "w") as f:
        f.write("DECISION THRESHOLD RECOMMENDATION\n")
        f.write("=" * 50 + "\n")
        f.write(
            f"Assumed cost ratio (missed defect : unnecessary rework) = "
            f"{recommendation['cost_ratio_assumed']}:1\n\n"
        )
        for k, v in recommendation.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    run_threshold_analysis()
