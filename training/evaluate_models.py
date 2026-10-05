"""
evaluate_models.py
===================
Evaluation utilities for the Wafer Defect Prediction pipeline:
    - Per-model metric computation (Accuracy, Precision, Recall, F1, ROC AUC)
    - Comparison table construction
    - Best-model selection logic
    - Diagnostic plot generation (confusion matrix, ROC, PR curve,
      feature importance, learning curve)
    - Text/CSV report generation

Author: ML Engineering Team
"""

import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import learning_curve, StratifiedKFold

from utils import get_logger, REPORTS_DIR, RANDOM_STATE

logger = get_logger(__name__)

sns.set_style("whitegrid")


# ------------------------------------------------------------------
# METRIC COMPUTATION
# ------------------------------------------------------------------
def evaluate_single_model(
    model, X_test, y_test
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """
    Compute the full metric suite for a single fitted model.

    Parameters
    ----------
    model : fitted estimator
    X_test : array-like
    y_test : array-like

    Returns
    -------
    metrics : Dict[str, float]
        Accuracy, Precision, Recall, F1 Score, ROC AUC.
    y_pred : np.ndarray
        Hard predictions.
    y_proba : np.ndarray
        Predicted probability of the positive class.
    prediction_time : float
        Time (seconds) taken to run predictions on X_test.
    """
    try:
        start = time.time()
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba (rare in this zoo)
            y_proba = model.decision_function(X_test)

        prediction_time = time.time() - start

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }

        return metrics, y_pred, y_proba, prediction_time

    except Exception as e:
        logger.exception(f"Failed to evaluate model: {e}")
        raise


def evaluate_all_models(
    fitted_models: Dict[str, object],
    X_test,
    y_test,
    training_times: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, float]]:
    """
    Evaluate every fitted model and build a comparison table.

    Parameters
    ----------
    fitted_models : Dict[str, object]
        Mapping of model name -> fitted estimator.
    X_test : array-like
    y_test : array-like
    training_times : Dict[str, float]
        Mapping of model name -> training time in seconds.

    Returns
    -------
    comparison_df : pd.DataFrame
        Sorted comparison table (by F1 Score desc, then ROC AUC desc).
    predictions_cache : Dict[str, dict]
        Mapping of model name -> {'y_pred':.., 'y_proba':..} for reuse
        in plotting without re-predicting.
    prediction_times : Dict[str, float]
        Mapping of model name -> prediction time in seconds.
    """
    rows = []
    predictions_cache = {}
    prediction_times = {}

    for name, model in fitted_models.items():
        try:
            metrics, y_pred, y_proba, pred_time = evaluate_single_model(
                model, X_test, y_test
            )
            metrics["Model"] = name
            metrics["Training Time"] = training_times.get(name, np.nan)
            metrics["Prediction Time"] = pred_time

            rows.append(metrics)
            predictions_cache[name] = {"y_pred": y_pred, "y_proba": y_proba}
            prediction_times[name] = pred_time

            logger.info(
                f"{name} -> Acc: {metrics['Accuracy']:.4f}, "
                f"F1: {metrics['F1 Score']:.4f}, "
                f"ROC AUC: {metrics['ROC AUC']:.4f}"
            )

        except Exception as e:
            logger.exception(f"Skipping {name} due to evaluation error: {e}")
            continue

    comparison_df = pd.DataFrame(rows)

    if not comparison_df.empty:
        column_order = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
            "Training Time",
            "Prediction Time",
        ]
        comparison_df = comparison_df[column_order]
        comparison_df = comparison_df.sort_values(
            by=["F1 Score", "ROC AUC"], ascending=False
        ).reset_index(drop=True)

    return comparison_df, predictions_cache, prediction_times


# ------------------------------------------------------------------
# BEST MODEL SELECTION
# ------------------------------------------------------------------
def select_best_model(comparison_df: pd.DataFrame) -> str:
    """
    Select the best model name based on the highest F1 Score, with
    ROC AUC as the tie-breaker.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Must contain 'Model', 'F1 Score', and 'ROC AUC' columns.

    Returns
    -------
    str
        Name of the best model.
    """
    try:
        if comparison_df.empty:
            raise ValueError("Comparison table is empty; cannot select a best model.")

        sorted_df = comparison_df.sort_values(
            by=["F1 Score", "ROC AUC"], ascending=False
        )
        best_name = sorted_df.iloc[0]["Model"]
        return best_name

    except Exception as e:
        logger.exception(f"Failed to select best model: {e}")
        raise


# ------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    """Save a confusion matrix heatmap for the given model."""
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {path}")
    except Exception as e:
        logger.exception(f"Failed to plot confusion matrix: {e}")


def plot_roc_curve(y_test, y_proba, model_name: str) -> None:
    """Save an ROC curve plot for the given model."""
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "roc_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved ROC curve plot to {path}")
    except Exception as e:
        logger.exception(f"Failed to plot ROC curve: {e}")


def plot_precision_recall_curve(y_test, y_proba, model_name: str) -> None:
    """Save a Precision-Recall curve plot for the given model."""
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=model_name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "precision_recall_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved precision-recall curve plot to {path}")
    except Exception as e:
        logger.exception(f"Failed to plot precision-recall curve: {e}")


def plot_feature_importance(
    model, feature_names, model_name: str, top_n: int = 20
) -> None:
    """
    Save a feature importance bar chart if the model exposes
    feature_importances_ or coef_. Silently skips otherwise.
    """
    try:
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()

        if importances is None:
            logger.warning(
                f"{model_name} does not expose feature importances; "
                f"skipping feature importance plot."
            )
            return

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False).head(top_n)

        plt.figure(figsize=(8, max(5, top_n * 0.3)))
        sns.barplot(
            data=importance_df, x="Importance", y="Feature", color="steelblue"
        )
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved feature importance plot to {path}")

    except Exception as e:
        logger.exception(f"Failed to plot feature importance: {e}")


def plot_learning_curve(model, X_train, y_train, model_name: str) -> None:
    """Save a learning curve plot showing train/CV score vs. training size."""
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(6, 5))
        plt.plot(train_sizes, train_mean, "o-", label="Training F1")
        plt.plot(train_sizes, val_mean, "o-", label="Validation F1")
        plt.xlabel("Training Set Size")
        plt.ylabel("F1 Score")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend(loc="best")
        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, "learning_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved learning curve plot to {path}")

    except Exception as e:
        logger.exception(f"Failed to plot learning curve: {e}")


def generate_all_plots(
    model,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
) -> None:
    """
    Generate the full plot suite (confusion matrix, ROC, PR curve,
    feature importance, learning curve) for the given model and save
    them all into the reports/ directory.
    """
    try:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(y_test, y_proba, model_name)
        plot_precision_recall_curve(y_test, y_proba, model_name)
        plot_feature_importance(model, feature_names, model_name)
        plot_learning_curve(model, X_train, y_train, model_name)

    except Exception as e:
        logger.exception(f"Failed while generating plot suite: {e}")


# ------------------------------------------------------------------
# REPORT GENERATION
# ------------------------------------------------------------------
def save_reports(
    comparison_df: pd.DataFrame,
    best_model,
    best_model_name: str,
    X_test,
    y_test,
    best_metrics: pd.Series,
) -> None:
    """
    Save the model comparison CSV, classification report txt, and
    best model metrics txt into reports/.
    """
    try:
        # 1) model_comparison.csv
        comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved model comparison table to {comparison_path}")

        # 2) classification_report.txt
        y_pred = best_model.predict(X_test)
        report_text = classification_report(
            y_test, y_pred, target_names=["Fail", "Pass"]
        )
        classification_report_path = os.path.join(
            REPORTS_DIR, "classification_report.txt"
        )
        with open(classification_report_path, "w") as f:
            f.write(f"Classification Report - {best_model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(report_text)
        logger.info(f"Saved classification report to {classification_report_path}")

        # 3) best_model_metrics.txt
        best_metrics_path = os.path.join(REPORTS_DIR, "best_model_metrics.txt")
        with open(best_metrics_path, "w") as f:
            f.write("BEST MODEL METRICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model          : {best_model_name}\n")
            f.write(f"Accuracy       : {best_metrics['Accuracy']:.4f}\n")
            f.write(f"Precision      : {best_metrics['Precision']:.4f}\n")
            f.write(f"Recall         : {best_metrics['Recall']:.4f}\n")
            f.write(f"F1 Score       : {best_metrics['F1 Score']:.4f}\n")
            f.write(f"ROC AUC        : {best_metrics['ROC AUC']:.4f}\n")
            f.write(f"Training Time  : {best_metrics['Training Time']:.4f}s\n")
            f.write(f"Prediction Time: {best_metrics['Prediction Time']:.4f}s\n")
        logger.info(f"Saved best model metrics to {best_metrics_path}")

    except Exception as e:
        logger.exception(f"Failed to save reports: {e}")
        raise
