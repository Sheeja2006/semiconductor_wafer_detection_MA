"""
train_models.py
================
Defines all candidate models and trains them on the wafer dataset.
Also serves as the MAIN PIPELINE ENTRY POINT, orchestrating the full
Steps 1-11 workflow:

    1. Load dataset
    2. Train/test split (80/20, stratified)
    3. Train 7 candidate models
    4. Evaluate every model
    5. Build comparison table
    6. Select best model (highest F1, tie-break ROC AUC)
    7. Hyperparameter tune the best model (RandomizedSearchCV, 5-fold CV)
    8. Save best_model.pkl, scaler.pkl, label_encoder.pkl
    9. Generate diagnostic plots
    10. Save reports (csv / txt)
    11. Print final summary

Run with:
    python training/train_models.py

Author: ML Engineering Team
"""

import time
from typing import Dict, Tuple

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils import (
    RANDOM_STATE,
    get_logger,
    load_dataset,
    get_feature_target_split,
    split_train_test,
    ensure_dirs,
)
from evaluate_models import (
    evaluate_all_models,
    select_best_model,
    generate_all_plots,
    save_reports,
)
from hyperparameter_tuning import tune_best_model
from save_model import save_artifacts

logger = get_logger(__name__)


# ------------------------------------------------------------------
# MODEL DEFINITIONS
# ------------------------------------------------------------------
def get_model_zoo() -> Dict[str, object]:
    """
    Build a dictionary of all candidate models with sensible default
    hyperparameters. Every model uses a fixed random_state for
    reproducibility.

    Returns
    -------
    Dict[str, object]
        Mapping of model name -> unfit estimator instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            random_state=RANDOM_STATE, verbose=0
        ),
    }
    return models


# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
def train_all_models(
    models: Dict[str, object], X_train, y_train
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Fit every model in the model zoo on the training data and record
    training time for each.

    Parameters
    ----------
    models : Dict[str, object]
        Mapping of model name -> unfit estimator.
    X_train : array-like
        Training features (scaled).
    y_train : array-like
        Training target.

    Returns
    -------
    fitted_models : Dict[str, object]
        Mapping of model name -> fitted estimator.
    training_times : Dict[str, float]
        Mapping of model name -> training time in seconds.
    """
    fitted_models = {}
    training_times = {}

    for name, model in models.items():
        try:
            logger.info(f"Training model: {name} ...")
            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            fitted_models[name] = model
            training_times[name] = elapsed

            logger.info(f"Finished training {name} in {elapsed:.3f}s")

        except Exception as e:
            logger.exception(f"Failed to train model {name}: {e}")
            # Continue training remaining models even if one fails
            continue

    return fitted_models, training_times


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def run_pipeline() -> None:
    """
    Execute the complete end-to-end training pipeline (Steps 1-11).
    """
    try:
        ensure_dirs()

        # STEP 1: Load dataset
        logger.info("STEP 1: Loading dataset")
        df = load_dataset()

        # STEP 2: Feature/target split + 80/20 stratified split
        logger.info("STEP 2: Splitting features/target and train/test")
        X, y = get_feature_target_split(df)

        # Label encode target (kept for pipeline consistency / Flask reuse
        # even though Final_Result is already binary 0/1).
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)

        # Feature scaling (fit ONLY on training data to avoid leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # STEP 3: Train all 7 candidate models
        logger.info("STEP 3: Training all candidate models")
        models = get_model_zoo()
        fitted_models, training_times = train_all_models(
            models, X_train_scaled, y_train
        )

        # STEP 4 & 5: Evaluate every model + build comparison table
        logger.info("STEP 4-5: Evaluating models and building comparison table")
        comparison_df, predictions_cache, prediction_times = evaluate_all_models(
            fitted_models, X_test_scaled, y_test, training_times
        )

        # STEP 6: Select best model (highest F1, tie-break ROC AUC)
        logger.info("STEP 6: Selecting best model")
        best_model_name = select_best_model(comparison_df)
        best_model = fitted_models[best_model_name]
        logger.info(f"Best model selected: {best_model_name}")

        # STEP 7: Hyperparameter tuning for the best model only
        logger.info("STEP 7: Hyperparameter tuning best model")
        tuned_model, tuning_time = tune_best_model(
            best_model_name, X_train_scaled, y_train
        )

        # Re-evaluate the tuned model on the test set
        logger.info("Re-evaluating tuned best model on test set")
        tuned_comparison_df, tuned_predictions_cache, tuned_pred_times = (
            evaluate_all_models(
                {best_model_name + " (Tuned)": tuned_model},
                X_test_scaled,
                y_test,
                {best_model_name + " (Tuned)": tuning_time},
            )
        )

        final_best_name = best_model_name + " (Tuned)"
        final_best_metrics = tuned_comparison_df.iloc[0]

        # STEP 8: Save best_model.pkl, scaler.pkl, label_encoder.pkl
        logger.info("STEP 8: Saving model artifacts")
        save_artifacts(tuned_model, scaler, label_encoder)

        # STEP 9: Generate diagnostic plots for the tuned best model
        logger.info("STEP 9: Generating diagnostic plots")
        generate_all_plots(
            model=tuned_model,
            model_name=final_best_name,
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            feature_names=X.columns.tolist(),
        )

        # STEP 10: Save reports
        logger.info("STEP 10: Saving reports")
        full_comparison_df = comparison_df  # all 7 baseline models
        save_reports(
            comparison_df=full_comparison_df,
            best_model=tuned_model,
            best_model_name=final_best_name,
            X_test=X_test_scaled,
            y_test=y_test,
            best_metrics=final_best_metrics,
        )

        # STEP 11: Print final summary
        print("\n" + "=" * 50)
        print("BEST MODEL SUMMARY")
        print("=" * 50)
        print(f"Model         : {final_best_name}")
        print(f"Accuracy      : {final_best_metrics['Accuracy']:.4f}")
        print(f"Precision     : {final_best_metrics['Precision']:.4f}")
        print(f"Recall        : {final_best_metrics['Recall']:.4f}")
        print(f"F1 Score      : {final_best_metrics['F1 Score']:.4f}")
        print(f"ROC AUC       : {final_best_metrics['ROC AUC']:.4f}")
        print(f"Training Time : {final_best_metrics['Training Time']:.4f}s")
        print(f"Prediction Time: {final_best_metrics['Prediction Time']:.4f}s")
        print("=" * 50 + "\n")

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
