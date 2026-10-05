"""
hyperparameter_tuning.py
=========================
Performs hyperparameter tuning for the single best model selected
during evaluation, using RandomizedSearchCV with 5-fold cross
validation. Only the best model is tuned (per project spec) to keep
runtime reasonable.

Author: ML Engineering Team
"""

import time
from typing import Tuple

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils import RANDOM_STATE, get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# SEARCH SPACES
# ------------------------------------------------------------------
def get_param_distributions(model_name: str) -> dict:
    """
    Return a reasonable RandomizedSearchCV hyperparameter distribution
    for the given model name.

    Parameters
    ----------
    model_name : str
        Name of the model (must match keys used in get_model_zoo()).

    Returns
    -------
    dict
        Hyperparameter distribution compatible with RandomizedSearchCV.
    """
    param_grids = {
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "Support Vector Machine": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [-1, 5, 10, 15],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63, 127],
            "subsample": [0.6, 0.8, 1.0],
        },
        "CatBoost": {
            "iterations": [200, 400, 600],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7],
        },
    }

    if model_name not in param_grids:
        raise ValueError(f"No parameter grid defined for model: {model_name}")

    return param_grids[model_name]


def get_fresh_estimator(model_name: str):
    """
    Return a brand-new, unfit estimator instance corresponding to
    model_name, used as the base_estimator for RandomizedSearchCV.
    """
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
    }

    if model_name not in estimators:
        raise ValueError(f"No estimator defined for model: {model_name}")

    return estimators[model_name]


# ------------------------------------------------------------------
# TUNING
# ------------------------------------------------------------------
def tune_best_model(
    model_name: str, X_train, y_train, n_iter: int = 20
) -> Tuple[object, float]:
    """
    Run RandomizedSearchCV (5-fold stratified CV) for the given model
    and return the best fitted estimator.

    Parameters
    ----------
    model_name : str
        Name of the best-performing baseline model.
    X_train : array-like
        Scaled training features.
    y_train : array-like
        Training target.
    n_iter : int
        Number of parameter settings sampled by RandomizedSearchCV.

    Returns
    -------
    best_estimator : fitted estimator
        The tuned model refit on the full training set.
    tuning_time : float
        Total time (seconds) taken for the search.
    """
    try:
        logger.info(f"Starting hyperparameter tuning for: {model_name}")

        base_estimator = get_fresh_estimator(model_name)
        param_distributions = get_param_distributions(model_name)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        start = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        logger.info(
            f"Tuning complete for {model_name}. Best CV F1: "
            f"{search.best_score_:.4f}. Best params: {search.best_params_}"
        )
        logger.info(f"Tuning took {elapsed:.2f}s")

        return search.best_estimator_, elapsed

    except Exception as e:
        logger.exception(f"Hyperparameter tuning failed for {model_name}: {e}")
        raise
