"""
utils.py
========
Shared utilities for the Wafer Defect Prediction pipeline:
    - Logging configuration
    - Global constants / paths
    - Data loading and train/test splitting
    - Small helper functions reused across the training package

Author: ML Engineering Team
"""

import os
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# GLOBAL CONSTANTS
# ------------------------------------------------------------------
RANDOM_STATE = 42

# Project root = parent of the "training" folder (this file's location)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

TARGET_COLUMN = "Final_Result"

# Columns to drop if present (per project spec). Only columns that
# actually exist in the dataset are dropped, so this is safe even if
# some of them are absent from a given dataset version.
COLUMNS_TO_DROP = [
    "Wafer_ID",
    "Final_Result",
    "Root_Cause_Stage",
    "Freeze_Wafer",
    "Rework_Required",
]

TEST_SIZE = 0.20

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Create (or fetch) a configured logger instance.

    Parameters
    ----------
    name : str
        Name of the logger, typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger that prints to console with timestamps.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on re-import
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = get_logger(__name__)


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the processed wafer dataset from disk.

    Parameters
    ----------
    path : str
        Path to the processed_dataset.csv file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the given path.
    ValueError
        If the target column is missing from the dataset.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in dataset columns."
            )

        return df

    except Exception as e:
        logger.exception(f"Failed to load dataset: {e}")
        raise


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into feature matrix X and target vector y,
    dropping identifier / leakage columns as defined in COLUMNS_TO_DROP.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (Final_Result).
    """
    try:
        y = df[TARGET_COLUMN].copy()

        drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
        X = df.drop(columns=drop_cols)

        # Keep only numeric columns as features (dataset is pre-processed,
        # but this guards against stray non-numeric columns).
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning(
                f"Dropping non-numeric feature columns not handled by "
                f"preprocessing: {non_numeric}"
            )
            X = X.drop(columns=non_numeric)

        logger.info(
            f"Feature/target split complete. X shape: {X.shape}, "
            f"y shape: {y.shape}"
        )
        return X, y

    except Exception as e:
        logger.exception(f"Failed during feature/target split: {e}")
        raise


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform an 80/20 stratified train-test split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        logger.info(
            f"Stratified split complete. Train: {X_train.shape}, "
            f"Test: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception(f"Failed during train/test split: {e}")
        raise


def ensure_dirs() -> None:
    """Create models/ and reports/ directories if they do not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
