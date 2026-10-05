# Explainable AI-Based Early Wafer Defect Prediction

End-to-end ML pipeline that predicts whether a semiconductor wafer will
**PASS** or **FAIL** before the final manufacturing stage, using 67
process features spanning all fabrication stages.

## Project Structure

```
Wafer_Defect_Prediction/
├── dataset/
│   └── processed_dataset.csv      # 7,921 rows × 68 columns (input)
├── models/                        # generated after running the pipeline
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── reports/                       # generated after running the pipeline
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── best_model_metrics.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── learning_curve.png
├── training/
│   ├── utils.py                   # config, logging, data loading/splitting
│   ├── train_models.py            # model zoo + MAIN PIPELINE ENTRY POINT
│   ├── evaluate_models.py         # metrics, comparison table, plots, reports
│   ├── hyperparameter_tuning.py   # RandomizedSearchCV for the best model
│   └── save_model.py              # joblib persistence of artifacts
├── app.py                         # Flask inference API
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost joblib flask
```

### 2. Run the full training pipeline

```bash
cd training
python train_models.py
```

This executes all 11 steps:

1. Load `dataset/processed_dataset.csv`
2. Stratified 80/20 train/test split
3. Train 7 models: Logistic Regression, Decision Tree, Random Forest,
   SVM, XGBoost, LightGBM, CatBoost
4. Evaluate every model (Accuracy, Precision, Recall, F1, ROC AUC,
   confusion matrix, classification report)
5. Build a model comparison table
6. Auto-select the best model (highest F1, tie-break ROC AUC)
7. Tune the best model with `RandomizedSearchCV` (5-fold CV)
8. Save `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl` to `models/`
9. Generate diagnostic plots into `reports/`
10. Save `model_comparison.csv`, `classification_report.txt`,
    `best_model_metrics.txt` into `reports/`
11. Print a final best-model summary to the console

### 3. Serve predictions via Flask

```bash
python app.py
```

Then:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [<67 feature values in the same column order as X_train>]}'
```

Response:

```json
{"prediction": "Pass", "probability": 0.66}
```

## Design Notes

- **Feature set**: all 67 numeric process columns except
  `Final_Result` (target). `Wafer_ID`, `Root_Cause_Stage`,
  `Freeze_Wafer`, and `Rework_Required` are dropped if present (none
  of them exist in the current `processed_dataset.csv`, so all
  remaining columns are used as features).
- **Scaling**: a single `StandardScaler`, fit only on the training
  split, is applied to all models (including tree-based ones) so the
  Flask app only needs to manage one preprocessing artifact.
- **Label encoding**: `Final_Result` is already binary (0 = Fail,
  1 = Pass), but a `LabelEncoder` is still fit and saved for pipeline
  consistency, per the project spec.
- **Reproducibility**: `random_state=42` is used everywhere (splits,
  models, cross-validation, RandomizedSearchCV).
- **Best model selection**: automatic, based on highest F1 Score on
  the held-out test set, with ROC AUC as a tie-breaker.
- **Hyperparameter tuning**: only the single best baseline model is
  tuned (RandomizedSearchCV, 5-fold `StratifiedKFold`, `n_iter=20`,
  scoring=`f1`) to keep runtime reasonable — the other 6 models are
  used only for comparison, per the project spec.
