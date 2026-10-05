# ==========================================================
# DATA CLEANING - STEP 1 (DATA UNDERSTANDING)
# Project:
# Explainable AI-Based Early Wafer Defect Prediction
# ==========================================================

import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("dataset/semiconductor_dataset.csv")

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

# ----------------------------------------------------------
# Display First 5 Rows
# ----------------------------------------------------------

print("\nFirst 5 Rows")
print(df.head())

# ----------------------------------------------------------
# Dataset Shape
# ----------------------------------------------------------

print("\nDataset Shape")
print(df.shape)

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

print("\nDataset Information")
print(df.info())

# ----------------------------------------------------------
# Column Names
# ----------------------------------------------------------

print("\nColumn Names")

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

# ----------------------------------------------------------
# Data Types
# ----------------------------------------------------------

print("\nData Types")

print(df.dtypes)

# ----------------------------------------------------------
# Statistical Summary
# ----------------------------------------------------------

print("\nStatistical Summary")

print(df.describe())

# ----------------------------------------------------------
# Missing Values Analysis
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing Values": missing.values,
    "Missing Percentage": np.round((missing.values / len(df)) * 100, 2)
})

missing_df = missing_df[missing_df["Missing Values"] > 0]

print(missing_df)

# Save report
missing_df.to_csv("reports/missing_values_report.csv", index=False)

# ----------------------------------------------------------
# Duplicate Records
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Total Duplicate Rows :", duplicate_count)

duplicates = df[df.duplicated()]

print("\nFirst Five Duplicate Rows")

print(duplicates.head())

duplicates.to_csv("reports/duplicate_records.csv", index=False)

# ----------------------------------------------------------
# Unique Values in Categorical Columns
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL VALUE ANALYSIS")
print("=" * 70)

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:

    print("\n------------------------------------------")

    print("Column :", column)

    print("------------------------------------------")

    print(df[column].unique())

# ----------------------------------------------------------
# Missing Value Percentage Summary
# ----------------------------------------------------------

print("\n" + "=" * 70)

print("MISSING VALUE SUMMARY")

print("=" * 70)

print(df.isnull().mean() * 100)

# ----------------------------------------------------------
# Duplicate Wafer IDs
# ----------------------------------------------------------

print("\n" + "=" * 70)

print("DUPLICATE WAFER IDs")

print("=" * 70)

duplicate_ids = df["Wafer_ID"].duplicated().sum()

print("Duplicate Wafer IDs :", duplicate_ids)

# ----------------------------------------------------------
# Dataset Memory Usage
# ----------------------------------------------------------

print("\nMemory Usage")

memory = df.memory_usage(deep=True).sum() / (1024**2)

print(f"{memory:.2f} MB")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------

print("\n" + "=" * 70)

print("DATASET SUMMARY")

print("=" * 70)

print(f"Rows                 : {df.shape[0]}")
print(f"Columns              : {df.shape[1]}")
print(f"Numerical Columns    : {len(df.select_dtypes(include=np.number).columns)}")
print(f"Categorical Columns  : {len(df.select_dtypes(include='object').columns)}")
print(f"Duplicate Rows       : {duplicate_count}")
print(f"Columns with Missing : {missing_df.shape[0]}")

print("\nData Understanding Completed Successfully.")