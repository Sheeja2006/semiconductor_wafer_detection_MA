# ==========================================================
# Explainable AI Based Early Wafer Defect Prediction
# Data Cleaning + Preprocessing Pipeline
# ==========================================================


import pandas as pd
import numpy as np
import os
import joblib


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split



# ==========================================================
# PROJECT PATH CONFIGURATION
# ==========================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



INPUT_PATH = os.path.join(
    DATASET_DIR,
    "semiconductor_dataset.csv"
)


OUTPUT_PATH = os.path.join(
    DATASET_DIR,
    "processed_dataset.csv"
)



# ==========================================================
# LOAD DATASET
# ==========================================================


print("="*70)
print("LOADING DATASET")
print("="*70)



df = pd.read_csv(INPUT_PATH)



print("\nOriginal Dataset Shape:")
print(df.shape)




# ==========================================================
# STEP 1
# REMOVE DUPLICATES
# ==========================================================


print("\nRemoving Duplicate Records")



duplicate_count = df.duplicated().sum()



df = df.drop_duplicates()



print(
    "Duplicates Removed:",
    duplicate_count
)



# ==========================================================
# STEP 2
# CLEAN CATEGORICAL VALUES
# ==========================================================


print("\nCleaning Categorical Columns")



categorical_columns = [

    "Final_Result",

    "Freeze_Wafer",

    "Rework_Required",

    "Root_Cause_Stage"

]



for col in categorical_columns:


    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )



# Final Result cleaning

df["Final_Result"] = df["Final_Result"].replace(
    {
        "pass":"Pass",
        "fail":"Fail"
    }
)



# Freeze cleaning

df["Freeze_Wafer"] = df["Freeze_Wafer"].replace(
    {
        "yes":"Yes",
        "no":"No"
    }
)



# Rework cleaning

df["Rework_Required"] = df["Rework_Required"].replace(
    {
        "yes":"Yes",
        "no":"No"
    }
)



# Root cause formatting

df["Root_Cause_Stage"] = (
    df["Root_Cause_Stage"]
    .str.title()
)



print(
    "Categorical Cleaning Completed"
)




# ==========================================================
# STEP 3
# MISSING VALUE HANDLING
# ==========================================================


print("\nHandling Missing Values")



numeric_columns = df.select_dtypes(
    include=np.number
).columns



imputer = SimpleImputer(
    strategy="median"
)



df[numeric_columns] = imputer.fit_transform(
    df[numeric_columns]
)



joblib.dump(
    imputer,
    os.path.join(
        MODEL_DIR,
        "missing_value_imputer.pkl"
    )
)



print(
    "Missing Values Filled"
)




# ==========================================================
# STEP 4
# OUTLIER HANDLING
# ==========================================================


print("\nOutlier Detection and Treatment")



for col in numeric_columns:


    Q1 = df[col].quantile(0.25)


    Q3 = df[col].quantile(0.75)


    IQR = Q3-Q1



    lower_limit = Q1 - (1.5*IQR)


    upper_limit = Q3 + (1.5*IQR)



    df[col] = np.clip(
        df[col],
        lower_limit,
        upper_limit
    )



print(
    "Outlier Treatment Completed"
)




# ==========================================================
# STEP 5
# FEATURE ENGINEERING
# ==========================================================


print("\nFeature Engineering")



# Combined particle contamination


df["Total_Particle_Count"] = (

    df["Particle_Count_Clean"]

    +

    df["Particle_Count_Final"]

)



# Manufacturing error index


df["Process_Error_Index"] = (

    df["Alignment_Error_um"]

    +

    df["Critical_Dimension_Error_nm"]

    +

    df["Surface_Roughness_Final_nm"]

)



# Temperature stress


df["Temperature_Stress"] = (

    abs(

        df["Furnace_Temperature_Growth_C"]

        -

        df["Furnace_Temperature_Oxidation_C"]

    )

)



print(
    "Feature Engineering Completed"
)




# ==========================================================
# STEP 6
# ENCODE TARGET VARIABLE
# ==========================================================


print("\nEncoding Target")



encoder = LabelEncoder()



df["Final_Result"] = encoder.fit_transform(
    df["Final_Result"]
)



# Fail = 0
# Pass = 1



joblib.dump(
    encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)



# ==========================================================
# STEP 7
# FEATURE TARGET SPLIT
# ==========================================================


X = df.drop(
    [

        "Final_Result",

        "Wafer_ID",

        "Root_Cause_Stage",

        "Freeze_Wafer",

        "Rework_Required"

    ],
    axis=1
)



y = df["Final_Result"]




print("\nFeature Count:")
print(X.shape[1])




# ==========================================================
# STEP 8
# FEATURE SCALING
# ==========================================================


print("\nFeature Scaling")



scaler = StandardScaler()



X_scaled = scaler.fit_transform(
    X
)



joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)



processed_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)



processed_df["Final_Result"] = y.values





# ==========================================================
# SAVE PROCESSED DATASET
# ==========================================================


processed_df.to_csv(
    OUTPUT_PATH,
    index=False
)



# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================


X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)




joblib.dump(
    X_train,
    os.path.join(
        MODEL_DIR,
        "X_train.pkl"
    )
)



joblib.dump(
    X_test,
    os.path.join(
        MODEL_DIR,
        "X_test.pkl"
    )
)



joblib.dump(
    y_train,
    os.path.join(
        MODEL_DIR,
        "y_train.pkl"
    )
)



joblib.dump(
    y_test,
    os.path.join(
        MODEL_DIR,
        "y_test.pkl"
    )
)




# ==========================================================
# FINAL REPORT
# ==========================================================


print("\n")
print("="*70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*70)



print(
    "\nFinal Dataset Shape:",
    processed_df.shape
)



print(
    "Training Data:",
    X_train.shape
)



print(
    "Testing Data:",
    X_test.shape
)



print("\nGenerated Files:")

print(
    "Processed Dataset:"
)

print(
    OUTPUT_PATH
)


print(
    "\nModel Files Saved:"
)

print(
    MODEL_DIR
)


print("\nAll preprocessing steps completed.")
