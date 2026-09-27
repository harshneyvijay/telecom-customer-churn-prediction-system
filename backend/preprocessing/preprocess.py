"""
This module is imported by:
  - model_training/train_model.py   (fitting the pipeline)
  - backend/services/predictor.py   (serving predictions)
"""

import pandas as pd
import numpy as np

# Columns in the raw IBM Telco Customer Churn CSV that we drop before modeling.
DROP_COLUMNS = ["customerID"]

TARGET_COLUMN = "Churn"

# Binary Yes/No style columns (mapped to 1/0)
BINARY_COLUMNS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

# Columns with more than 2 categories -> one-hot encoded
MULTI_CATEGORY_COLUMNS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# The final, ordered list of feature columns the model expects.
# Generated once during training and stored in feature_columns.json,
# then reused at inference time to guarantee identical column order.


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the base cleaning steps used in the original notebook:
      - TotalCharges -> numeric, coerce errors
      - fill missing TotalCharges (new customers, tenure=0) with 0
      - drop customerID
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "SeniorCitizen" in df.columns:
        # dataset stores this as 0/1 already, ensure int
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features:
      - binary Yes/No columns -> 1/0
      - multi-category columns -> one-hot encoded
    Does NOT touch the target column if present.
    """
    df = df.copy()

    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    present_multi = [c for c in MULTI_CATEGORY_COLUMNS if c in df.columns]
    if present_multi:
        df = pd.get_dummies(df, columns=present_multi, drop_first=True)

    return df


def build_feature_frame(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Cleans + encodes a raw dataframe, then aligns it to the exact
    feature_columns used at training time (adds missing dummy columns as 0,
    drops anything unexpected, and fixes column order).
    """
    df = clean_raw_dataframe(df)
    df = encode_features(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df


def encode_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
