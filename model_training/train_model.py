import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from preprocessing.preprocess import clean_raw_dataframe, encode_features, encode_target, TARGET_COLUMN

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "telco_churn.csv")
MODELS_DIR = os.path.join(ROOT, "models")

RANDOM_STATE = 42


def find_best_threshold(y_true, probas):
    """Scan thresholds 0.01-0.99 and return the one maximizing F1."""
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (probas >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


def evaluate(y_true, probas, threshold):
    preds = (probas >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probas), 4),
        "threshold": round(float(threshold), 2),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download the IBM Telco Customer Churn dataset and save it as data/telco_churn.csv"
        )

    raw = pd.read_csv(DATA_PATH)
    df = clean_raw_dataframe(raw)

    y = encode_target(df)
    X = encode_features(df.drop(columns=[TARGET_COLUMN]))

    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    results = {}
    artifacts = {}

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_res, y_train_res)
    lr_probas = lr.predict_proba(X_test_scaled)[:, 1]
    lr_threshold = find_best_threshold(y_test, lr_probas)
    results["logistic_regression"] = evaluate(y_test, lr_probas, lr_threshold)
    artifacts["logistic_regression"] = lr

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    )
    rf.fit(X_train_res, y_train_res)
    rf_probas = rf.predict_proba(X_test_scaled)[:, 1]
    rf_threshold = find_best_threshold(y_test, rf_probas)
    results["random_forest"] = evaluate(y_test, rf_probas, rf_threshold)
    artifacts["random_forest"] = rf

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300, use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train_res, y_train_res)
    xgb_probas = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_threshold = find_best_threshold(y_test, xgb_probas)
    results["xgboost"] = evaluate(y_test, xgb_probas, xgb_threshold)
    artifacts["xgboost"] = xgb

    # Save artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    for name, model in artifacts.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Random Forest feature importance (used for global explainability)
    importances = dict(zip(feature_columns, rf.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)

    # Store churn probability distribution (test set, RF) for the insights chart
    prob_dist = {
        "churned": rf_probas[y_test.values == 1].tolist(),
        "not_churned": rf_probas[y_test.values == 0].tolist(),
    }
    with open(os.path.join(MODELS_DIR, "probability_distribution.json"), "w") as f:
        json.dump(prob_dist, f)

    # Class distribution
    class_dist = {"churned": int((y == 1).sum()), "not_churned": int((y == 0).sum())}
    with open(os.path.join(MODELS_DIR, "class_distribution.json"), "w") as f:
        json.dump(class_dist, f)

    print("Training complete. Metrics:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
