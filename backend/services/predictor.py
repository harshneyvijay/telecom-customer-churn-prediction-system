"""
Loads trained artifacts (model, scaler, feature column order) once at
startup and exposes prediction functions used by the API routes.

Uses the SAME preprocessing module used at training time
(backend/preprocessing/preprocess.py) to avoid any train/serve skew.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd

from preprocessing.preprocess import build_feature_frame

MODELS_DIR = os.environ.get(
    "MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "logistic_regression")
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.64"))

_scaler = None
_models = {}
_feature_columns = None
_feature_importance = None


class ArtifactsNotFoundError(Exception):
    pass


def _artifact_path(name):
    return os.path.join(MODELS_DIR, name)


def load_artifacts():
    """Loads scaler, models, and feature column order into memory.
    Safe to call multiple times (idempotent)."""
    global _scaler, _models, _feature_columns, _feature_importance

    feature_columns_path = _artifact_path("feature_columns.json")
    scaler_path = _artifact_path("scaler.pkl")

    if not os.path.exists(feature_columns_path) or not os.path.exists(scaler_path):
        # Artifacts not trained yet. App still boots so the frontend/metrics
        # pages work, but /predict will raise a clear error until the user
        # runs model_training/train_model.py
        _scaler = None
        _models = {}
        _feature_columns = None
        _feature_importance = None
        return

    with open(feature_columns_path) as f:
        _feature_columns = json.load(f)

    _scaler = joblib.load(scaler_path)

    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model_path = _artifact_path(f"{name}.pkl")
        if os.path.exists(model_path):
            _models[name] = joblib.load(model_path)

    importance_path = _artifact_path("feature_importance.json")
    if os.path.exists(importance_path):
        with open(importance_path) as f:
            _feature_importance = json.load(f)


def artifacts_ready():
    return _scaler is not None and _feature_columns is not None and len(_models) > 0


def get_feature_importance(top_n=10):
    if not _feature_importance:
        return []
    items = list(_feature_importance.items())[:top_n]
    return [{"feature": k, "importance": round(v, 4)} for k, v in items]


def get_feature_columns():
    return _feature_columns or []


def predict_single(customer: dict, model_name: str = None, threshold: float = None):
    if not artifacts_ready():
        raise ArtifactsNotFoundError(
            "Model artifacts not found. Run model_training/train_model.py first."
        )

    model_name = model_name or DEFAULT_MODEL
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

    if model_name not in _models:
        raise ValueError(f"Unknown model '{model_name}'")

    df = pd.DataFrame([customer])
    features = build_feature_frame(df, _feature_columns)
    scaled = _scaler.transform(features)

    model = _models[model_name]
    proba = float(model.predict_proba(scaled)[0, 1])
    prediction = int(proba >= threshold)

    return _format_result(proba, prediction, model_name, threshold, features.iloc[0])


def predict_batch(df: pd.DataFrame, model_name: str = None, threshold: float = None):
    if not artifacts_ready():
        raise ArtifactsNotFoundError(
            "Model artifacts not found. Run model_training/train_model.py first."
        )

    model_name = model_name or DEFAULT_MODEL
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    model = _models[model_name]

    features = build_feature_frame(df, _feature_columns)
    scaled = _scaler.transform(features)
    probas = model.predict_proba(scaled)[:, 1]
    predictions = (probas >= threshold).astype(int)

    out = df.copy()
    out["churn_probability"] = np.round(probas, 4)
    out["prediction"] = predictions
    out["risk_level"] = [_risk_level(p) for p in probas]
    return out


def _risk_level(proba):
    if proba >= 0.7:
        return "High"
    elif proba >= 0.4:
        return "Medium"
    return "Low"


def _format_result(proba, prediction, model_name, threshold, feature_row):
    return {
        "churn_probability": round(proba, 4),
        "prediction": prediction,
        "risk_level": _risk_level(proba),
        "model_used": model_name,
        "threshold_used": threshold,
        "predicted_outcome": "Likely to churn" if prediction == 1 else "Likely to stay",
        "_feature_row": feature_row,  # used internally by explainer, stripped before response
    }
