"""
Explainability service.

Two distinct, clearly-labeled explanations are provided:

1. GLOBAL FEATURE IMPORTANCE
   - Comes directly from the trained Random Forest's `feature_importances_`.
   - Describes which features matter most across the whole dataset.
   - Same for every customer. Loaded from models/feature_importance.json.

2. PREDICTION EXPLANATION (per-customer)
   - Computed from the Logistic Regression model's coefficients applied to
     THIS customer's scaled feature values (contribution = coefficient *
     scaled_value). This is a standard, transparent linear-model explanation
     technique (equivalent to how logistic regression's log-odds are
     decomposed) — not SHAP, and not claimed to be SHAP.
   - This is intentionally simple and avoids the complexity/fragility of
     wiring SHAP to three different model types.
"""

import os
import json

import joblib
import numpy as np

MODELS_DIR = os.environ.get(
    "MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

_lr_model = None
_feature_columns = None


def load_explainer_artifacts():
    global _lr_model, _feature_columns
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    fc_path = os.path.join(MODELS_DIR, "feature_columns.json")

    if os.path.exists(lr_path) and os.path.exists(fc_path):
        _lr_model = joblib.load(lr_path)
        with open(fc_path) as f:
            _feature_columns = json.load(f)


def explain_prediction(scaled_feature_row: np.ndarray, top_n: int = 5):
    """
    scaled_feature_row: 1D array of scaled feature values for one customer,
                         in the same order as _feature_columns.
    Returns the top contributing factors (positive = increases churn risk,
    negative = decreases churn risk) for THIS prediction.
    """
    if _lr_model is None or _feature_columns is None:
        return []

    coefs = _lr_model.coef_[0]
    contributions = coefs * scaled_feature_row

    ranked = sorted(
        zip(_feature_columns, contributions),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": _clean_feature_name(name),
            "contribution": round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk",
        }
        for name, val in ranked
    ]


def _clean_feature_name(name: str) -> str:
    return name.replace("_", " ")
