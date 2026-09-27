import os
import json

from flask import Blueprint, jsonify

from services import predictor

insights_bp = Blueprint("insights", __name__)

STATIC_METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "static_metrics.json")
MODELS_DIR = os.environ.get(
    "MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models")
)


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


@insights_bp.route("/api/metrics", methods=["GET"])
def metrics():
    """Evaluation results from the existing ML pipeline (fixed, not recomputed)."""
    data = _load_json(STATIC_METRICS_PATH, {})
    return jsonify(data)


@insights_bp.route("/api/insights", methods=["GET"])
def insights():
    """Chart-ready data: class distribution, feature importance, probability distribution."""
    class_dist = _load_json(
        os.path.join(MODELS_DIR, "class_distribution.json"),
        {"churned": 1869, "not_churned": 5174},  # falls back to known dataset totals
    )
    feature_importance = predictor.get_feature_importance(top_n=10)
    prob_dist = _load_json(
        os.path.join(MODELS_DIR, "probability_distribution.json"), None
    )

    return jsonify({
        "class_distribution": class_dist,
        "feature_importance": feature_importance,
        "probability_distribution": prob_dist,
        "artifacts_ready": predictor.artifacts_ready(),
    })


@insights_bp.route("/api/status", methods=["GET"])
def status():
    return jsonify({"artifacts_ready": predictor.artifacts_ready()})
