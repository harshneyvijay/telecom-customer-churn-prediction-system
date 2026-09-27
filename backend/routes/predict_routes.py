import io
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, send_file

from services import predictor, explainer, database

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Missing or invalid JSON body"}), 400

    model_name = payload.pop("model", None)

    try:
        result = predictor.predict_single(payload, model_name=model_name)
    except predictor.ArtifactsNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    feature_row = result.pop("_feature_row")

    explanation = []
    try:
        scaled_row = predictor._scaler.transform([feature_row.values])[0]
        explanation = explainer.explain_prediction(scaled_row)
    except Exception:
        explanation = []

    database.log_prediction(
        result["churn_probability"], result["prediction"],
        result["risk_level"], result["model_used"],
    )

    result["explanation"] = explanation
    return jsonify(result)


@predict_bp.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    if df.empty:
        return jsonify({"error": "Uploaded CSV is empty"}), 400

    try:
        result_df = predictor.predict_batch(df)
    except predictor.ArtifactsNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {e}"}), 400

    for _, row in result_df.iterrows():
        database.log_prediction(
            float(row["churn_probability"]), int(row["prediction"]),
            row["risk_level"], "logistic_regression",
        )

    buf = io.StringIO()
    result_df.to_csv(buf, index=False)
    buf.seek(0)

    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name="churn_predictions.csv",
    )
