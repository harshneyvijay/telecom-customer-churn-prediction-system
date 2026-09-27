import os
from flask import Flask, send_from_directory

from services import predictor, explainer, database
from routes.predict_routes import predict_bp
from routes.insights_routes import insights_bp

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

app.register_blueprint(predict_bp)
app.register_blueprint(insights_bp)

# Load ML artifacts once at startup (not per-request)
predictor.load_artifacts()
explainer.load_explainer_artifacts()
database.init_db()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
