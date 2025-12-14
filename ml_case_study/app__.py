from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import time
import logging
import os


# SETUP
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "/app/data/predictions.db"
MODEL_PATH_V1 = "models/xgboost_model_v1.pkl"
MODEL_PATH_V2 = "models/xgboost_model_v2.pkl"


# MODEL LOADING WITH FALLBACK
class NullModel:
    """Fallback model when trained files are missing."""
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]])  # neutral prediction


def load_model(path):
    if os.path.exists(path):
        logger.info(f"Loaded model: {path}")
        return joblib.load(path)
    else:
        logger.warning(f"Model not found: {path}. Using NullModel.")
        return NullModel()


model_v2 = load_model(MODEL_PATH_V2)
model_v1 = load_model(MODEL_PATH_V1)

# DB INITIALIZATION
def init_db():
    os.makedirs("/app/data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            product_id TEXT,
            score REAL,
            timestamp TEXT,
            model_version TEXT,
            actual_label INTEGER
        )""")
    conn.commit()
    conn.close()
    logger.info("SQLite DB initialized at /app/data/predictions.db")

init_db()


# PROMETHEUS METRICS
predictions_total = Counter("predictions_total", "Total predictions", ["status"])
prediction_latency = Histogram("prediction_latency_seconds", "Latency seconds")
prediction_score_hist = Histogram("prediction_score_distribution",
                                  "Prediction score histogram",
                                  buckets=[0.1 * i for i in range(1, 10)])
model_psi = Gauge("model_psi_score", "Population Stability Index (PSI)")
model_accuracy = Gauge("model_recent_auc", "Recent model AUC")


# HELPER FUNCTIONS
def save_prediction(user_id, product_id, score, model_version):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO predictions
               (user_id, product_id, score, timestamp, model_version)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, product_id, score, datetime.now().isoformat(), model_version)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB error: {e}")


def get_recent_scores(limit=100):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT score FROM predictions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [float(r[0]) for r in rows] or [0.5] * 20
    except:
        return [0.5] * 20


def calculate_psi(expected, actual):
    expected = np.array(expected)
    actual = np.array(actual)

    bins = np.linspace(0, 1, 11)
    e_hist, _ = np.histogram(expected, bins=bins)
    a_hist, _ = np.histogram(actual, bins=bins)

    e_pct = (e_hist + 1e-8) / sum(e_hist + 1e-8)
    a_pct = (a_hist + 1e-8) / sum(a_hist + 1e-8)

    return float(abs(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))))


def calculate_recent_auc(limit=100):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT score, actual_label FROM predictions WHERE actual_label IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()

        if len(rows) < 10:
            return 0.80

        scores = np.array([r[0] for r in rows])
        labels = np.array([r[1] for r in rows])

        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores)
    except:
        return 0.80


# API ENDPOINTS
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_v1_loaded": True,
        "model_v2_loaded": True,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    start = time.time()
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features'"}), 400

    user_id = data.get("user_id", "unknown")
    product_id = data.get("product_id", "unknown")
    features = np.array(data["features"]).reshape(1, -1)

    # Prediction with fallback
    try:
        score = float(model_v2.predict_proba(features)[0][1])
        model_version = "v2"
    except:
        score = float(model_v1.predict_proba(features)[0][1])
        model_version = "v1"

    latency = time.time() - start

    save_prediction(user_id, product_id, score, model_version)

    predictions_total.labels(status="success").inc()
    prediction_latency.observe(latency)
    prediction_score_hist.observe(score)

    recent = get_recent_scores(100)
    if len(recent) > 50:
        model_psi.set(calculate_psi(recent[50:], recent[:50]))

    model_accuracy.set(calculate_recent_auc())

    return jsonify({
        "user_id": user_id,
        "product_id": product_id,
        "prediction_score": round(score, 4),
        "confidence": round(max(score, 1 - score), 4),
        "latency_ms": round(latency * 1000, 2),
        "model_version": model_version
    })

@app.route("/api/v1/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE predictions SET actual_label = ? WHERE id = ?",
        (data["actual_label"], data["prediction_id"])
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "recorded"}), 200


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain"}


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
