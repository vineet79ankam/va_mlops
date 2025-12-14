from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flasgger import Swagger
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import time
import logging
import os
import yaml


# load the config file & defining the variables
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

APP_NAME = CONFIG["app"]["name"]
ENV = CONFIG["app"]["environment"]
DB_PATH = CONFIG["database"]["path"]
FETCH_LIMIT = CONFIG["database"]["fetch_limit"]
MODEL_V2_PATH = CONFIG["model"]["v2_path"]
MODEL_V1_PATH = CONFIG["model"]["v1_path"]
NULL_PREDICTION = CONFIG["model"]["null_prediction"]
EXPECTED_FEATURES = CONFIG["features"]["expected_feature_count"]
PSI_WINDOW = CONFIG["monitoring"]["psi"]["window_size"]
PSI_BASELINE = CONFIG["monitoring"]["psi"]["baseline_size"]



### Define the app ####
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(APP_NAME)

### Swagger configuration ####
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "CTR Prediction Service",
        "description": "Multi-tenant real-time CTR prediction API",
        "version": "1.0.0",
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http"],
    "consumes": ["application/json"],
    "produces": ["application/json"]
})



#### Load the model with fallback (v2 --> v1 --> null)
class NullModel:
    version = "null"

    def predict_proba(self, X):
        return np.array([[1 - NULL_PREDICTION, NULL_PREDICTION]])


def load_model_with_fallback(v2_path, v1_path):
    if os.path.exists(v2_path):
        try:
            model = joblib.load(v2_path)
            model.version = "v2"
            logger.info("Loaded model v2")
            return model
        except Exception as e:
            logger.error(f"Failed loading v2: {e}")

    if os.path.exists(v1_path):
        try:
            model = joblib.load(v1_path)
            model.version = "v1"
            logger.warning("Falling back to model v1")
            return model
        except Exception as e:
            logger.error(f"Failed loading v1: {e}")

    logger.critical("No trained models available. Using NullModel.")
    return NullModel()


MODEL = load_model_with_fallback(MODEL_V2_PATH, MODEL_V1_PATH)


#### DB setup ####
def init_db():
    """Initialize SQLite database - fail gracefully if unable"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT,
                user_id TEXT,
                product_id TEXT,
                score REAL,
                model_version TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    except sqlite3.OperationalError as e:
        logger.warning(f"Database init failed: {e}")
        logger.warning("Continuing without persistent database")

## initialize the db
init_db()


#### Prometheus Metrics ####
predictions_total = Counter(
    "predictions_total",
    "Total predictions",
    ["tenant_id", "status"]
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["tenant_id"]
)

prediction_score_hist = Histogram(
    "prediction_score_distribution",
    "Prediction score distribution",
    ["tenant_id"],
    buckets=[i / 10 for i in range(1, 10)]
)

tenant_model_psi = Gauge(
    "tenant_model_psi",
    "PSI drift per tenant",
    ["tenant_id"]
)


#### Define Helper Functions ####
def save_prediction(tenant_id, user_id, product_id, score):
    """Save prediction to database - fail gracefully if unable"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute(
            """INSERT INTO predictions
               (tenant_id, user_id, product_id, score, model_version, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tenant_id, user_id, product_id, score, MODEL.version, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        logger.info(f"Prediction saved for {tenant_id}")
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not save prediction: {e}")


def get_recent_scores(tenant_id, limit):
    """Get recent prediction scores for tenant"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        rows = conn.execute(
            """SELECT score FROM predictions
               WHERE tenant_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (tenant_id, limit)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows] or [NULL_PREDICTION] * 20
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not fetch scores: {e}")
        return [NULL_PREDICTION] * 20


def calculate_psi(expected, actual):
    """Calculate Population Stability Index"""
    bins = np.linspace(0, 1, 11)
    e_hist, _ = np.histogram(expected, bins=bins)
    a_hist, _ = np.histogram(actual, bins=bins)

    e_pct = (e_hist + 1e-8) / sum(e_hist + 1e-8)
    a_pct = (a_hist + 1e-8) / sum(a_hist + 1e-8)

    return float(abs(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))))



### Define API Endpoints with Swagger docs####
# API for health status
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Service health status
        schema:
          type: object
          properties:
            service:
              type: string
              example: "CTR Prediction Service"
            environment:
              type: string
              example: "production"
            active_model_version:
              type: string
              example: "v2"
            timestamp:
              type: string
              example: "2025-12-14T10:44:00"
    """
    return jsonify({
        "service": APP_NAME,
        "environment": ENV,
        "active_model_version": MODEL.version,
        "timestamp": datetime.utcnow().isoformat()
    }), 200


# Prediction API
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    """
    Real-time CTR prediction
    ---
    tags:
      - Prediction
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - tenant_id
            - features
          properties:
            tenant_id:
              type: string
              example: "walmart_us"
            user_id:
              type: string
              example: "user_123"
            product_id:
              type: string
              example: "prod_456"
            features:
              type: array
              items:
                type: number
              example: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]
    responses:
      200:
        description: Prediction successful
        schema:
          type: object
          properties:
            tenant_id:
              type: string
            prediction_score:
              type: number
              format: float
            confidence:
              type: number
              format: float
            model_version:
              type: string
            latency_ms:
              type: number
              format: float
      400:
        description: Invalid request
      500:
        description: Server error
    """
    start = time.time()

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body required"}), 400

        tenant_id = data.get("tenant_id")
        features = data.get("features")

        if not tenant_id or not features:
            return jsonify({"error": "tenant_id and features required"}), 400

        if len(features) != EXPECTED_FEATURES:
            return jsonify({"error": f"Expected {EXPECTED_FEATURES} features, got {len(features)}"}), 400

        # Make prediction
        X = np.array(features).reshape(1, -1)
        score = float(MODEL.predict_proba(X)[0][1])

        # Save to DB
        save_prediction(
            tenant_id,
            data.get("user_id", "unknown"),
            data.get("product_id", "unknown"),
            score
        )

        # Calculate latency
        latency = time.time() - start

        # Update metrics
        predictions_total.labels(tenant_id=tenant_id, status="success").inc()
        prediction_latency.labels(tenant_id=tenant_id).observe(latency)
        prediction_score_hist.labels(tenant_id=tenant_id).observe(score)

        # Calculate PSI if enough data
        recent = get_recent_scores(tenant_id, PSI_WINDOW)
        if len(recent) >= PSI_WINDOW:
            psi_score = calculate_psi(recent[PSI_BASELINE:], recent[:PSI_BASELINE])
            tenant_model_psi.labels(tenant_id=tenant_id).set(psi_score)

            if psi_score > 0.25:
                logger.warning(f"PSI Alert for {tenant_id}: {psi_score:.4f}")

        return jsonify({
            "tenant_id": tenant_id,
            "prediction_score": round(score, 4),
            "confidence": round(max(score, 1 - score), 4),
            "model_version": MODEL.version,
            "latency_ms": round(latency * 1000, 2)
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        predictions_total.labels(tenant_id="unknown", status="error").inc()
        return jsonify({"error": str(e)}), 500


# Get predictions from db API
@app.route("/api/v1/predictions", methods=["GET"])
def get_predictions():
    """
    Fetch prediction history for a tenant
    ---
    tags:
      - Analytics
    parameters:
      - name: tenant_id
        in: query
        required: true
        type: string
      - name: limit
        in: query
        required: false
        type: integer
        default: 100
    responses:
      200:
        description: List of predictions
        schema:
          type: object
          properties:
            tenant_id:
              type: string
            predictions:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                  score:
                    type: number
                  timestamp:
                    type: string
      400:
        description: Missing tenant_id
    """
    try:
        tenant_id = request.args.get("tenant_id")
        limit = request.args.get("limit", default=FETCH_LIMIT, type=int)

        if not tenant_id:
            return jsonify({"error": "tenant_id required"}), 400

        conn = sqlite3.connect(DB_PATH, timeout=5)
        rows = conn.execute(
            """SELECT id, user_id, product_id, score, model_version, timestamp
               FROM predictions
               WHERE tenant_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (tenant_id, limit)
        ).fetchall()
        conn.close()

        predictions = [
            {
                "id": r[0],
                "user_id": r[1],
                "product_id": r[2],
                "score": r[3],
                "model_version": r[4],
                "timestamp": r[5]
            }
            for r in rows
        ]

        return jsonify({
            "tenant_id": tenant_id,
            "count": len(predictions),
            "predictions": predictions
        }), 200

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({"error": str(e)}), 500


# Get metrics from prometheus API
@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Prometheus metrics endpoint
    ---
    tags:
      - Internal
    responses:
      200:
        description: Prometheus metrics in text format
    """
    return generate_latest(), 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    logger.info(f"Starting {APP_NAME} in {ENV} mode")
    logger.info(f"Using model: {MODEL.version}")
    logger.info(f"Swagger UI: http://localhost:5000/apidocs/")
    logger.info(f"Prometheus UI: http://localhost:9090")
    app.run(host="0.0.0.0", port=5000, debug=False)