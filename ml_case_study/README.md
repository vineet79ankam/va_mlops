
# Real-Time Multi-Tenant CTR Prediction Platform (MLOps Case Study)
## Machine Learning Use Case
### CTR Prediction
The service predicts the probability that a user will click on an ad or product based on precomputed features.

This repository contains a **implementation** of a real-time **Click-Through Rate (CTR) prediction service** designed for an online retail or advertising ecosystem (e.g., Amazon Ads, Walmart Connect, Target, etc.) using sample or dummy data.

It focuses on **model serving, multi-tenancy, monitoring, drift detection, and some of the MLOps best practices**, rather than full-scale end to end ML lifecycle.

**Implemented Scope in this Repository:**
- Model serving layer
- Monitoring and observability
- Multi-tenancy
- Fallback and resilience patterns

**Output includes:**
- CTR prediction score
- Confidence
- Model version used
- Inference latency


## Key Features
Below are some of the key features of this model serving API
- Real-time CTR prediction using REST API (Flask)
- Multi-tenant inference using `tenant_id` with multiple users (`user_id`) per tenant for demonstration
- Model versioning with fallback (`v2 → v1 → NullModel`) (Just for demonstration)
- Low-latency inference using Flask + Gunicorn
- Prometheus for metrics and model based monitoring (e.g.,Drift detection (PSI), Prediction monitoring, Latency, Prediction Rate, etc.)
- Tenant-level drift detection using PSI as the drift detection metrics
- Swagger for API documentation (e.g., Endpoint documentation, Request/response schemas, Try-it-out functionality)
- Dockerized local deployment using docker compose


## High-Level Architecture (Serving Focus)
**Event Flow (Conceptual):**
1. Retailer applications generate user interaction events
2. Features are computed upstream (assumed)
3. Real-time API serves CTR predictions and saved into the SQLite DB
4. Further, the predictions are logged for monitoring and analysis
5. Drift and performance metrics are tracked per tenant

## Model Versioning & Fallback Strategy
Note: The fallback stragey mentioned is only for the demonstration in actual it might vary based on the scenario and use case.
Easy rollback if the model fails to load during inference using model fallback strategy
    v2 version --> v1 version --> null

**Fallback mechanism in the inference serving**:
1. **Model v2** – latest preferred model
2. **Model v1** – previous stable model
3. **NullModel** – safe fallback returning a neutral prediction

## Multi-Tenancy Design
Multi-tenancy is handled using a required `tenant_id` in every request. Moreover, it provides tenant isolation
and monitoring of tanent specific drift detection using a single model for making the predictions.
This allows multiple retailers to share the same platform securely and efficiently.

## Monitoring & Drift Detection
### Metrics (Prometheus)
- Total predictions per tenant
- Prediction latency histogram
- Prediction score distribution
- Tenant-level PSI (Population Stability Index)
### Drift Detection
- PSI is computed on rolling windows of prediction scores
- Neutral fallback distributions are used during cold start
- Drift alerts are logged when thresholds are exceeded


## To run the Service Locally
### 1. Build & Start the API using docker-compose

docker-compose up --build

### 2. Health Check

docker-compose ps

curl http://localhost:5000/health

### 3. Make a Prediction using sample data

curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "amazon_in",
    "user_id": "u1",
    "product_id": "p1",
    "features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1]
  }'

### 4. API Documentation (Swagger)
Swagger UI can be accessed at:

URL: http://localhost:5000/apidocs/


### 5. Model Monitoring Dashboard (Prometheus)
Prometheus UI can be accessed at:
URL: http://localhost:9090

