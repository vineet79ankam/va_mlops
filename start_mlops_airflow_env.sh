#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlops-airflow

# Set Airflow home directory
export AIRFLOW_HOME=/mnt/d/ml_projects/va_mlops/airflow

# --- Start Jupyter Lab ---
echo "📓 Launching Jupyter Lab on port 8888..."
jupyter lab --port 8888 --no-browser &
echo "Jupyter Lab started. Access it at http://localhost:8888"

echo "Starting Airflow webserver on port 8080..."
airflow webserver --port 8080 > /mnt/d/ml_projects/va_mlops/airflow/webserver.log 2>&1 &

echo "Starting Airflow scheduler..."
airflow scheduler > /mnt/d/ml_projects/va_mlops/airflow/scheduler.log 2>&1 &

echo "Starting MLflow UI on port 5000..."
mlflow ui --backend-store-uri file:///mnt/d/ml_projects/va_mlops/mlflow --port 5000 > /mnt/d/ml_projects/va_mlops/mlflow/mlflow.log 2>&1 &

echo "All services started. Access Airflow at http://localhost:8080 and MLflow at http://127.0.0.1:5000"
