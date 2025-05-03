#!/bin/bash

# --- CONFIG ---
PROJECT_DIR="/mnt/d/ml_projects/va_mlops"
ENV_NAME="mlops-core"

# --- Activate Conda ---
echo "🔄 Activating conda environment: $ENV_NAME"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME || { echo "❌ Failed to activate environment"; exit 1; }

# --- Change to project directory ---
echo "📁 Moving to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "❌ Project directory not found"; exit 1; }

# --- Version Checks ---
echo "🧪 Verifying core tool versions..."
echo "ZenML:  $(zenml version)"
echo "DVC:    $(dvc --version)"
echo "MLflow: $(mlflow --version)"

# --- ZenML Setup ---
if [ ! -d ".zen" ]; then
    echo "🔧 Initializing ZenML..."
    zenml init
fi

echo "⚙️  Bringing up ZenML stack..."
zenml up

# --- DVC Setup ---
if [ ! -d ".dvc" ]; then
    echo "🔧 Initializing DVC..."
    dvc init
fi

echo "⬇️  Pulling latest data artifacts with DVC..."
dvc pull || echo "⚠️  DVC pull skipped or failed (check remote setup)."

# --- Start MLflow ---
echo "🚀 Starting MLflow UI on port 5000..."
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 &

# --- Start Jupyter Lab ---
echo "📓 Launching Jupyter Lab on port 8888..."
jupyter lab --port 8888 --no-browser &

echo "✅ All tools launched successfully."
echo "🔗 MLflow:   http://localhost:5000"
echo "🔗 Jupyter:  http://localhost:8888"
