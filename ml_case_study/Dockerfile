FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed for Flask + Swagger
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create folders for models & data
RUN mkdir -p /app/models /app/data

# Copy application
COPY app.py .
COPY config.yaml .
# COPY models/ ./models/

# Expose API port
EXPOSE 5000

# Run API using gunicorn
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]
