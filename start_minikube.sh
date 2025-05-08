#!/bin/bash

PROFILE="va-mlops"

echo "🚀 Starting Minikube cluster with profile: $PROFILE..."

# Start minikube with specific resources (adjust as needed)
minikube start --profile=$PROFILE --memory=4096 --cpus=2 --driver=docker

# Set kubectl to use the correct context
kubectl config use-context $PROFILE

# Enable local Docker environment inside Minikube
eval $(minikube -p $PROFILE docker-env)

echo "✅ Minikube cluster '$PROFILE' is up and running."
kubectl get nodes
