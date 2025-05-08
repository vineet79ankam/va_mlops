#!/bin/bash

# Start Minikube with recommended settings for Kubeflow
echo "Starting Minikube with required resources..."
minikube start --driver=docker --cpus=4 --memory=12288

# Set up namespace
echo "Creating kubeflow namespace..."
kubectl create namespace kubeflow || echo "Namespace already exists"

# Clone Kubeflow manifests if not already
if [ ! -d "manifests" ]; then
  echo "Cloning Kubeflow manifests..."
  git clone https://github.com/kubeflow/manifests.git
fi

# Deploy Kubeflow using kustomize
cd manifests
echo "Deploying Kubeflow with Kustomize..."
kustomize build example | kubectl apply -f -

# Wait and then port forward
echo "Waiting for Kubeflow services to initialize..."
sleep 60
echo "Port forwarding Kubeflow dashboard to localhost:8080..."
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
echo "Kubeflow is now running. Access it at http://localhost:8080"
echo "To stop Minikube, run: minikube stop"
echo "To delete Minikube, run: minikube delete"
echo "To access the Kubeflow dashboard, open your browser and go to http://localhost:8080"
echo "Kubeflow deployment complete."