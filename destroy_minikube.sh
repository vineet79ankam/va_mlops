#!/bin/bash

echo "Deleting Minikube cluster (profile: va-mlops)..."
minikube delete --profile va-mlops
echo "🗑️ Minikube cluster deleted."
