# Makefile for managing MLOps environment and Minikube cluster

# Paths to scripts
START_CLUSTER=./start_minikube.sh
STOP_CLUSTER=./stop_minikube.sh
DESTROY_CLUSTER=./destroy_minikube.sh
START_CORE=./start_mlops_core.sh

.PHONY: help start stop destroy core status

help:
	@echo "Available commands:"
	@echo "  make start      - Start Minikube cluster"
	@echo "  make stop       - Stop Minikube cluster"
	@echo "  make destroy    - Delete Minikube cluster"
	@echo "  make core       - Activate mlops-core environment"
	@echo "  make status     - Show Minikube cluster status"

start:
	bash $(START_CLUSTER)

stop:
	bash $(STOP_CLUSTER)

destroy:
	bash $(DESTROY_CLUSTER)

core:
	bash $(START_CORE)

status:
	minikube status --profile=va-mlops
