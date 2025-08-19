#!/bin/bash

# CPU-only deployment script for quick testing
set -e

echo "🚀 Starting CPU-only lightweight deployment..."

# Check if minikube is running
if ! minikube status > /dev/null 2>&1; then
    echo "Minikube is not running. Please start minikube first:"
    echo "   minikube start"
    exit 1
fi

echo "Minikube is running"

# Build CPU images
echo "Building CPU-only images..."
make build-cpu

# Deploy CPU configuration
echo "Deploying CPU-only configuration..."
make deploy-cpu

# Wait for deployments
echo "Waiting for deployments to be ready..."
make wait

# Show status
echo "Deployment status:"
make get

# Show URL
echo "Access URL:"
make url

echo ""
echo "CPU-only deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Access the app: make serve"
echo "  2. Prepare test data: make prepare-data-cpu"
echo "  3. View logs: make logs name=deploy/ui"
echo "  4. Clean up: make nuke"
