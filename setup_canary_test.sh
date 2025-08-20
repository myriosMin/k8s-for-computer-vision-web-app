#!/bin/bash

echo "🚀 Setting up Canary Deployment Test Environment"
echo "================================================"

# Check if namespace exists
if ! kubectl get namespace xview &> /dev/null; then
    echo "❌ Namespace 'xview' not found. Please run: make deploy-cpu"
    exit 1
fi

# Check if pods are running
echo "📋 Checking pod status..."
kubectl get pods -n xview

# Kill any existing port forwards
echo "🔧 Cleaning up existing port forwards..."
pkill -f "kubectl port-forward" 2>/dev/null || true

# Set up port forwarding
echo "🌐 Setting up port forwarding..."
kubectl port-forward svc/infer 9000:9000 -n xview &
INFER_PID=$!
kubectl port-forward svc/ui 8080:80 -n xview &
UI_PID=$!

# Wait for port forwarding to be ready
echo "⏳ Waiting for port forwarding to be ready..."
sleep 5

# Test connectivity
echo "🧪 Testing connectivity..."
if curl -s http://localhost:9000/healthz | grep -q "ok"; then
    echo "✅ Inference service is accessible"
else
    echo "❌ Cannot connect to inference service"
    kill $INFER_PID $UI_PID 2>/dev/null
    exit 1
fi

if curl -s http://localhost:9000/model-status &> /dev/null; then
    echo "✅ Model status endpoint is accessible"
    echo "📊 Current model status:"
    curl -s http://localhost:9000/model-status | python3 -m json.tool 2>/dev/null || curl -s http://localhost:9000/model-status
else
    echo "❌ Model status endpoint not accessible"
fi

echo ""
echo "🎯 Environment ready! You can now:"
echo "1. Run the test: python test_canary_deployment_k8s.py --ui-url http://localhost:8080 --infer-url http://localhost:9000"
echo "2. Access UI: http://localhost:8080"
echo "3. Check model status: curl http://localhost:9000/model-status"
echo ""
echo "📝 To stop port forwarding later:"
echo "   kill $INFER_PID $UI_PID"
echo "   or run: pkill -f 'kubectl port-forward'"
