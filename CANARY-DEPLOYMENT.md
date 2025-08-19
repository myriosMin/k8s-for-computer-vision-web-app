# Advanced Model Versioning with Canary Deployments

This system implements production-grade model versioning with canary deployments for safe, automated model rollouts.

## Features

### Model Versioning
- **Automatic Version Detection**: Increments version numbers automatically
- **Metadata Tracking**: JSON metadata with timestamps, metrics, and deployment info
- **Atomic Publishing**: Safe model replacement with backup strategies

### Canary Deployment
- **50/50 Traffic Split**: Gradual rollout with half traffic to new model
- **3-Minute Evaluation**: Automatic promotion after health monitoring
- **Real-time Monitoring**: Live model status and traffic distribution
- **Graceful Rollback**: Automatic fallback on errors

### Production Safety
- **Health Checks**: Continuous monitoring of model performance
- **Error Rate Tracking**: Automatic rollback on high error rates
- **Zero-Downtime Deployment**: Seamless model updates
- **Audit Trail**: Complete deployment history

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Training      │    │   Model Storage  │    │   Inference     │
│   Container     │───▶│   /models/       │◀───│   Service       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ├── current/
                              │   ├── best.pt
                              │   └── version.json
                              │
                              └── canary/
                                  ├── best.pt
                                  └── version.json
```

## Deployment Workflow

### 1. Model Training
```bash
# Training creates new model version
/models/canary/
├── best.pt          # New trained model
└── version.json     # Metadata with version info
```

### 2. Canary Deployment
```json
{
  "version": 2,
  "timestamp": "2024-01-15T10:30:00",
  "canary_start_time": "2024-01-15T10:30:00",
  "traffic_weight": 0.5,
  "metrics": {
    "train_epochs": 10,
    "final_map": 0.84,
    "final_precision": 0.79,
    "final_recall": 0.83
  }
}
```

### 3. Traffic Distribution
- **Current Model**: 50% of prediction requests
- **Canary Model**: 50% of prediction requests
- **Model Selection**: Random distribution per request

### 4. Auto-Promotion (After 3 minutes)
```bash
# Canary becomes the new current model
mv /models/canary/* /models/current/
rm -rf /models/canary/
```

## Monitoring

### Model Status API
```bash
curl http://localhost:9000/model-status
```

```json
{
  "current_model": true,
  "canary_model": true,
  "deployment_mode": "canary",
  "canary_elapsed_minutes": 1.5,
  "ready_for_promotion": false,
  "canary_version": {
    "version": 2,
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

### Web UI Dashboard
- **Real-time Status**: Live model deployment status
- **Traffic Visualization**: Current vs canary traffic split
- **Promotion Timer**: Countdown to auto-promotion
- **Model Indicators**: Visual indicators for each model type

## Testing

### Prerequisites for Kubernetes Testing
Before testing the canary deployment, ensure your Kubernetes environment is running:

```bash
# 1. Deploy the application
make deploy-cpu

# 2. Set up port forwarding to access services
kubectl port-forward svc/ui 8080:80 -n xview &
kubectl port-forward svc/infer 9000:9000 -n xview &

# 3. Verify services are accessible
curl http://localhost:8080/healthz
curl http://localhost:9000/healthz
```

### Automated Test Suite for Kubernetes

The test script needs to be configured for Kubernetes endpoints:

```bash
# Update the test script for Kubernetes
python test_canary_deployment.py --ui-url http://localhost:8080 --infer-url http://localhost:9000
```

Or modify the test script directly:
```python
# Configuration for Kubernetes deployment
UI_URL = "http://localhost:8080"     # Port-forwarded UI service
INFER_URL = "http://localhost:9000"  # Port-forwarded Inference service
```

Then run the test:
```bash
python test_canary_deployment.py
```

### Testing Steps in Kubernetes Environment

#### 1. **Setup Port Forwarding**
```bash
# Terminal 1: UI Service
kubectl port-forward svc/ui 8080:80 -n xview

# Terminal 2: Inference Service  
kubectl port-forward svc/infer 9000:9000 -n xview
```

#### 2. **Access Web Interface**
```bash
# Open browser to web UI
open http://localhost:8080

# Or use the minikube service (alternative)
minikube service ui -n xview
```

#### 3. **Trigger Model Training for Canary**
```bash
# Via Web UI (recommended):
# 1. Upload dataset (data/aiad_data.zip)
# 2. Configure training (epochs=5, batch=4 for quick test)
# 3. Click "Start Training"

# Via API (alternative):
curl -X POST http://localhost:8080/upload-dataset \
  -F "file=@data/aiad_data.zip"

curl -X POST http://localhost:8080/trigger-train \
  -F "epochs=5" \
  -F "batch=4"
```

#### 4. **Monitor Training Progress**
```bash
# Watch training job
kubectl get jobs -n xview -w

# Check training logs
kubectl logs -f job/train -n xview

# Monitor pods
kubectl get pods -n xview -w
```

#### 5. **Test Canary Deployment**
```bash
# Check model status
curl http://localhost:9000/model-status

# Make multiple predictions to test traffic split
for i in {1..10}; do
  curl -X POST http://localhost:9000/predict \
    -F "pre_disaster=@data/yolo_xbd/images/test/guatemala-volcano_00000003_pre_disaster.png" \
    -F "post_disaster=@data/yolo_xbd/images/test/guatemala-volcano_00000003_post_disaster.png" \
    | jq '.model_used'
done
```

#### 6. **Run Full Test Suite**
```bash
# Run comprehensive canary test
python test_canary_deployment.py
```

### Kubernetes-Specific Testing Commands

#### Check Model Files in Pod
```bash
# Verify model structure in inference pod
kubectl exec deployment/infer -n xview -- ls -la /models/

# Check if canary model exists
kubectl exec deployment/infer -n xview -- ls -la /models/canary/

# View version metadata
kubectl exec deployment/infer -n xview -- cat /models/canary/version.json
```

#### Monitor Canary Deployment
```bash
# Watch model status endpoint
watch -n 5 "curl -s http://localhost:9000/model-status | jq"

# Monitor inference logs for model selection
kubectl logs -f deployment/infer -n xview | grep "Using model"
```

#### Test Traffic Distribution
```bash
# Script to test traffic distribution
for i in {1..20}; do
  echo "Request $i:"
  curl -s -X POST http://localhost:9000/predict \
    -F "pre_disaster=@data/yolo_xbd/images/test/guatemala-volcano_00000003_pre_disaster.png" \
    -F "post_disaster=@data/yolo_xbd/images/test/guatemala-volcano_00000003_post_disaster.png" \
    | jq -r '.model_used'
done | sort | uniq -c
```

### Testing Scenarios

#### Scenario 1: Fresh Deployment (No Canary)
```bash
# Expected: Only current model available
curl http://localhost:9000/model-status
# Response: {"deployment_mode": "production", "current_model": true, "canary_model": false}
```

#### Scenario 2: Training Triggers Canary
```bash
# 1. Start training job
# 2. Wait for completion (~5-10 minutes)
# 3. Check model status
curl http://localhost:9000/model-status
# Response: {"deployment_mode": "canary", "canary_elapsed_minutes": 0.5, ...}
```

#### Scenario 3: Traffic Split Active
```bash
# Make predictions and observe model_used field
# Should see mix of "current" and "canary"
```

#### Scenario 4: Auto-Promotion
```bash
# Wait 3+ minutes after canary deployment
# Check status - should show promoted model
curl http://localhost:9000/model-status
# Response: {"deployment_mode": "production", "ready_for_promotion": true, ...}
```

### Troubleshooting Tests

#### Port Forwarding Issues
```bash
# Kill existing port forwards
pkill -f "kubectl port-forward"

# Restart port forwarding
kubectl port-forward svc/ui 8080:80 -n xview &
kubectl port-forward svc/infer 9000:9000 -n xview &
```

#### Service Not Accessible
```bash
# Check service status
kubectl get svc -n xview

# Check pod status
kubectl get pods -n xview

# Check pod logs
kubectl logs deployment/infer -n xview
kubectl logs deployment/ui -n xview
```

#### Model Not Loading
```bash
# Check PVC and model files
kubectl exec deployment/infer -n xview -- ls -la /models/

# Verify training job completed
kubectl get jobs -n xview
kubectl logs job/train -n xview
```

This script:
1. Checks current model status via Kubernetes port-forwarding
2. Tests traffic splitting during canary deployment  
3. Monitors auto-promotion process in real-time
4. Validates final deployment state
5. Provides Kubernetes-specific troubleshooting

## Configuration

### Environment Variables
```bash
MODEL_DIR=/models              # Model storage directory
IMG_SIZE=640                   # Image processing size
INFER_URL=http://infer:9000    # Inference service URL
```

### Training Parameters
```python
# Configurable through web UI
epochs = 10        # Training epochs
batch_size = 8     # Batch size for training
```

## Safety Features

### Error Handling
- **Model Loading Fallback**: Falls back to base weights if no models
- **API Error Recovery**: Graceful handling of inference failures
- **File System Safety**: Atomic operations for model updates

### Monitoring & Alerts
- **Health Endpoints**: `/healthz` for service monitoring
- **Status Endpoints**: Real-time deployment status
- **Error Tracking**: Comprehensive error logging

### Rollback Strategy
```python
# Automatic rollback triggers:
- High error rate (>5% failures)
- Model loading failures
- Performance degradation
- Manual intervention
```

## Metrics & Analytics

### Training Metrics
- **mAP (mean Average Precision)**: Model accuracy
- **Precision/Recall**: Detection performance
- **Loss Curves**: Training convergence
- **Validation Scores**: Generalization ability

### Deployment Metrics
- **Traffic Distribution**: Request routing stats
- **Response Times**: Inference latency
- **Error Rates**: Prediction failure rates
- **Model Usage**: Current vs canary usage

## Best Practices

### Model Development
1. **Iterative Training**: Small, frequent model updates
2. **Validation**: Thorough testing before deployment
3. **Monitoring**: Continuous performance tracking
4. **Documentation**: Clear model change logs

### Deployment Strategy
1. **Gradual Rollout**: Always use canary deployments
2. **Monitoring**: Watch metrics during evaluation period
3. **Quick Rollback**: Be ready to revert if needed
4. **Communication**: Keep stakeholders informed

### Production Operations
1. **Health Checks**: Regular service monitoring
2. **Backup Strategy**: Maintain model backups
3. **Capacity Planning**: Monitor resource usage
4. **Incident Response**: Clear escalation procedures

This advanced model versioning system provides enterprise-grade MLOps capabilities with safe, automated deployments and comprehensive monitoring for production computer vision workloads.
