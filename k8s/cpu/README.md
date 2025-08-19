# CPU-Only Lightweight Deployment

This directory contains a lightweight, CPU-only version of the Kubernetes deployment for easier testing and development.

## Key Differences from Full Deployment

### Resource Reduction
- **UI Pod**: 50m CPU / 128Mi RAM (was 100m CPU / 256Mi RAM)
- **Infer Pod**: 100m CPU / 256Mi RAM (was 250m CPU / 512Mi RAM)
- **Replicas**: 1 replica each (was 2 replicas each)

### Storage Reduction
- **datasets-pvc**: 2Gi (was 20Gi)
- **models-pvc**: 1Gi (was 5Gi)
- **outputs-pvc**: 1Gi (was 10Gi)

### Images
- **CPU-only PyTorch**: Uses `python:3.10-slim` base with CPU-only PyTorch installation
- **No CUDA**: Removes GPU dependencies for faster builds and smaller images
- **Smaller footprint**: Significantly reduced image sizes

## Quick Start

### Build and Deploy CPU Version
```bash
# Build CPU-only images and deploy lightweight setup
make cpu-all

# Or step by step:
make build-cpu
make deploy-cpu
make wait
make url
```

### Using CPU-specific Data Preparation
```bash
# Use minimal test data instead of full dataset
make prepare-data-cpu
```

### Clean Up
```bash
# Delete CPU deployment
kubectl delete -k k8s/cpu

# Or full cleanup
make nuke
```

## Development Workflow

1. **Quick Testing**: Use `make cpu-all` for rapid iteration
2. **Resource Monitoring**: Check resource usage with `make get`
3. **Scaling**: Test with `make scale-ui n=2` or `make scale-infer n=2`
4. **Logs**: Monitor with `make logs name=deploy/ui`

## Benefits for Development

- **Faster builds**: No CUDA dependencies
- **Lower resource usage**: Runs on any machine with minikube
- **Quick iteration**: Reduced image sizes and faster deployments
- **Cost effective**: Minimal storage and compute requirements
- **Testing friendly**: Perfect for CI/CD and local development

## Migration Path

Once you're satisfied with testing on the CPU version:

1. Test with GPU version: `make all`
2. Scale up storage: Use full PVC sizes
3. Deploy to production with GPU-enabled nodes

## Notes

- The CPU version still supports the same API endpoints
- Training will be significantly slower on CPU
- Inference quality should be identical (assuming the same model)
- Perfect for functional testing, UI development, and integration testing
