#!/usr/bin/env python3
"""
Test script for canary deployment functionality.

This script simulates the complete canary deployment workflow:
1. Train a new model version
2. Verify canary deployment is set up
3. Make multiple predictions to test traffic split
4. Wait for auto-promotion
5. Verify promotion

Usage:
    python test_canary_deployment.py
    python test_canary_deployment.py --ui-url http://localhost:8080 --infer-url http://localhost:9000
"""

import requests
import time
import json
import argparse
import sys
from collections import Counter

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test canary deployment functionality')
    parser.add_argument('--ui-url', default='http://localhost:5000', 
                       help='URL for UI service (default: http://localhost:5000)')
    parser.add_argument('--infer-url', default='http://localhost:9000',
                       help='URL for inference service (default: http://localhost:9000)')
    parser.add_argument('--test-images', default='data/yolo_xbd/images/test',
                       help='Path to test images directory')
    parser.add_argument('--requests', type=int, default=20,
                       help='Number of requests for traffic split test (default: 20)')
    parser.add_argument('--wait-minutes', type=int, default=4,
                       help='Minutes to wait for auto-promotion (default: 4)')
    return parser.parse_args()

# Global configuration (will be set by parse_args)
args = None

def test_model_status():
    """Test model status endpoint"""
    print("📊 Testing model status endpoint...")
    try:
        response = requests.get(f"{args.infer_url}/model-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Model status: {json.dumps(status, indent=2)}")
            return status
        else:
            print(f"❌ Failed to get model status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting model status: {e}")
        return None

def test_canary_traffic_split():
    """Test that traffic is split between current and canary models"""
    print("\n🚀 Testing canary traffic split...")
    
    # Make multiple requests to see traffic distribution
    model_usage = Counter()
    num_requests = args.requests
    
    # Create simple test images (we'll just use placeholder data)
    test_data = {
        'pre_disaster': ('test_pre.png', b'fake_image_data', 'image/png'),
        'post_disaster': ('test_post.png', b'fake_image_data', 'image/png')
    }
    
    print(f"Making {num_requests} prediction requests...")
    for i in range(num_requests):
        try:
            response = requests.post(f"{args.infer_url}/predict", files=test_data)
            if response.status_code == 200:
                result = response.json()
                model_used = result.get('model_used', 'unknown')
                model_usage[model_used] += 1
                print(f"Request {i+1}: {model_used}")
            else:
                print(f"Request {i+1}: Failed ({response.status_code})")
        except Exception as e:
            print(f"Request {i+1}: Error - {e}")
        
        time.sleep(0.1)  # Small delay
    
    print(f"\n📊 Traffic distribution: {dict(model_usage)}")
    
    # Check if we have a reasonable split (not necessarily perfect 50/50)
    if len(model_usage) > 1:
        print("✅ Traffic is being split between models!")
        return True
    else:
        print("ℹ️  All traffic going to single model (no canary or already promoted)")
        return False

def wait_for_promotion(max_wait_minutes=None):
    """Wait for canary to be promoted"""
    if max_wait_minutes is None:
        max_wait_minutes = args.wait_minutes
        
    print(f"\n⏳ Waiting up to {max_wait_minutes} minutes for auto-promotion...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_minutes * 60:
        status = test_model_status()
        if status:
            if status.get('deployment_mode') == 'production':
                print("✅ Model has been promoted to production!")
                return True
            elif status.get('ready_for_promotion'):
                print("🎯 Model is ready for promotion...")
        
        print("⏳ Still waiting...")
        time.sleep(30)  # Check every 30 seconds
    
    print("⏰ Timeout waiting for promotion")
    return False

def simulate_training():
    """Simulate triggering a training job that would create a canary"""
    print("\n🎯 Simulating model training for canary deployment...")
    print("Note: This would normally trigger through the UI training workflow")
    print(f"For full testing, trigger a training job through the web UI at {args.ui_url}")

def main():
    """Main test function"""
    global args
    args = parse_args()
    
    print("🧪 Canary Deployment Test Suite")
    print("=" * 50)
    print(f"UI URL: {args.ui_url}")
    print(f"Inference URL: {args.infer_url}")
    print(f"Test requests: {args.requests}")
    print(f"Wait time: {args.wait_minutes} minutes")
    print("=" * 50)
    
    # Test 1: Check current model status
    initial_status = test_model_status()
    if not initial_status:
        print("❌ Cannot connect to inference service. Make sure it's running.")
        print("\nFor Kubernetes deployment:")
        print("1. Ensure pods are running: kubectl get pods -n xview")
        print("2. Set up port forwarding:")
        print(f"   kubectl port-forward svc/ui 8080:80 -n xview")
        print(f"   kubectl port-forward svc/infer 9000:9000 -n xview")
        print("3. Then run: python test_canary_deployment.py --ui-url http://localhost:8080 --infer-url http://localhost:9000")
        return
    
    # Test 2: Test traffic splitting if canary exists
    if initial_status.get('deployment_mode') == 'canary':
        test_canary_traffic_split()
        
        # Test 3: Wait for auto-promotion
        wait_for_promotion()
        
        # Test 4: Check final status
        final_status = test_model_status()
        
    else:
        print("\nℹ️  No canary deployment currently active.")
        simulate_training()
        print("\nTo test canary deployment:")
        print("1. Upload a dataset through the web UI")
        print("2. Start a training job")
        print("3. The new model will be deployed as a canary")
        print("4. Run this test again to see traffic splitting")
    
    print("\n✅ Test suite completed!")
    print("\nCanary Deployment Features:")
    print("- 🎯 50/50 traffic split between current and canary models")
    print("- ⏰ 3-minute evaluation period")
    print("- 🚀 Automatic promotion after evaluation")
    print("- 📊 Real-time model status monitoring")
    print("- 🔄 Graceful rollback capability")

if __name__ == "__main__":
    main()
