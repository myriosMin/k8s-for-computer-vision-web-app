from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, requests
import tempfile
import zipfile
import subprocess, json
from kubernetes import client, config

INFER_URL = os.getenv("INFER_URL", "http://infer:9000")
NAMESPACE = os.getenv("NAMESPACE", "xview")
CONFIGMAP_NAME = "app-config"

# Initialize Kubernetes client
try:
    # Load in-cluster config when running in a pod
    config.load_incluster_config()
except:
    # Fall back to kubeconfig for local development
    config.load_kube_config()

k8s_v1 = client.CoreV1Api()
k8s_batch = client.BatchV1Api()

def patch_configmap(overrides: dict):
    """
    Update the app-config ConfigMap using Kubernetes API.
    """
    try:
        # Get current configmap
        configmap = k8s_v1.read_namespaced_config_map(
            name=CONFIGMAP_NAME, 
            namespace=NAMESPACE
        )
        
        # Update data
        if not configmap.data:
            configmap.data = {}
        
        for key, value in overrides.items():
            configmap.data[key] = str(value)
        
        # Patch the configmap
        k8s_v1.patch_namespaced_config_map(
            name=CONFIGMAP_NAME,
            namespace=NAMESPACE,
            body=configmap
        )
        
    except Exception as e:
        raise Exception(f"Failed to update configmap: {str(e)}")

def create_preprocess_job():
    """
    Create a preprocess job using Kubernetes API.
    """
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "preprocess",
            "namespace": NAMESPACE,
            "labels": {
                "app": "preprocess",
                "part-of": "xview"
            }
        },
        "spec": {
            "backoffLimit": 2,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "preprocess",
                        "part-of": "xview"
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "preprocess",
                        "image": "localhost/worker:cpu",
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["python", "-m", "app.preprocess"],
                        "envFrom": [{
                            "configMapRef": {
                                "name": "app-config"
                            }
                        }],
                        "volumeMounts": [{
                            "name": "datasets",
                            "mountPath": "/data"
                        }],
                        "resources": {
                            "requests": {
                                "cpu": "100m",
                                "memory": "256Mi"
                            },
                            "limits": {
                                "cpu": "500m",
                                "memory": "1Gi"
                            }
                        }
                    }],
                    "volumes": [{
                        "name": "datasets",
                        "persistentVolumeClaim": {
                            "claimName": "datasets-pvc"
                        }
                    }]
                }
            }
        }
    }
    
    try:
        # Delete existing job if it exists
        try:
            k8s_batch.delete_namespaced_job(
                name="preprocess",
                namespace=NAMESPACE,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5
                )
            )
            # Wait a moment for cleanup
            import time
            time.sleep(2)
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if job doesn't exist
                raise
        
        # Create the new job
        k8s_batch.create_namespaced_job(
            namespace=NAMESPACE,
            body=job_manifest
        )
        
    except Exception as e:
        raise Exception(f"Failed to create preprocess job: {str(e)}")

def create_train_job():
    """
    Create a training job using Kubernetes API.
    """
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "train",
            "namespace": NAMESPACE,
            "labels": {
                "app": "train",
                "part-of": "xview"
            }
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "train",
                        "part-of": "xview"
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "train",
                        "image": "localhost/worker:cpu",
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["python", "-m", "app.train"],
                        "envFrom": [{
                            "configMapRef": {
                                "name": "app-config"
                            }
                        }],
                        "env": [
                            {
                                "name": "BASE_WEIGHTS",
                                "valueFrom": {
                                    "configMapKeyRef": {
                                        "name": "app-config",
                                        "key": "BASE_WEIGHTS",
                                        "optional": True
                                    }
                                }
                            },
                            {
                                "name": "EPOCHS",
                                "valueFrom": {
                                    "configMapKeyRef": {
                                        "name": "app-config",
                                        "key": "EPOCHS",
                                        "optional": True
                                    }
                                }
                            },
                            {
                                "name": "IMG_SIZE",
                                "valueFrom": {
                                    "configMapKeyRef": {
                                        "name": "app-config",
                                        "key": "IMG_SIZE",
                                        "optional": True
                                    }
                                }
                            },
                            {
                                "name": "BATCH",
                                "valueFrom": {
                                    "configMapKeyRef": {
                                        "name": "app-config",
                                        "key": "BATCH",
                                        "optional": True
                                    }
                                }
                            }
                        ],
                        "volumeMounts": [
                            {
                                "name": "datasets",
                                "mountPath": "/data"
                            },
                            {
                                "name": "models",
                                "mountPath": "/models"
                            }
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "200m",
                                "memory": "512Mi"
                            },
                            "limits": {
                                "cpu": "1",
                                "memory": "2Gi"
                            }
                        }
                    }],
                    "volumes": [
                        {
                            "name": "datasets",
                            "persistentVolumeClaim": {
                                "claimName": "datasets-pvc"
                            }
                        },
                        {
                            "name": "models",
                            "persistentVolumeClaim": {
                                "claimName": "models-pvc"
                            }
                        }
                    ]
                }
            }
        }
    }
    
    try:
        # Delete existing job if it exists
        try:
            k8s_batch.delete_namespaced_job(
                name="train",
                namespace=NAMESPACE,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5
                )
            )
            # Wait a moment for cleanup
            import time
            time.sleep(2)
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if job doesn't exist
                raise
        
        # Create the new job
        k8s_batch.create_namespaced_job(
            namespace=NAMESPACE,
            body=job_manifest
        )
        
    except Exception as e:
        raise Exception(f"Failed to create train job: {str(e)}")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Health check
@app.get("/healthz")
def healthz():
    return {"ok": True}

# Serve main training template
@app.get("/")
def index():
    # Page 1: model training, batch upload, config, metrics
    return render_template("index.html")

# Serve inference template
@app.get("/inference")
def inference_page():
    # Page 2: inference
    return render_template("inference.html")

# Mount uploaded dataset
@app.post("/upload-dataset")
def upload_dataset():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    save_path = os.path.join("/data/raw", file.filename)  # volume mount
    file.save(save_path)

    return jsonify({"status": "ok", "path": save_path}), 200

# Trigger preprocessing job
@app.post("/trigger-preprocess")
def trigger_preprocess():
    try:
        dataset_file = request.form.get("dataset", "aiad_data.zip")
        img_size = "640"  # Fixed image size for consistency

        # Build the full path for the uploaded dataset
        if not dataset_file.startswith("/data/"):
            dataset_path = f"/data/raw/{dataset_file}"
        else:
            dataset_path = dataset_file

        # Update configmap with new settings
        patch_configmap({
            "RAW_ZIP_PATH": dataset_path,
            "IMG_SIZE": img_size
        })

        # Create the preprocess job
        create_preprocess_job()
        
        return jsonify({
            "status": "ok", 
            "message": "Preprocess job started successfully",
            "dataset": dataset_path
        }), 202

    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to start preprocess job: {str(e)}"
        }), 500

# Trigger training job
@app.post("/trigger-train")
def trigger_train():
    try:
        # For fine-tuning, use fixed values for model and image size
        base_weights = "/models/best.pt"  # Use existing fine-tuned model
        img_size = "640"  # Fixed to match training data
        
        # Get user-configurable parameters
        epochs = request.form.get("epochs", "10")
        batch = request.form.get("batch", "8")

        # Update configmap with training parameters
        patch_configmap({
            "BASE_WEIGHTS": base_weights,
            "EPOCHS": epochs,
            "IMG_SIZE": img_size,
            "BATCH": batch
        })

        # Create the training job
        create_train_job()
        
        return jsonify({
            "status": "ok", 
            "message": "Training job started successfully",
            "config": {
                "model": "best.pt (fine-tuning)",
                "epochs": epochs,
                "img_size": img_size,
                "batch": batch
            }
        }), 202

    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to start training job: {str(e)}"
        }), 500

# Retrieve job status
@app.get("/job-status/<jobname>")
def job_status(jobname):
    try:
        job = k8s_batch.read_namespaced_job_status(
            name=jobname,
            namespace=NAMESPACE
        )
        
        status = job.status
        return jsonify({
            "active": status.active or 0,
            "succeeded": status.succeeded or 0,
            "failed": status.failed or 0,
            "conditions": [
                {
                    "type": condition.type,
                    "status": condition.status,
                    "reason": condition.reason,
                    "message": condition.message
                } for condition in (status.conditions or [])
            ]
        })
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return jsonify({"error": "Job not found"}), 404
        else:
            return jsonify({"error": f"Failed to get job status: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to get job status: {str(e)}"}), 500

# Get job logs
@app.get("/job-logs/<jobname>")
def job_logs(jobname):
    try:
        # Find the pod for this job
        pods = k8s_v1.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"job-name={jobname}"
        )
        
        if not pods.items:
            return jsonify({"error": "No pods found for job"}), 404
        
        # Get logs from the first pod
        pod_name = pods.items[0].metadata.name
        logs = k8s_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=NAMESPACE
        )
        
        # Parse preprocess results from logs if it's a preprocess job
        results = {"logs": logs}
        if jobname == "preprocess":
            results["preprocess_summary"] = parse_preprocess_logs(logs)
        elif jobname == "train":
            results["training_summary"] = parse_training_logs(logs)
        
        return jsonify(results)
        
    except client.exceptions.ApiException as e:
        return jsonify({"error": f"Failed to get job logs: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to get job logs: {str(e)}"}), 500

def parse_preprocess_logs(logs):
    """Parse preprocess job logs to extract summary statistics."""
    import re
    
    summary = {
        "status": "unknown",
        "splits": {},
        "total_images": 0,
        "total_polygons": 0
    }
    
    try:
        # Look for split results like: [train] images: 8  polygons: 112  pairs: 8
        split_pattern = r'\[(\w+)\] images: (\d+)\s+polygons: (\d+)\s+pairs: (\d+)'
        
        for match in re.finditer(split_pattern, logs):
            split_name = match.group(1)
            images = int(match.group(2))
            polygons = int(match.group(3))
            pairs = int(match.group(4))
            
            summary["splits"][split_name] = {
                "images": images,
                "polygons": polygons,
                "pairs": pairs
            }
            summary["total_images"] += images
            summary["total_polygons"] += polygons
        
        # Check if YOLO config was written successfully
        if "Wrote /data/yolo_xbd/xbd6.yaml" in logs:
            summary["status"] = "success"
        elif "ERROR" in logs.upper() or "FAILED" in logs.upper():
            summary["status"] = "error"
        else:
            summary["status"] = "completed"
            
    except Exception:
        summary["status"] = "parse_error"
    
    return summary

def parse_training_logs(logs):
    """
    Parse YOLO training logs to extract metrics and model information.
    """
    import re
    
    summary = {
        "status": "unknown",
        "epochs_completed": 0,
        "total_epochs": 0,
        "final_metrics": {},
        "best_metrics": {},
        "model_info": {},
        "training_config": {},
        "model_path": None,
        "published_path": None
    }
    
    try:
        lines = logs.split('\n')
        
        # Extract training configuration
        for line in lines:
            # Parse environment config
            env_match = re.search(r'\[env\] model=(.+?), data=(.+?), epochs=(\d+), img=(\d+), batch=(\d+), device=(.+)', line)
            if env_match:
                summary["training_config"] = {
                    "model": env_match.group(1),
                    "data": env_match.group(2), 
                    "total_epochs": int(env_match.group(3)),
                    "image_size": int(env_match.group(4)),
                    "batch_size": int(env_match.group(5)),
                    "device": env_match.group(6)
                }
                summary["total_epochs"] = int(env_match.group(3))
                
            # Parse model info
            if "Model summary:" in line:
                # Look for model parameters in following lines
                for i, next_line in enumerate(lines[lines.index(line):lines.index(line)+10]):
                    param_match = re.search(r'(\d+) parameters', next_line)
                    if param_match:
                        summary["model_info"]["parameters"] = int(param_match.group(1))
                    layer_match = re.search(r'(\d+) layers', next_line)
                    if layer_match:
                        summary["model_info"]["layers"] = int(layer_match.group(1))
            
            # Parse epoch results - look for patterns like:
            # Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
            # 1/10      1.2G      1.234      0.567      0.890        120        640
            epoch_match = re.search(r'^\s*(\d+)/(\d+)\s+[\d.]+G?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line.strip())
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                box_loss = float(epoch_match.group(3))
                cls_loss = float(epoch_match.group(4))
                dfl_loss = float(epoch_match.group(5))
                
                summary["epochs_completed"] = max(summary["epochs_completed"], epoch_num)
                summary["total_epochs"] = total_epochs
                summary["final_metrics"] = {
                    "box_loss": box_loss,
                    "cls_loss": cls_loss,
                    "dfl_loss": dfl_loss,
                    "total_loss": box_loss + cls_loss + dfl_loss
                }
                
            # Parse validation results - look for mAP values
            # val: Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
            # val: all        123       456      0.789      0.654      0.721     0.432
            val_match = re.search(r'val:\s+all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if val_match:
                summary["final_metrics"].update({
                    "val_images": int(val_match.group(1)),
                    "val_instances": int(val_match.group(2)),
                    "precision": float(val_match.group(3)),
                    "recall": float(val_match.group(4)),
                    "mAP50": float(val_match.group(5)),
                    "mAP50_95": float(val_match.group(6))
                })
                
            # Parse results directory
            results_match = re.search(r'\[done\] Results saved at (.+)', line)
            if results_match:
                summary["model_info"]["results_dir"] = results_match.group(1).strip()
                summary["model_path"] = f"{results_match.group(1).strip()}/weights/best.pt"
                
            # Parse published model path
            publish_match = re.search(r'\[publish\] Copied best weight to (.+)', line)
            if publish_match:
                summary["published_path"] = publish_match.group(1).strip()
                
            # Check for completion or errors
            if "[done] Results saved at" in line:
                summary["status"] = "completed"
            elif "RuntimeError" in line or "ERROR" in line.upper() or "FAILED" in line.upper():
                summary["status"] = "error"
            elif "Training complete" in line or "[publish] Copied best weight to" in line:
                summary["status"] = "success"
                
        # Set best metrics same as final if we have final metrics
        if summary["final_metrics"]:
            summary["best_metrics"] = summary["final_metrics"].copy()
            
        # If no explicit status was found but we have metrics, assume completed
        if summary["status"] == "unknown" and summary["epochs_completed"] > 0:
            summary["status"] = "completed"
            
    except Exception as e:
        summary["status"] = "parse_error"
        summary["error"] = str(e)
    
    return summary

# Single inference
@app.post("/infer-file")
def infer_file():
    # Expect both pre- and post-disaster files from the form
    pre_file = request.files.get("pre_disaster")
    post_file = request.files.get("post_disaster")

    if not pre_file or not post_file:
        return jsonify({"error": "Both pre_disaster and post_disaster files are required"}), 400

    # Save temporarily
    tmp_pre = os.path.join("/tmp", pre_file.filename)
    tmp_post = os.path.join("/tmp", post_file.filename)
    pre_file.save(tmp_pre)
    post_file.save(tmp_post)

    # Send both files to the FastAPI inference API
    with open(tmp_pre, "rb") as fd_pre, open(tmp_post, "rb") as fd_post:
        r = requests.post(
            f"{INFER_URL}/predict",
            files={
                "pre_disaster": ("pre_disaster.png", fd_pre, "image/png"),
                "post_disaster": ("post_disaster.png", fd_post, "image/png"),
            },
        )

    if r.status_code != 200:
        return jsonify({"error": r.text}), 500

    return r.json(), 200

# Model status endpoint
@app.get("/infer-status")
def infer_status():
    """Get model deployment status from inference service"""
    try:
        r = requests.get(f"{INFER_URL}/model-status")
        if r.status_code != 200:
            return jsonify({"error": "Failed to get model status"}), 500
        return r.json(), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Batch inference
@app.post("/upload-batch")
def upload_batch():
    # Get uploaded image files
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # Create a temporary zip file
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "batch_input.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in files:
            # Save each file directly into the zip
            # Keep only filename to avoid directory nesting
            zipf.writestr(f.filename, f.read())

    # Send zip to FastAPI batch endpoint
    with open(zip_path, "rb") as fd:
        r = requests.post(
            f"{INFER_URL}/batch_predict",
            files={"zipfile": ("batch_input.zip", fd, "application/zip")}
        )

    if r.status_code != 200:
        return jsonify({"error": r.text}), 500

    return r.json(), 200

# Batch inference with uploaded file
@app.post("/batch-infer")
def batch_infer():
    try:
        data = request.get_json()
        zipfile_path = data.get('zipfile_path')
        
        if not zipfile_path:
            return jsonify({"error": "No zipfile_path provided"}), 400
        
        # Send the uploaded zip file to FastAPI batch endpoint
        with open(zipfile_path, "rb") as fd:
            r = requests.post(
                f"{INFER_URL}/batch_predict",
                files={"zipfile": (os.path.basename(zipfile_path), fd, "application/zip")}
            )

        if r.status_code != 200:
            return jsonify({"error": r.text}), 500

        return r.json(), 200
        
    except FileNotFoundError:
        return jsonify({"error": "Uploaded file not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Batch inference failed: {str(e)}"}), 500

# Proxy endpoint to download files from inference service
@app.get("/files/<filename>")
def download_file(filename):
    try:
        # Forward the request to the inference service
        response = requests.get(f"{INFER_URL}/files/{filename}")
        
        if response.status_code != 200:
            return jsonify({"error": "File not found"}), 404
            
        # Return the file with proper headers
        return response.content, 200, {
            'Content-Type': response.headers.get('Content-Type', 'application/octet-stream'),
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"}), 500