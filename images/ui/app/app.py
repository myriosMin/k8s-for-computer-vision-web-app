from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, requests
import tempfile
import zipfile
import subprocess, json
import logging
from minio import Minio

logging.basicConfig(level=logging.DEBUG)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-service:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "datasets"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

INFER_URL = os.getenv("INFER_URL", "http://infer:9000")
NAMESPACE = os.getenv("NAMESPACE", "xview")
CONFIGMAP_NAME = "app-config"

def patch_configmap(overrides: dict):
    """
    Generic helper to merge a dict of key→value into the app-config ConfigMap.
    values must be strings.
    """
    # ensure all values are strings
    data = {k: str(v) for k, v in overrides.items()}
    patch = {"data": data}
    subprocess.run([
        "kubectl", "patch", "configmap", CONFIGMAP_NAME,
        "-n", NAMESPACE,
        "--type=merge",
        "-p", json.dumps(patch)
    ], check=True)

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

# upload file to Minio database
@app.post("/upload-dataset")
def upload_dataset():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    object_name = file.filename
    minio_client.put_object(
        BUCKET_NAME,
        object_name,
        file.stream,
        length=-1,
        part_size=10*1024*1024,
    )

    return jsonify({"status": "ok", "bucket": BUCKET_NAME, "object": object_name}), 200

# Trigger preprocessing job
# @app.post("/trigger-preprocess")
# def trigger_preprocess():
#     try:
#         dataset_file = request.form.get("dataset", "aiad_data.zip")
#         print(f"[DEBUG] Dataset file: {dataset_file}")
#         img_size      = request.form.get("img_size", "1024")
#         print(f"[DEBUG] Image size: {img_size}")
#         patch_configmap({
#             "RAW_ZIP_PATH": dataset_file,
#             "IMG_SIZE":      img_size
#         })
#         print(f"[DEBUG] Patching ConfigMap with: {patch_configmap}")
#         # launch job

#         subprocess.run(["make", "job-preprocess"], check=True)
        
#         return jsonify({"status": "ok", "message": "Preprocess job started"}), 202
#         print(f"[DEBUG] Subprocess output: {result.stdout}")

#     except subprocess.CalledProcessError as e:
#         return jsonify({"status": "error", "message": "Triggering preprocess job failed." + str(e)}), 500

@app.post("/trigger-preprocess")
def trigger_preprocess():
    try:
        dataset_file = request.form.get("dataset", "s3://datasets/aiad_data.zip")
        logging.debug(f"Dataset file: {dataset_file}")
        
        img_size = request.form.get("img_size", "1024")
        logging.debug(f"Image size: {img_size}")

        patch_data = {
            "RAW_ZIP_PATH": dataset_file,
            "IMG_SIZE": img_size
        }
        logging.debug(f"Patching ConfigMap with: {patch_data}")
        patch_configmap(patch_data)

        logging.debug("Launching preprocess job with Makefile...")
        result = subprocess.run(
            ["make", "job-preprocess"],
            check=True,
            capture_output=True,
            text=True
        )
        logging.debug(f"Subprocess stdout: {result.stdout}")
        logging.debug(f"Subprocess stderr: {result.stderr}")
        
        return jsonify({"status": "ok", "message": "Preprocess job started"}), 202

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed: {e.stderr}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Triggering preprocess job failed. stderr={e.stderr}, stdout={e.stdout}"
        }), 500

    except Exception as e:
        logging.exception("Unexpected error in /trigger-preprocess")
        return jsonify({"status": "error", "message": str(e)}), 500

# Trigger training job

@app.post("/trigger-train")
def trigger_train():
    try:
        base_weights = request.form.get("base_weights", "yolo11n-seg.pt")
        epochs       = request.form.get("epochs",       "10")
        img_size     = request.form.get("img_size",     "1024")
        batch        = request.form.get("batch",        "8")

        patch_configmap({
            "BASE_WEIGHTS": base_weights,
            "EPOCHS":       epochs,
            "IMG_SIZE":     img_size,
            "BATCH":        batch
        })

        # launch job
        subprocess.run(["make", "job-train"], check=True)
        return jsonify({"status": "ok", "message": "Train job started"}), 202

    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": "Triggering train job failed." + str(e)}), 500

# Retrieve job status
@app.get("/job-status/<jobname>")
def job_status(jobname):
    try:
        output = subprocess.check_output(
            ["kubectl", "-n", NAMESPACE, "get", "job", jobname, "-o", "json"]
        )
        import json
        status = json.loads(output)["status"]
        return jsonify({
            "active": status.get("active", 0),
            "succeeded": status.get("succeeded", 0),
            "failed": status.get("failed", 0),
        })
    except subprocess.CalledProcessError:
        return jsonify({"error": "Job not found"}), 404

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