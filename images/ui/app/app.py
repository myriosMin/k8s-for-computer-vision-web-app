from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, requests
import tempfile
import zipfile
import subprocess, json

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
        img_size      = request.form.get("img_size", "640")

        patch_configmap({
            "RAW_ZIP_PATH": dataset_file,
            "IMG_SIZE":      img_size
        })

        # launch job
        subprocess.run(["make", "job-preprocess"], check=True)
        return jsonify({"status": "ok", "message": "Preprocess job started"}), 202

    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": "Triggering preprocess job failed." + str(e)}), 500

# Trigger training job

@app.post("/trigger-train")
def trigger_train():
    try:
        base_weights = request.form.get("base_weights", "yolo11n-seg.pt")
        epochs       = request.form.get("epochs",       "10")
        img_size     = request.form.get("img_size",     "640")
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