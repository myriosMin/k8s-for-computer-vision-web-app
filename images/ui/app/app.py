from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, requests

INFER_URL = os.getenv("INFER_URL", "http://infer:9000")
NAMESPACE = os.getenv("NAMESPACE", "xview")

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.get("/")
def index():
    # Page 1: batch upload, config, metrics
    return render_template("index.html")

@app.get("/inference")
def inference_page():
    # Page 2: single-file inference
    return render_template("inference.html")

@app.post("/upload-batch")
def upload_batch():
    # Save uploaded files into the mounted PVC path shared by jobs
    up_dir = os.getenv("DATASET_MOUNT", "/datasets/raw")
    os.makedirs(up_dir, exist_ok=True)
    files = request.files.getlist("files")
    for f in files:
        f.save(os.path.join(up_dir, secure_filename(f.filename)))
    return redirect(url_for("index"))

@app.post("/trigger-preprocess")
def trigger_preprocess():
    # In dev, we rely on Kubernetes Job already mounted to read /datasets/raw
    # You can add parameters here if needed via a small control plane or k8s client.
    return jsonify({"status": "ok", "message": "Preprocess Job will run via kubectl apply (see README)."}), 202

@app.post("/trigger-train")
def trigger_train():
    # Same—training is kicked by kubectl job (see README). You can pass config via ConfigMap/Secret.
    return jsonify({"status":"ok","message":"Train Job will run via kubectl apply (see README)."}), 202

@app.post("/infer-file")
def infer_file():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded"}), 400
    tmp = "/tmp/" + f.filename
    f.save(tmp)
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        f.save(tmpfile.name)
        tmpfile_path = tmpfile.name
    with open(tmpfile_path, "rb") as fd:
        r = requests.post(f"{INFER_URL}/predict", files={"file": fd})
    if r.status_code != 200:
        return jsonify({"error": r.text}), 500
    return r.json(), 200