import os, torch

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "RAW_ZIP_PATH": os.getenv("RAW_ZIP_PATH", "/data/aiad_data.zip"),
    "YOLO_DIR": os.getenv("YOLO_DIR", "/data/yolo_xbd"),
}
MODELS_DIR = os.getenv("MODELS_DIR", "/models")