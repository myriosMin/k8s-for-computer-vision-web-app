import os, torch

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "RAW_DIR": os.getenv("RAW_DIR", "/datasets/raw"),
    "YOLO_DIR": os.getenv("YOLO_DIR", "/datasets/yolo"),
}
MODELS_DIR = os.getenv("MODELS_DIR", "/models")