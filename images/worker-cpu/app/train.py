# train.py
import shutil
from ultralytics import YOLO
from ultralytics.data import dataset as dsmod
import torch
import os
from pathlib import Path
import csv
import torch.nn as nn
import cv2
import numpy as np
from datetime import datetime


# --- Custom 6-channel YOLO Dataset ---
class XBD6ChannelDataset(dsmod.YOLODataset):
    def __init__(self, *args, pre_post_pairs=None, image_index=None, split='train', **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_post_pairs = pre_post_pairs or {}
        self.image_index = image_index or {}
        self.split = split

    def infer_pre_path(self, post_path):
        name = Path(post_path).name
        pre_name = self.pre_post_pairs.get(name)
        candidates = ([pre_name] if pre_name else []) + [
            name.replace("post_disaster", "pre_disaster"),
            name.replace("_post_", "_pre_"),
            name.replace("-post", "-pre"),
            name.replace("post", "pre")
        ]
        for cand in candidates:
            if not cand: continue
            p = self.image_index.get(cand)
            if p: return str(p)
            for k, v in self.image_index.items():
                if k.lower() == cand.lower(): return str(v)
        return str(post_path)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        post_img = sample['img']            # (3,H,W)
        post_path = sample['im_file']
        
        # Lookup pre-disaster image
        pre_path = self.infer_pre_path(post_path)
        pre_bgr = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        if pre_bgr is None:
            raise FileNotFoundError(f"Pre-disaster image not found or unreadable: {pre_path}")
        pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)

        H, W = post_img.shape[1:]
        if pre_rgb.shape[:2] != (H, W):
            pre_rgb = cv2.resize(pre_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        pre_tensor = torch.from_numpy(pre_rgb).permute(2, 0, 1).float() / 255.0 # (3,H,W)
        img6 = torch.cat([pre_tensor, post_img], dim=0) # (6,H,W)
        sample['img'] = img6
        return sample


# --- Patch model first Conv2D for 6-channel input ---
def patch_first_conv_to_6ch(model_nn: nn.Module):
    for name, module in model_nn.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                6, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                bias=module.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = module.weight
                new_conv.weight[:, 3:] = module.weight
            # Replace original conv in model
            parent = model_nn
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], new_conv)
            print(f"[patch] Patched conv '{name}' to accept 6-channel input.")
            return
    raise RuntimeError("[patch] No 3-channel Conv2D layer found to patch.")


# --- Utility Functions ---
def load_pairs(csv_path):
    m = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            post, pre = row["post_img"].strip(), row["pre_img"].strip()
            if post and pre:
                m[post] = pre
    return m

def build_image_index(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return {p.name: p for p in images_dir.rglob("*") if p.suffix.lower() in exts}


# --- Main Entrypoint ---
def main():
    # Config from ENV or default
    root = Path(os.getenv("YOLO_DIR", "/data/yolo_xbd"))
    model_path = os.getenv("MODEL_PATH", "/models/best.pt")
    base_weights = os.getenv("BASE_WEIGHTS", "yolo11n-seg.pt")
    data_yaml = os.getenv("DATA_YAML", str(root / "xbd6.yaml"))
    epochs = int(os.getenv("EPOCHS", "10"))
    imgsz = int(os.getenv("IMG_SIZE", "640"))
    batch = int(os.getenv("BATCH", "8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[env] model={model_path}, data={data_yaml}, epochs={epochs}, img={imgsz}, batch={batch}, device={device}")
    
    # Print model info for better UI display
    if Path(model_path).exists():
        print(f"[model] Loading existing model from {model_path}")
    else:
        print(f"[model] Starting with base weights: {base_weights}")
        model_path = base_weights

    # Prepare custom pairing
    train_pairs = load_pairs(root / "train_pairs.csv")
    val_pairs   = load_pairs(root / "val_pairs.csv")
    train_index = build_image_index(root / "images/train")
    val_index   = build_image_index(root / "images/val")

    # Patch dataset globally
    dsmod.YOLODataset = lambda *args, **kwargs: XBD6ChannelDataset(
        *args,
        pre_post_pairs=train_pairs if kwargs.get('split') == 'train' else val_pairs,
        image_index=train_index if kwargs.get('split') == 'train' else val_index,
        **kwargs
    )

    # Load and patch model
    model_file = Path(model_path)
    if model_file.exists():
        print(f"[INFO] Loading model from {model_file}. This will be a fine-tuning task.")
        model = YOLO(str(model_file))
    else:
        print(f"[WARN] {model_file} not found. Falling back to base weights: {base_weights}")
        model = YOLO(base_weights)
    patch_first_conv_to_6ch(model.model)

    # Start training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=42,
        cache=False,
        project="/output",
        name="yolo_xbd_6ch",
        exist_ok=True
    )
    print(f"[done] Results saved at {results.save_dir}")
    
    # Print final metrics for UI parsing
    if hasattr(results, 'results_dict') and results.results_dict:
        metrics = results.results_dict
        print(f"[metrics] Final training metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"[metrics] {key}: {value}")
    
    print(f"[training] Training complete after {epochs} epochs")
    
    best = Path(results.save_dir) / "weights" / "best.pt"
    models_dir = Path(os.getenv("MODELS_DIR", "/models"))
    
    try:
        # Ensure directory structure exists
        current_dir = models_dir / "current"
        canary_dir = models_dir / "canary"
        current_dir.mkdir(parents=True, exist_ok=True)
        canary_dir.mkdir(parents=True, exist_ok=True)
        
        # Model versioning logic for canary deployment
        import time
        import json
        
        # Find current version
        current_version_file = current_dir / "version.json"
        if current_version_file.exists():
            with open(current_version_file) as f:
                current_info = json.load(f)
            current_version = current_info.get("version", 1)
            new_version = current_version + 1
        else:
            current_version = 1
            new_version = 2
            
        print(f"[versioning] Creating canary deployment: v{current_version} -> v{new_version}")
        
        # Save new model to canary directory
        canary_model_path = canary_dir / "best.pt"
        shutil.copy2(best, canary_model_path)
        print(f"[versioning] Saved canary model to {canary_model_path}")
        
        # Extract training metrics
        final_metrics = {}
        if hasattr(results, 'results_dict') and results.results_dict:
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    final_metrics[key] = value
        
        # Create canary version metadata
        canary_version_data = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "canary_start_time": datetime.now().isoformat(),
            "deployment_mode": "canary",
            "traffic_weight": 0.5,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch,
                "image_size": imgsz,
                "device": device
            },
            "metrics": final_metrics,
            "parent_version": current_version
        }
        
        # Save canary metadata
        with open(canary_dir / "version.json", 'w') as f:
            json.dump(canary_version_data, f, indent=2)
        
        print(f"[versioning] Canary deployment active: 50/50 traffic split")
        print(f"[versioning] Auto-promotion scheduled in 3 minutes")
        print(f"[versioning] Canary version: {new_version}")
        
        # Schedule auto-promotion (simplified - in production this would be handled by the inference service)
        print(f"[versioning] Canary deployment setup complete")
        
    except Exception as e:
        print(f"[publish][WARN] Could not publish canary model: {e}")
        # Fallback to old behavior
        pub = models_dir / "best.pt"
        tmp = pub.with_suffix(".pt.tmp")
        shutil.copy2(best, tmp)
        os.replace(tmp, pub)
        print(f"[publish] Fallback: Copied best weight to {pub}")


if __name__ == "__main__":
    main()