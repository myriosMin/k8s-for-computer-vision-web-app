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
    imgsz = int(os.getenv("IMG_SIZE", "1024"))
    batch = int(os.getenv("BATCH", "8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[env] model={model_path}, data={data_yaml}, epochs={epochs}, img={imgsz}, batch={batch}, device={device}")

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
    
    best = Path(results.save_dir) / "weights" / "best.pt"
    pub = Path(os.getenv("MODELS_DIR", "/models")) / "best.pt"

    try:
        pub.parent.mkdir(parents=True, exist_ok=True)
        # Prefer atomic replace to avoid partial reads by infer pods
        tmp = pub.with_suffix(".pt.tmp")
        shutil.copy2(best, tmp)
        os.replace(tmp, pub)
        print(f"[publish] Copied best weight to {pub}")
    except Exception as e:
        print(f"[publish][WARN] Could not publish best.pt to {pub}: {e}")


if __name__ == "__main__":
    main()