from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.utils.plotting import Colors

import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps as cm
from PIL import Image
import cv2


# ========== Configuration ==========
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
IMG_SIZE = int(os.getenv("IMG_SIZE", 1024))


# ========== Patching ==========
def patch_first_conv_to_6ch(model_nn: nn.Module):
    for name, module in model_nn.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                6, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding, bias=module.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = module.weight
                new_conv.weight[:, 3:] = module.weight
            parent = model_nn
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], new_conv)
            print(f"[INFO] Patched conv '{name}' for 6-channel input.")
            return
    raise RuntimeError("No 3-channel Conv2d found to patch.")


# ========== Image Handling ==========
def stack_pre_post(pre_bytes, post_bytes, size=1024):
    pre_arr = np.asarray(Image.open(io.BytesIO(pre_bytes)).convert("RGB"))
    post_arr = np.asarray(Image.open(io.BytesIO(post_bytes)).convert("RGB"))

    pre = cv2.resize(pre_arr, (size, size))
    post = cv2.resize(post_arr, (size, size))

    pre_t = torch.from_numpy(pre).permute(2, 0, 1).float() / 255.0
    post_t = torch.from_numpy(post).permute(2, 0, 1).float() / 255.0

    img6 = torch.cat([pre_t, post_t], dim=0)  # (6, H, W)
    return img6.unsqueeze(0)  # (1, 6, H, W)


# ========== Overlay Plotting ==========
def get_overlay_plot(res, class_names, img_size, post_bytes):
    cmap = cm.get_cmap("tab10")
    custom_colors = {cls: cmap(cls / 10)[:3] for cls in range(10)}

    img_rgb = np.asarray(Image.open(io.BytesIO(post_bytes)).convert("RGB"))
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(img_rgb)
    ax.axis("off")

    boxes = res.boxes
    if boxes is not None and boxes.data is not None and boxes.data.shape[0] > 0:
        for box in boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)
            color = custom_colors.get(cls, (1.0, 0.0, 0.0))
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, linewidth=2, edgecolor=color)
            ax.add_patch(rect)

        unique_classes = sorted(set([int(c.item()) for c in boxes.cls]))
        handles = []
        for cls in unique_classes:
            color = custom_colors.get(cls, (1.0, 0.0, 0.0))
            patch = mpatches.Patch(facecolor='none', edgecolor=color,
                                   linewidth=2, label=class_names[cls])
            handles.append(patch)
        ax.legend(handles=handles, loc="upper right", title="Classes", framealpha=0.5)

    fig.canvas.draw()

    try:
        width, height = fig.canvas.get_width_height()
        plot_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_np = plot_np.reshape((height, width, 3))
        plt.close(fig)
        return Image.fromarray(plot_np)
    except Exception as e:
        # Fallback: save to PNG in-memory
        buf_img = io.BytesIO()
        fig.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        img = Image.open(buf_img).convert("RGB")
        buf_img.close()
        plt.close(fig)
        return img


# ========== FastAPI with Lifespan ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    model = YOLO(MODEL_PATH)
    patch_first_conv_to_6ch(model.model)
    model.model.eval()
    app.state.model = model
    print("[INFO] YOLO model loaded and patched.")
    yield
    print("[INFO] App shutdown.")


app = FastAPI(lifespan=lifespan)


# ========== Inference Endpoint ==========
@app.post("/predict")
async def predict(
    request: Request,
    pre_disaster: UploadFile = File(...),
    post_disaster: UploadFile = File(...)
):
    model = request.app.state.model

    # Read uploaded images
    pre_bytes = await pre_disaster.read()
    post_bytes = await post_disaster.read()

    # Preprocess and predict
    img6 = stack_pre_post(pre_bytes, post_bytes, size=IMG_SIZE)
    device = next(model.model.parameters()).device
    img6 = img6.to(device)

    with torch.no_grad():
        results = model.predict(source=img6, device=device, verbose=False)
    res = results[0]

    # Generate base64 plot
    plot_img = get_overlay_plot(res, model.names, IMG_SIZE, post_bytes)
    buf = io.BytesIO()
    plot_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Extract detections
    dets = []
    if hasattr(res, "boxes") and res.boxes is not None:
        for b in res.boxes:
            dets.append({
                "cls": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()]
            })

    return JSONResponse({
        "detections": dets,
        "overlay_png_base64": encoded
    })