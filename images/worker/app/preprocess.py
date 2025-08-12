import os, json, re, shutil, pathlib
from pathlib import Path
from PIL import Image
from .common import DATASETS

# ==== CONFIG ====
SPLITS = ["train", "test", "tier3"]
USE_ONLY_POST = True
MIN_POINTS = 3
MIN_PERIM_PX = 8

# Directories from env
RAW_ROOT = Path(DATASETS["RAW_DIR"])
YOLO_ROOT = Path(DATASETS["YOLO_DIR"])

# Setup YOLO subdirs
def setup_dirs():
    for split in SPLITS:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] Prepared YOLO dirs at {YOLO_ROOT}")

# WKT polygon parser
wkt_poly_re = re.compile(r"POLYGON\s*\(\(\s*(.*?)\s*\)\)", re.IGNORECASE | re.DOTALL)
def parse_wkt_polygon(wkt: str):
    m = wkt_poly_re.search(wkt)
    if not m:
        return []
    pts = []
    for pair in m.group(1).split(","):
        a = pair.strip().split()
        if len(a) != 2:
            continue
        pts.append((float(a[0]), float(a[1])))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts

def poly_perimeter(px_pts):
    return sum(((px_pts[i][0] - px_pts[(i+1)%len(px_pts)][0])**2 + 
                (px_pts[i][1] - px_pts[(i+1)%len(px_pts)][1])**2) ** 0.5 
                for i in range(len(px_pts)))

# Actual preprocessing logic
def convert_split(split: str):
    split_root = RAW_ROOT / split
    lbl_dir = split_root / "labels"
    img_dir = split_root / "images"
    out_img_dir = YOLO_ROOT / "images" / split
    out_lbl_dir = YOLO_ROOT / "labels" / split

    json_files = sorted(lbl_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] no JSON in {lbl_dir}")
        return

    for jpath in json_files:
        with open(jpath, "r") as f:
            data = json.load(f)

        feats = data.get("features", {}).get("xy", []) or []
        meta = data.get("metadata", {})
        img_name = meta.get("img_name") or ""
        W = int(meta.get("width", meta.get("original_width", 1024)))
        H = int(meta.get("height", meta.get("original_height", 1024)))

        if USE_ONLY_POST and "post" not in img_name.lower():
            continue

        src_img = img_dir / img_name
        if not src_img.exists():
            matches = list(img_dir.rglob(img_name))
            if matches:
                src_img = matches[0]
            else:
                print(f"[WARN] missing image for {jpath.name}: {img_name}")
                continue

        dst_img = out_img_dir / src_img.name
        if not dst_img.exists():
            try: os.link(src_img, dst_img)
            except Exception: shutil.copy2(src_img, dst_img)

        yolo_lines = []
        for obj in feats:
            if obj.get("properties", {}).get("feature_type") != "building":
                continue

            pts = parse_wkt_polygon(obj.get("wkt", ""))
            if len(pts) < MIN_POINTS or poly_perimeter(pts) < MIN_PERIM_PX:
                continue

            norm_pts = []
            for x, y in pts:
                xn = max(0.0, min(1.0, x / W))
                yn = max(0.0, min(1.0, y / H))
                norm_pts.extend([xn, yn])

            yolo_lines.append("0 " + " ".join(f"{v:.6f}" for v in norm_pts))

        out_txt = out_lbl_dir / (dst_img.stem + ".txt")
        out_txt.write_text("\n".join(yolo_lines))

        if not yolo_lines:
            print(f"[INFO] no buildings in {jpath.name}")

    print(f"[INFO] Finished processing {split} split")

# Entry point
def main():
    setup_dirs()
    for s in SPLITS:
        convert_split(s)

if __name__ == "__main__":
    main()