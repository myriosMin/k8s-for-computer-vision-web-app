import json, re, os, shutil
from pathlib import Path
from .common import DATASETS
import zipfile, tarfile, tempfile

CLS_MAP = {"no-damage":0, "minor-damage":1, "major-damage":2, "destroyed":3}
WKT_RE = re.compile(r"POLYGON\s*\(\(\s*(.*?)\s*\)\)", re.IGNORECASE | re.DOTALL)

def parse_wkt_polygon(wkt: str):
    m = WKT_RE.search(wkt or "")
    if not m: return []
    pts = []
    for pair in m.group(1).split(","):
        a = pair.strip().split()
        if len(a) != 2: continue
        x, y = float(a[0]), float(a[1])
        pts.append((x, y))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts

def infer_pre_name(post_name: str):
    cands = [
        post_name.replace("post_disaster","pre_disaster"),
        post_name.replace("_post_", "_pre_"),
        post_name.replace("-post", "-pre"),
        post_name.replace("post", "pre"),
    ]
    return cands

def convert_split(xbd_split_root: Path, out_root: Path, split_name: str):
    img_dir = xbd_split_root / "images"
    lbl_dir = xbd_split_root / "labels"
    out_img = out_root / "images" / split_name
    out_lab = out_root / "labels" / split_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    # Build a quick index of all images by basename for pre/post matching
    IMG_INDEX = {p.name: p for p in img_dir.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}}

    pairs_csv = out_root / f"{split_name}_pairs.csv"
    pairs_f = pairs_csv.open("w")
    pairs_f.write("post_img,pre_img\n")

    jsons = sorted(lbl_dir.glob("*.json"))
    n_imgs = n_poly = 0
    for jp in jsons:
        data = json.loads(jp.read_text())
        feats_xy = (data.get("features", {}) or {}).get("xy", []) or []
        meta = data.get("metadata", {}) or {}
        post_name = meta.get("img_name", "")
        if not post_name:
            continue
        post_path = IMG_INDEX.get(post_name)
        if not post_path:
            # try slow fallback
            for k,v in IMG_INDEX.items():
                if k.lower() == post_name.lower():
                    post_path = v; break
        if not post_path:
            continue
        
        # find pre counterpart
        pre_path = None
        for c in infer_pre_name(post_name):
            pre_path = IMG_INDEX.get(c)
            if pre_path: break
            for k,v in IMG_INDEX.items():
                if k.lower() == c.lower():
                    pre_path = v; break
            if pre_path: break

        W = int(meta.get("width", meta.get("original_width", 1024)))
        H = int(meta.get("height", meta.get("original_height", 1024)))

        # write YOLO polygons with damage class id
        lines = []
        for obj in feats_xy:
            props = obj.get("properties", {}) or {}
            if props.get("feature_type") != "building":
                continue
            subtype = props.get("subtype")
            if subtype not in CLS_MAP:
                continue
            cls_id = CLS_MAP[subtype]
            pts = parse_wkt_polygon(obj.get("wkt", ""))
            if len(pts) < 3:
                continue
            # normalize polygon
            xy = []
            for x,y in pts:
                xn = max(0.0, min(1.0, x / W))
                yn = max(0.0, min(1.0, y / H))
                xy.extend([xn, yn])
            lines.append(str(cls_id) + " " + " ".join(f"{v:.6f}" for v in xy))
            n_poly += 1

        # copy image to yolo structure
        dst_img = out_img / post_path.name
        if not dst_img.exists():
            try:
                os.link(post_path, dst_img)
            except Exception:
                shutil.copy2(post_path, dst_img)

        # write label txt
        (out_lab / (dst_img.stem + ".txt")).write_text("\n".join(lines))
        n_imgs += 1

        # write pair if pre exists
        if pre_path:
            pairs_f.write(f"{dst_img.name},{pre_path.name}\n")

    pairs_f.close()
    print(f"[{split_name}] images: {n_imgs}  polygons: {n_poly}  pairs: {sum(1 for _ in open(pairs_csv))-1}")

def extract_archive_to_temp(archive_path: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="xbd_preprocess_"))
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tar":
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(tmp_dir)
    else:
        raise ValueError(f"[ERROR] Unsupported archive type: {archive_path.name}")

    print(f"[INFO] Extracted archive to: {tmp_dir}")

    # check if there's a single subfolder
    contents = list(tmp_dir.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        print(f"[INFO] Zip contains a root folder: {contents[0].name}, using it")
        return contents[0]
    
    return tmp_dir

def is_flat_xbd_structure(path: Path) -> bool:
    return (path / "images").exists() and (path / "labels").exists()

def main():
    input_path = Path(DATASETS["RAW_ZIP_PATH"])
    out_root = Path(DATASETS["YOLO_DIR"])

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        print(f"[INFO] Processing archive: {input_path.name}")
        extracted = extract_archive_to_temp(input_path)
    else:
        print(f"[INFO] Processing folder: {input_path}")
        extracted = input_path

    # Detect structure
    if is_flat_xbd_structure(extracted):
        print("[INFO] Detected flat structure")
        convert_split(extracted, out_root, "train")
    else:
        print("[INFO] Detected split subfolders")
        for sub in extracted.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if "train" in name:
                convert_split(sub, out_root, "train")
            elif any(n in name for n in ["tier3", "val", "valid", "eval"]):
                convert_split(sub, out_root, "val")
            elif "test" in name:
                convert_split(sub, out_root, "test")
            else:
                print(f"[INFO] Unrecognized folder '{name}' — defaulting to train")
                convert_split(sub, out_root, "train")

    # Write YOLO-compatible dataset yaml
    yaml = f"""path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: building_no
  1: building_minor
  2: building_major
  3: building_destroyed
"""
    (out_root / "xbd6.yaml").write_text(yaml)
    print("Wrote", out_root / "xbd6.yaml")

if __name__ == "__main__":
    main()