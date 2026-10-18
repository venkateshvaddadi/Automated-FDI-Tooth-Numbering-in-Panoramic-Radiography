"""
prepare_dataset.py
------------------
Copies images and labels from the original darknet `data/` folder
into the YOLOv11 dataset structure, honouring the existing train/val split.

Source layout (darknet):
    darknet/
        data/1.jpg  data/1.txt  ...
        train.txt   val.txt

Target layout (Ultralytics):
    yolov11_training/datasets/aneesa/
        images/train/   images/val/
        labels/train/   labels/val/
"""

import os
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]          # …/aneesa/
DARKNET_DIR = BASE_DIR / "yolov4" / "darknet"
DATA_DIR    = DARKNET_DIR / "data"

DATASET_DIR = BASE_DIR / "yolov11_training" / "datasets" / "aneesa"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL   = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL   = DATASET_DIR / "labels" / "val"

# ── Read split lists ──────────────────────────────────────────────────────────
def read_list(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]

train_files = read_list(DARKNET_DIR / "train.txt")   # e.g. "data/430.jpg"
val_files   = read_list(DARKNET_DIR / "val.txt")

print(f"Train images : {len(train_files)}")
print(f"Val   images : {len(val_files)}")

# ── Copy helper ───────────────────────────────────────────────────────────────
def copy_split(file_list, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    copied, missing_lbl = 0, 0
    for rel_path in file_list:
        # rel_path looks like "data/430.jpg"
        img_src = DARKNET_DIR / rel_path
        if not img_src.exists():
            print(f"  [WARN] Image not found: {img_src}")
            continue
        stem     = img_src.stem           # "430"
        lbl_src  = DATA_DIR / f"{stem}.txt"

        # Copy image
        shutil.copy2(img_src, dst_img_dir / img_src.name)

        # Copy label
        if lbl_src.exists():
            shutil.copy2(lbl_src, dst_lbl_dir / lbl_src.name)
        else:
            print(f"  [WARN] Label not found: {lbl_src}")
            missing_lbl += 1
        copied += 1
    print(f"  Copied {copied} images, {missing_lbl} missing labels")

print("\nCopying train split …")
copy_split(train_files, IMAGES_TRAIN, LABELS_TRAIN)

print("Copying val split …")
copy_split(val_files, IMAGES_VAL, LABELS_VAL)

# ── Verification ─────────────────────────────────────────────────────────────
train_imgs = list(IMAGES_TRAIN.glob("*.jpg"))
val_imgs   = list(IMAGES_VAL.glob("*.jpg"))
train_lbls = list(LABELS_TRAIN.glob("*.txt"))
val_lbls   = list(LABELS_VAL.glob("*.txt"))

print("\n── Verification ──────────────────────────────────────────────────")
print(f"  images/train : {len(train_imgs)}")
print(f"  images/val   : {len(val_imgs)}")
print(f"  labels/train : {len(train_lbls)}")
print(f"  labels/val   : {len(val_lbls)}")
print("\n✅  Dataset ready!")
