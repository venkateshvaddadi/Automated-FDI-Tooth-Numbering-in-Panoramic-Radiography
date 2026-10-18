"""
eval_yolonas_full.py
---------------------
Evaluates YOLO-NAS-M (100 Epochs - ckpt_best.pth and average_model.pth) on:
1. Institutional Validation Dataset (102 images)
2. External Multi-Center DENTEX Benchmark (634 images / 18,095 teeth)
Using standardized micro Precision, Recall, and F1 (IoU >= 0.50).
"""

import sys
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch

_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

try:
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
YAML_PATH = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
INTERNAL_IMG_DIR = BASE_DIR / "datasets" / "aneesa" / "images" / "val"
INTERNAL_LBL_DIR = BASE_DIR / "datasets" / "aneesa" / "labels" / "val"

DENTEX_IMG_DIR = BASE_DIR.parent / "yolov4" / "yolo11" / "external" / "dentex_yolo" / "images"
DENTEX_LBL_DIR = BASE_DIR.parent / "yolov4" / "yolo11" / "external" / "dentex_yolo" / "labels"

CKPT_PATH = BASE_DIR / "runs" / "yolonas" / "yolo_nas_m" / "RUN_20261017_051359_176479" / "ckpt_best.pth"
if not CKPT_PATH.exists():
    print(f"Error: {CKPT_PATH} does not exist.")
    sys.exit(1)

with open(YAML_PATH) as f:
    data_cfg = yaml.safe_load(f)
NAMES = data_cfg["names"]
BASE_NAMES = {i: str(n).split("-")[0] for i, n in NAMES.items()}
STD_TEETH = {str(t) for t in [11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28, 31,32,33,34,35,36,37,38, 41,42,43,44,45,46,47,48]}

print("=" * 65)
print(f"Loading YOLO-NAS-M Checkpoint: {CKPT_PATH}")
print("=" * 65)

from super_gradients.training import models
net = models.get("yolo_nas_m", num_classes=48, checkpoint_path=str(CKPT_PATH))
net.eval().cuda()

def box_iou(b1, b2):
    x1 = np.maximum(b1[:, 0:1], b2[:, 0:1].T)
    y1 = np.maximum(b1[:, 1:2], b2[:, 1:2].T)
    x2 = np.minimum(b1[:, 2:3], b2[:, 2:3].T)
    y2 = np.minimum(b1[:, 3:4], b2[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / np.maximum(union, 1e-6)

def run_eval(img_dir, lbl_dir, dataset_name="Dataset", conf_thresh=0.25, iou_thresh=0.50):
    img_files = sorted(Path(img_dir).glob("*.*"))
    img_files = [p for p in img_files if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    
    tp, fp, fn = 0, 0, 0
    total_gt = 0
    total_det = 0

    print(f"\n[Evaluating {dataset_name}] Total images: {len(img_files)} (Conf >= {conf_thresh}, IoU >= {iou_thresh})")

    for idx, img_p in enumerate(img_files):
        lbl_p = Path(lbl_dir) / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
            
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        # Ground truth
        gt_boxes = []
        gt_classes = []
        for line in lbl_p.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                c, cx, cy, bw, bh = (float(x) for x in parts)
                c_id = int(c)
                c_name = BASE_NAMES.get(c_id, str(c_id))
                if c_name in STD_TEETH:
                    gx1 = (cx - bw/2) * w
                    gy1 = (cy - bh/2) * h
                    gx2 = (cx + bw/2) * w
                    gy2 = (cy + bh/2) * h
                    gt_boxes.append([gx1, gy1, gx2, gy2])
                    gt_classes.append(c_name)

        total_gt += len(gt_boxes)

        # Inference
        res = net.predict(str(img_p), conf=conf_thresh, fuse_model=False)
        pred_labels = res.prediction.labels
        pred_scores = res.prediction.confidence
        pred_boxes = res.prediction.bboxes_xyxy

        det_boxes = []
        det_classes = []
        for b, c_id, s in zip(pred_boxes, pred_labels, pred_scores):
            c_name = BASE_NAMES.get(int(c_id), str(int(c_id)))
            if c_name in STD_TEETH:
                det_boxes.append(b)
                det_classes.append(c_name)

        total_det += len(det_boxes)

        if len(gt_boxes) == 0:
            fp += len(det_boxes)
            continue
        if len(det_boxes) == 0:
            fn += len(gt_boxes)
            continue

        gt_boxes = np.array(gt_boxes)
        det_boxes = np.array(det_boxes)
        iou_mat = box_iou(det_boxes, gt_boxes)

        matched_gt = set()
        for i in range(len(det_boxes)):
            best_iou, best_j = 0.0, -1
            for j in range(len(gt_boxes)):
                if j in matched_gt: continue
                if det_classes[i] == gt_classes[j] and iou_mat[i, j] >= iou_thresh:
                    if iou_mat[i, j] > best_iou:
                        best_iou = iou_mat[i, j]
                        best_j = j
            if best_j >= 0:
                matched_gt.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += (len(gt_boxes) - len(matched_gt))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(img_files):
            print(f"  Processed {idx + 1}/{len(img_files)} images...", flush=True)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    
    print(f"  => {dataset_name} Results:")
    print(f"     Ground Truth: {total_gt} | Predictions: {total_det}")
    print(f"     TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"     Precision : {prec * 100:.2f}%")
    print(f"     Recall    : {rec * 100:.2f}%")
    print(f"     F1-Score  : {f1 * 100:.2f}%\n")
    return {"P": prec, "R": rec, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "GT": total_gt, "Det": total_det}

if __name__ == "__main__":
    print("------------------------------------------------------------------")
    print(" 1. Institutional Validation Set (datasets/aneesa/images/val)")
    print("------------------------------------------------------------------")
    int_25 = run_eval(INTERNAL_IMG_DIR, INTERNAL_LBL_DIR, "Internal Validation (Conf=0.25)", conf_thresh=0.25)
    int_50 = run_eval(INTERNAL_IMG_DIR, INTERNAL_LBL_DIR, "Internal Validation (Conf=0.50)", conf_thresh=0.50)

    print("------------------------------------------------------------------")
    print(" 2. External DENTEX Benchmark (dentex_yolo)")
    print("------------------------------------------------------------------")
    dentex_25 = run_eval(DENTEX_IMG_DIR, DENTEX_LBL_DIR, "External DENTEX (Conf=0.25)", conf_thresh=0.25)
    dentex_50 = run_eval(DENTEX_IMG_DIR, DENTEX_LBL_DIR, "External DENTEX (Conf=0.50)", conf_thresh=0.50)
