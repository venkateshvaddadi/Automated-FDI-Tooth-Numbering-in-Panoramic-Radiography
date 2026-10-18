"""
eval_cascade_benchmarks.py
--------------------------
Evaluates Cascade R-CNN (100 Epochs - checkpoint_ep100.pt and best.pt) on:
1. Internal Institutional Validation Dataset (datasets/aneesa/images/val) - 102 images
2. External Multi-Center DENTEX Benchmark (dentex_yolo) - 634 images / 18,095 teeth
Across standard 32 FDI teeth using standardized micro Precision, Recall, and F1 (IoU >= 0.50, Conf >= 0.50).
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_DIR = Path(__file__).resolve().parents[1]
INTERNAL_IMG_DIR = BASE_DIR / "datasets" / "aneesa" / "images" / "val"
INTERNAL_LBL_DIR = BASE_DIR / "datasets" / "aneesa" / "labels" / "val"

DENTEX_IMG_DIR = BASE_DIR.parent / "yolov4" / "yolo11" / "external" / "dentex_yolo" / "images"
DENTEX_LBL_DIR = BASE_DIR.parent / "yolov4" / "yolo11" / "external" / "dentex_yolo" / "labels"


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


def evaluate_cascade(model, img_dir, lbl_dir, dataset_name="Dataset", conf_thresh=0.50, iou_thresh=0.50):
    img_paths = sorted(Path(img_dir).glob("*.*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

    tp, fp, fn = 0, 0, 0
    total_gt = 0
    total_det = 0

    print(f"\n[Evaluating {dataset_name}] Total images: {len(img_paths)} (Conf >= {conf_thresh}, IoU >= {iou_thresh})")

    with torch.no_grad():
        for idx, img_path in enumerate(img_paths):
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_res = cv2.resize(im_rgb, (1280, 1280))
            t = torch.from_numpy(im_res).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
            preds = model(t)[0]

            lbl_file = Path(lbl_dir) / f"{img_path.stem}.txt"
            gt_boxes, gt_classes = [], []
            if lbl_file.exists():
                for line in lbl_file.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, cx, cy, bw, bh = (float(x) for x in parts)
                        c = int(c)
                        if c < 32:
                            x1 = (cx - bw / 2.0) * 1280
                            y1 = (cy - bh / 2.0) * 1280
                            x2 = (cx + bw / 2.0) * 1280
                            y2 = (cy + bh / 2.0) * 1280
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_classes.append(c + 1)

            total_gt += len(gt_boxes)
            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            gt_classes = np.array(gt_classes) if gt_classes else np.zeros((0,))

            keep = preds["scores"].cpu().numpy() >= conf_thresh
            p_boxes = preds["boxes"][keep].cpu().numpy()
            p_labels = preds["labels"][keep].cpu().numpy()
            valid = (p_labels >= 1) & (p_labels <= 32)
            p_boxes = p_boxes[valid]
            p_labels = p_labels[valid]
            total_det += len(p_boxes)

            if len(gt_boxes) == 0:
                fp += len(p_boxes)
                continue
            if len(p_boxes) == 0:
                fn += len(gt_boxes)
                continue

            iou_mat = box_iou(p_boxes, gt_boxes)
            matched_gt = set()
            for i in range(len(p_boxes)):
                best_iou, best_j = 0.0, -1
                for j in range(len(gt_boxes)):
                    if j in matched_gt:
                        continue
                    if p_labels[i] == gt_classes[j] and iou_mat[i, j] >= iou_thresh:
                        if iou_mat[i, j] > best_iou:
                            best_iou = iou_mat[i, j]
                            best_j = j
                if best_j >= 0:
                    matched_gt.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(img_paths):
                print(f"  Processed {idx + 1}/{len(img_paths)} images...", flush=True)

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


def main():
    checkpoints = [
        ("Cascade-RCNN (Best Val Loss)", BASE_DIR / "runs" / "classic" / "Cascade-RCNN" / "best.pt"),
        ("Cascade-RCNN (Epoch 100)", BASE_DIR / "runs" / "classic" / "Cascade-RCNN" / "checkpoint_ep100.pt"),
    ]

    for name, ckpt in checkpoints:
        if not ckpt.exists():
            continue
        print("=" * 65)
        print(f"  Evaluating {name}: {ckpt.name}")
        print("=" * 65)

        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            min_size=1280,
            max_size=1280,
            box_batch_size_per_image=512,
            box_positive_fraction=0.33,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 49)

        model.load_state_dict(torch.load(ckpt, map_location="cuda:0"))
        model.cuda()
        model.eval()

        print(f"--- 1. Institutional Validation Set (102 images) ---")
        evaluate_cascade(model, INTERNAL_IMG_DIR, INTERNAL_LBL_DIR, f"{name} - Internal Val", conf_thresh=0.50)

        print(f"--- 2. External DENTEX Benchmark (634 images) ---")
        evaluate_cascade(model, DENTEX_IMG_DIR, DENTEX_LBL_DIR, f"{name} - DENTEX Full (634 imgs)", conf_thresh=0.50)


if __name__ == "__main__":
    main()
