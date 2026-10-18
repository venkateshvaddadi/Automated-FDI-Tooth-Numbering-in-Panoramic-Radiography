"""
evaluate_all_benchmarks.py
--------------------------
Calculates Precision, Recall, and F1-Score at IoU >= 0.50, Conf >= 0.50
across standard 32 FDI teeth for all trained models on:
1. Internal Institutional Dataset (val split)
2. External Multi-Center DENTEX Benchmark
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

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


def evaluate_torchvision_model(model, img_dir, lbl_dir, max_imgs=None):
    img_paths = sorted(Path(img_dir).glob("*.*"))
    if max_imgs:
        img_paths = img_paths[:max_imgs]
    
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for img_path in img_paths:
            im = cv2.imread(str(img_path))
            if im is None: continue
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
                            x1 = (cx - bw/2) * 1280
                            y1 = (cy - bh/2) * 1280
                            x2 = (cx + bw/2) * 1280
                            y2 = (cy + bh/2) * 1280
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_classes.append(c + 1)

            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            gt_classes = np.array(gt_classes) if gt_classes else np.zeros((0,))

            keep = (preds["scores"].cpu().numpy() >= 0.50)
            p_boxes = preds["boxes"][keep].cpu().numpy()
            p_labels = preds["labels"][keep].cpu().numpy()
            valid = (p_labels >= 1) & (p_labels <= 32)
            p_boxes = p_boxes[valid]
            p_labels = p_labels[valid]

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
                    if j in matched_gt: continue
                    if p_labels[i] == gt_classes[j] and iou_mat[i, j] >= 0.50:
                        if iou_mat[i, j] > best_iou:
                            best_iou = iou_mat[i, j]
                            best_j = j
                if best_j >= 0:
                    matched_gt.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += (len(gt_boxes) - len(matched_gt))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return prec, rec, f1, tp, fp, fn


def main():
    print("=" * 65)
    print("  Full Benchmark Evaluation (Standard 32 FDI Teeth)")
    print("=" * 65)

    models_to_eval = [
        ("Faster-RCNN", "runs/classic/Faster-RCNN/best.pt", "fasterrcnn"),
        ("RetinaNet", "runs/classic/RetinaNet/best.pt", "retinanet"),
        ("Cascade-RCNN", "runs/classic/Cascade-RCNN/best.pt", "fasterrcnn"),
    ]

    for name, path_str, arch in models_to_eval:
        ckpt_path = BASE_DIR / path_str
        if not ckpt_path.exists():
            continue

        print(f"\n▶ Evaluating {name} ({ckpt_path.name})...")
        if arch == "fasterrcnn":
            model = fasterrcnn_resnet50_fpn_v2(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 49)
        elif arch == "retinanet":
            model = retinanet_resnet50_fpn_v2(weights=None)
            in_channels = model.backbone.out_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head = RetinaNetHead(in_channels, num_anchors, 49)

        model.load_state_dict(torch.load(ckpt_path, map_location="cuda:0"))
        model.cuda()
        model.eval()

        # Internal
        iprec, irec, if1, itp, ifp, ifn = evaluate_torchvision_model(model, INTERNAL_IMG_DIR, INTERNAL_LBL_DIR)
        print(f"  [Internal] Precision: {iprec:.4f} | Recall: {irec:.4f} | F1: {if1:.4f}")

        # DENTEX
        if DENTEX_IMG_DIR.exists():
            dprec, drec, df1, dtp, dfp, dfn = evaluate_torchvision_model(model, DENTEX_IMG_DIR, DENTEX_LBL_DIR, max_imgs=100)
            print(f"  [DENTEX]   Precision: {dprec:.4f} | Recall: {drec:.4f} | F1: {df1:.4f}")


if __name__ == "__main__":
    main()
