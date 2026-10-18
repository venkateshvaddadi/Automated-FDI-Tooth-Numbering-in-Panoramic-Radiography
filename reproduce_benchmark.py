"""
reproduce_benchmark.py
======================
Standalone reproduction script for:
'Benchmarking Modern Object Detection Paradigms for Automated FDI Tooth Numbering
in Panoramic Radiography: A Multi-Center Generalization Study and Interactive Decision-Support Tool'

Computes:
1. Precision, Recall, F1-Score at IoU >= 0.50, Conf >= 0.50 across 32 standard adult FDI tooth positions
2. Spatial boundary IoU distribution (Mean, Median, IQR)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

# Standard adult 32 FDI tooth identities (ISO 3950:2016)
FDI_32 = [
    "18","17","16","15","14","13","12","11",
    "21","22","23","24","25","26","27","28",
    "38","37","36","35","34","33","32","31",
    "41","42","43","44","45","46","47","48"
]

def clean_fdi(name):
    return str(name).split("-")[0].strip()

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

def evaluate_cohort(model, img_dir, lbl_dir, names_map=None, imgsz=1280, conf=0.50, iou_thresh=0.50):
    img_paths = sorted(Path(img_dir).glob("*.*"))
    if not img_paths:
        print(f"No images found in {img_dir}")
        return {}

    tp, fp, fn = 0, 0, 0
    ious = []

    for img_p in img_paths:
        im = cv2.imread(str(img_p))
        if im is None: continue
        h_img, w_img = im.shape[:2]

        lbl_p = Path(lbl_dir) / f"{img_p.stem}.txt"
        gt_boxes, gt_classes = [], []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    c = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:]]
                    fdi_name = names_map.get(c, str(c)) if names_map else model.names.get(c, str(c))
                    fdi = clean_fdi(fdi_name)
                    if fdi in FDI_32:
                        x1 = (cx - bw/2) * w_img
                        y1 = (cy - bh/2) * h_img
                        x2 = (cx + bw/2) * w_img
                        y2 = (cy + bh/2) * h_img
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(fdi)

        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))

        # Model inference
        res = model.predict(str(img_p), imgsz=imgsz, conf=conf, iou=iou_thresh, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
        p_boxes, p_classes = [], []
        for b in res.boxes:
            c = int(b.cls.item())
            fdi = clean_fdi(model.names[c])
            if fdi in FDI_32:
                p_boxes.append(b.xyxy[0].cpu().numpy())
                p_classes.append(fdi)

        p_boxes = np.array(p_boxes) if p_boxes else np.zeros((0, 4))

        if len(gt_boxes) == 0:
            fp += len(p_boxes)
            continue
        if len(p_boxes) == 0:
            fn += len(gt_boxes)
            continue

        iou_matrix = box_iou(p_boxes, gt_boxes)
        matched_gt = set()
        for i in range(len(p_boxes)):
            best_iou, best_j = 0.0, -1
            for j in range(len(gt_boxes)):
                if j in matched_gt: continue
                if p_classes[i] == gt_classes[j] and iou_matrix[i, j] >= iou_thresh:
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_j = j
            if best_j >= 0:
                matched_gt.add(best_j)
                tp += 1
                ious.append(best_iou)
            else:
                fp += 1
        fn += (len(gt_boxes) - len(matched_gt))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "ious": np.array(ious) if ious else np.zeros((0,)),
    }

def main():
    parser = argparse.ArgumentParser(description="Reproduce FDI tooth numbering benchmark metrics.")
    parser.add_argument("--weights", type=str, default="yolo11m_teeth_1280.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-type", type=str, default="yolo", choices=["yolo", "rtdetr"], help="Model type")
    parser.add_argument("--img-dir", type=str, default="samples/images", help="Image directory")
    parser.add_argument("--lbl-dir", type=str, default="samples/labels", help="Label directory")
    parser.add_argument("--imgsz", type=int, default=1280, help="Evaluation input resolution")
    args = parser.parse_args()

    print("=" * 70)
    print("  Reproduction Benchmark: Automated FDI Tooth Numbering")
    print("=" * 70)
    print(f"Weights:    {args.weights}")
    print(f"Image Dir:  {args.img_dir}")
    print(f"Label Dir:  {args.lbl_dir}")
    print(f"Resolution: {args.imgsz}x{args.imgsz}")

    if not Path(args.weights).exists():
        print(f"\n[Notice] Checkpoint {args.weights} not found locally.")
        print("Please specify a valid path with --weights <path>.")
        return

    print(f"Loading {args.model_type.upper()} model...")
    model = RTDETR(args.weights) if args.model_type == "rtdetr" else YOLO(args.weights)

    # Names mapping from model
    results = evaluate_cohort(model, args.img_dir, args.lbl_dir, names_map=model.names, imgsz=args.imgsz)
    if not results: return

    print("\n" + "-" * 70)
    print("  Quantitative Metrics (32 Adult FDI Classes, IoU >= 0.50, Conf >= 0.50)")
    print("-" * 70)
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  TP: {results['tp']} | FP: {results['fp']} | FN: {results['fn']}")
    
    if len(results['ious']) > 0:
        ious = results['ious']
        print("-" * 70)
        print("  Spatial Localization Fidelity on Matched True Positives:")
        print(f"  Mean IoU:   {ious.mean():.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  IQR [25%, 75%]: [{np.percentile(ious, 25):.4f}, {np.percentile(ious, 75):.4f}]")

    print("=" * 70)

if __name__ == "__main__":
    main()
