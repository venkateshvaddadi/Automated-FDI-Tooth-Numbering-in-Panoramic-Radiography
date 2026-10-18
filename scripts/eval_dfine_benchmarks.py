"""
eval_dfine_benchmarks.py
-------------------------
Evaluates D-FINE-M on:
1. Institutional Validation Dataset (datasets/aneesa/images/val)
2. External Multi-Center DENTEX Benchmark (dentex_yolo)
Across standard 32 FDI teeth using micro Precision, Recall, and F1 at IoU >= 0.50.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "dfine_repo"))

from src.core import YAMLConfig

CFG_PATH = BASE_DIR / "dfine_repo" / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_dental_1280.yml"
BEST_CKPT = BASE_DIR / "runs" / "dfine_m_1280" / "best_stg1.pth"
LAST_CKPT = BASE_DIR / "runs" / "dfine_m_1280" / "last.pth"

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

class DeployModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def evaluate_dfine(model, img_dir, lbl_dir, device="cuda", conf_thresh=0.50, iou_thresh=0.50):
    img_paths = sorted(Path(img_dir).glob("*.*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    
    transforms = T.Compose([
        T.Resize((1280, 1280)),
        T.ToTensor(),
    ])

    tp, fp, fn = 0, 0, 0
    total_gt = 0
    total_det = 0

    with torch.no_grad():
        for img_path in img_paths:
            lbl_file = Path(lbl_dir) / f"{img_path.stem}.txt"
            if not lbl_file.exists():
                continue

            try:
                im_pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            orig_w, orig_h = im_pil.size
            orig_size = torch.tensor([[orig_w, orig_h]], device=device)
            im_data = transforms(im_pil).unsqueeze(0).to(device)

            labels, boxes, scores = model(im_data, orig_size)
            
            p_labels = labels[0].cpu().numpy()
            p_boxes = boxes[0].cpu().numpy()
            p_scores = scores[0].cpu().numpy()

            # Filter by confidence and standard 32 tooth classes (ids 0..31)
            keep = (p_scores >= conf_thresh) & (p_labels < 32)
            p_labels = p_labels[keep]
            p_boxes = p_boxes[keep]
            total_det += len(p_boxes)

            # Ground truth
            gt_boxes, gt_classes = [], []
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    c, cx, cy, bw, bh = (float(x) for x in parts)
                    c = int(c)
                    if c < 32:
                        gx1 = (cx - bw / 2.0) * orig_w
                        gy1 = (cy - bh / 2.0) * orig_h
                        gx2 = (cx + bw / 2.0) * orig_w
                        gy2 = (cy + bh / 2.0) * orig_h
                        gt_boxes.append([gx1, gy1, gx2, gy2])
                        gt_classes.append(c)

            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            gt_classes = np.array(gt_classes) if gt_classes else np.zeros((0,), dtype=int)
            total_gt += len(gt_boxes)

            if len(gt_boxes) == 0:
                fp += len(p_boxes)
                continue
            if len(p_boxes) == 0:
                fn += len(gt_boxes)
                continue

            iou_mat = box_iou(p_boxes, gt_boxes)
            matched_gt = set()

            for i in range(len(p_boxes)):
                best_iou = -1.0
                best_j = -1
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

            fn += (len(gt_boxes) - len(matched_gt))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "total_det": total_det,
    }

def main():
    ckpt_path = BEST_CKPT if BEST_CKPT.exists() else LAST_CKPT
    print(f"Loading D-FINE-M configuration: {CFG_PATH}")
    print(f"Loading checkpoint: {ckpt_path}")

    cfg = YAMLConfig(str(CFG_PATH), resume=str(ckpt_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("Using EMA weights from checkpoint.")
    else:
        state = checkpoint["model"]
        print("Using standard model weights from checkpoint.")

    cfg.model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeployModel(cfg).to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("  D-FINE-M EVALUATION (32 Standard FDI Tooth Classes)")
    print("=" * 70)

    # Internal Validation Set
    print("\n[1/2] Evaluating on Internal Institutional Dataset (102 images)...")
    res_int_50 = evaluate_dfine(model, INTERNAL_IMG_DIR, INTERNAL_LBL_DIR, device=device, conf_thresh=0.50)
    res_int_25 = evaluate_dfine(model, INTERNAL_IMG_DIR, INTERNAL_LBL_DIR, device=device, conf_thresh=0.25)

    print(f"  Internal @ Conf >= 0.50: Precision = {res_int_50['precision']*100:.2f}% | Recall = {res_int_50['recall']*100:.2f}% | F1 = {res_int_50['f1']*100:.2f}% (TP={res_int_50['tp']}, FP={res_int_50['fp']}, FN={res_int_50['fn']})")
    print(f"  Internal @ Conf >= 0.25: Precision = {res_int_25['precision']*100:.2f}% | Recall = {res_int_25['recall']*100:.2f}% | F1 = {res_int_25['f1']*100:.2f}% (TP={res_int_25['tp']}, FP={res_int_25['fp']}, FN={res_int_25['fn']})")

    # DENTEX Benchmark
    if DENTEX_IMG_DIR.exists():
        print("\n[2/2] Evaluating on External Multi-Center DENTEX Benchmark (271 images)...")
        res_den_50 = evaluate_dfine(model, DENTEX_IMG_DIR, DENTEX_LBL_DIR, device=device, conf_thresh=0.50)
        res_den_25 = evaluate_dfine(model, DENTEX_IMG_DIR, DENTEX_LBL_DIR, device=device, conf_thresh=0.25)

        print(f"  DENTEX   @ Conf >= 0.50: Precision = {res_den_50['precision']*100:.2f}% | Recall = {res_den_50['recall']*100:.2f}% | F1 = {res_den_50['f1']*100:.2f}% (TP={res_den_50['tp']}, FP={res_den_50['fp']}, FN={res_den_50['fn']})")
        print(f"  DENTEX   @ Conf >= 0.25: Precision = {res_den_25['precision']*100:.2f}% | Recall = {res_den_25['recall']*100:.2f}% | F1 = {res_den_25['f1']*100:.2f}% (TP={res_den_25['tp']}, FP={res_den_25['fp']}, FN={res_den_25['fn']})")
    else:
        print("\nDENTEX directory not found at", DENTEX_IMG_DIR)

    print("\n" + "=" * 70)
    print("  Evaluation Completed Successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
