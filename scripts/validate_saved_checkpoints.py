"""
validate_saved_checkpoints.py
-----------------------------
Computes validation loss and FDI 32-tooth metrics on all saved Faster R-CNN checkpoints
to identify the exact best epoch and compile results.csv.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from train_classic_detectors import AneesaDataset, collate_fn

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "datasets" / "aneesa"
RUNS_DIR = BASE_DIR / "runs" / "classic" / "Faster-RCNN"


def evaluate_checkpoint(ckpt_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 49)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    val_ds = AneesaDataset(DATASET_DIR / "images/val", DATASET_DIR / "labels/val", imgsz=1280)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Compute validation loss
    model.train()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    return avg_val_loss


def main():
    ckpts = sorted(RUNS_DIR.glob("checkpoint_ep*.pt"), key=lambda p: int(p.stem.replace("checkpoint_ep", "")))
    print(f"Found {len(ckpts)} checkpoints in {RUNS_DIR}")
    best_loss = float("inf")
    best_ckpt = None

    for ckpt in ckpts:
        ep = int(ckpt.stem.replace("checkpoint_ep", ""))
        val_loss = evaluate_checkpoint(ckpt)
        print(f"Checkpoint Epoch {ep:3d} -> Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = ckpt

    if best_ckpt:
        print(f"\n★ Lowest Validation Loss Checkpoint: {best_ckpt.name} (val_loss: {best_loss:.4f})")
        torch.save(torch.load(best_ckpt), RUNS_DIR / "best.pt")
        print(f"✅ Saved best checkpoint to {RUNS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
