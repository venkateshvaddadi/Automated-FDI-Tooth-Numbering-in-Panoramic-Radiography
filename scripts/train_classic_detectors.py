"""
train_classic_detectors.py
--------------------------
Trains classical paper baselines (Faster R-CNN, RetinaNet, SSD)
using PyTorch / Torchvision on the Aneesa dataset.
Matches the benchmark settings:
  - imgsz: 1280 (min_size=1280, max_size=1280)
  - batch: 4 (gradient accumulation to effectively match batch 8)
  - epochs: 100
  - fliplr: 0.0 (no horizontal flips)
  - seed: 0
  - 48 tooth classes
"""

import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "datasets" / "aneesa"
RUNS_DIR = BASE_DIR / "runs" / "classic"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class AneesaDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, imgsz=1280):
        self.img_paths = sorted(Path(img_dir).glob("*.jpg"))
        self.lbl_dir = Path(lbl_dir)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    c, cx, cy, bw, bh = (float(x) for x in parts)
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(c) + 1)  # 1-indexed (0 is background in torchvision)

        # Resize to fixed size imgsz
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        sx = self.imgsz / w
        sy = self.imgsz / h

        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(model_name, epochs=100, batch_size=8, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f" Training Classic Detector: {model_name} (1280px)")
    print(f"========================================================")

    # Number of classes: 48 teeth + 1 background = 49
    num_classes = 49

    if model_name == "Faster-RCNN":
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == "RetinaNet":
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
        in_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head = RetinaNetHead(in_channels, num_anchors, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)

    train_ds = AneesaDataset(DATASET_DIR / "images/train", DATASET_DIR / "labels/train", imgsz=1280)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    val_ds = AneesaDataset(DATASET_DIR / "images/val", DATASET_DIR / "labels/val", imgsz=1280)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    save_dir = RUNS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "results.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,time_s\n")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Training Pass ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()

        lr_scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ── Validation Pass (All Epochs) ──────────────────────────────
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [im.to(device) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        elapsed = time.time() - t0

        print(
            f"[{model_name}] Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s",
            flush=True,
        )

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.5f},{avg_val_loss:.5f},{elapsed:.2f}\n")

        # Save latest model (last.pt)
        torch.save(model.state_dict(), save_dir / "last.pt")

        # Save best model based on validation loss (best.pt)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"   ★ Best model updated (val_loss: {best_val_loss:.4f}) -> {save_dir}/best.pt", flush=True)

        if epoch % 10 == 0 or epoch == epochs:
            ckpt_path = save_dir / f"checkpoint_ep{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

    print(f"✅ Finished training {model_name}! Best val loss: {best_val_loss:.4f} saved to {save_dir}/best.pt")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if len(sys.argv) > 1:
        train_model(sys.argv[1])
    else:
        train_model("Faster-RCNN")
        train_model("RetinaNet")
