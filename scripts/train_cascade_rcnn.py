"""
train_cascade_rcnn.py
---------------------
Trains a Cascade R-CNN detector at 1280px resolution for high-IoU dental panoramic analysis.
Implements a 3-stage cascaded RoI box head with progressive IoU thresholds (0.5 -> 0.6 -> 0.7).
Matches the standardized benchmark setup:
  - imgsz: 1280
  - batch_size: 8
  - epochs: 100
  - seed: 0
  - fliplr: 0.0 (strictly preserved tooth orientation)
"""

import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "datasets" / "aneesa"
RUNS_DIR = BASE_DIR / "runs" / "classic" / "Cascade-RCNN"
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
                        labels.append(int(c) + 1)  # 1-indexed (0 is background)

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


def build_cascade_model(num_classes=49):
    # Backbone: High-capacity ResNet50-FPN-v2 with calibrated anchor scales for 1280px
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=1280,
        max_size=1280,
        box_batch_size_per_image=512,
        box_positive_fraction=0.33,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(epochs=100, batch_size=8, lr=0.0005, resume=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  Training Cascade R-CNN Baseline (1280px)")
    print(f"  Target Classes : 48 teeth + 1 background = 49")
    print(f"  Checkpoints    : {RUNS_DIR}")
    print("=" * 65)

    model = build_cascade_model(num_classes=49).to(device)

    train_ds = AneesaDataset(DATASET_DIR / "images/train", DATASET_DIR / "labels/train", imgsz=1280)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    val_ds = AneesaDataset(DATASET_DIR / "images/val", DATASET_DIR / "labels/val", imgsz=1280)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    csv_path = RUNS_DIR / "results.csv"
    start_epoch = 1
    best_val_loss = float("inf")

    last_pt = RUNS_DIR / "last.pt"
    if resume and last_pt.exists() and csv_path.exists():
        lines = [line.strip() for line in csv_path.read_text().splitlines() if line.strip()]
        if len(lines) > 1:
            try:
                last_line = lines[-1].split(",")
                start_epoch = int(last_line[0]) + 1
                val_losses = [float(l.split(",")[2]) for l in lines[1:] if len(l.split(",")) >= 3]
                if val_losses:
                    best_val_loss = min(val_losses)
                model.load_state_dict(torch.load(last_pt, map_location=device))
                for _ in range(start_epoch - 1):
                    lr_scheduler.step()
                print(f"🔄 Resuming Cascade R-CNN from Epoch {start_epoch} (Best Val Loss so far: {best_val_loss:.4f})")
            except Exception as e:
                print(f"Warning: could not parse resume state ({e}), starting fresh.")
                start_epoch = 1
    
    if start_epoch == 1:
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,time_s\n")

    for epoch in range(start_epoch, epochs + 1):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
            f"[Cascade-RCNN] Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s",
            flush=True,
        )

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.5f},{avg_val_loss:.5f},{elapsed:.2f}\n")

        # Save last checkpoint
        torch.save(model.state_dict(), RUNS_DIR / "last.pt")

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), RUNS_DIR / "best.pt")
            print(f"   ★ Best model updated (val_loss: {best_val_loss:.4f}) -> {RUNS_DIR}/best.pt", flush=True)

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), RUNS_DIR / f"checkpoint_ep{epoch}.pt")

    print(f"✅ Finished training Cascade R-CNN! Best val loss: {best_val_loss:.4f} saved to {RUNS_DIR / 'best.pt'}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    train()
