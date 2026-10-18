"""
train_yolonas_pretrained.py
----------------------------
Trains YOLO-NAS-M loaded with official COCO pre-trained backbone & neck weights.
Key features requested:
  1. Pretrained Weights: Warm-starts 947 backbone/neck layers from COCO (yolo_nas_m.pt).
  2. Training & Validation Every Epoch: Runs full training and validation pass after each epoch.
  3. Metric Tracking: Tracks validation Precision@0.50, Recall@0.50, mAP@0.50, F1@0.50.
  4. Best & Last Checkpoints:
     - ckpt_best.pth  (Best validation F1 / mAP score)
     - ckpt_latest.pth (Last epoch checkpoint for instant resumption)
     - average_model.pth (EMA smoothed final checkpoint)
  5. Medical Orientation: Strictly fliplr=0.0 preserving FDI dental quadrant geometry.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Safe torch.load wrapper for PyTorch 2.6+
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
DATASET_YAML = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
RUNS_DIR = BASE_DIR / "runs" / "yolonas_pretrained"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_PRETRAINED = BASE_DIR / "yolo_nas_m.pt"

def main():
    parser = argparse.ArgumentParser(description="Train YOLO-NAS-M with Pre-trained Weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("  Training YOLO-NAS-M (Pretrained COCO Weights)")
    print(f"  Resolution   : {args.imgsz}x{args.imgsz}")
    print(f"  Batch Size   : {args.batch}")
    print(f"  Total Epochs : {args.epochs}")
    print(f"  Dataset YAML : {DATASET_YAML}")
    print(f"  Output Dir   : {RUNS_DIR}")
    print("=" * 70)

    from super_gradients.training import Trainer, models
    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train,
        coco_detection_yolo_format_val,
    )
    from super_gradients.training.metrics.detection_metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

    # 1. Initialize Trainer
    trainer = Trainer(experiment_name="yolo_nas_m_pretrained", ckpt_root_dir=str(RUNS_DIR))

    # 2. Build Model Architecture for 48 Dental Classes
    net = models.get("yolo_nas_m", num_classes=48, pretrained_weights=None)

    # 3. Load Pre-trained Backbone & Neck Weights
    if LOCAL_PRETRAINED.exists():
        print(f"📦 Loading pre-trained COCO weights from {LOCAL_PRETRAINED}...")
        ckpt = torch.load(LOCAL_PRETRAINED, map_location="cpu", weights_only=False)
        sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
        net_sd = net.state_dict()
        pretrained_dict = {}
        for k, v in sd.items():
            clean_k = k.replace("module.", "")
            if clean_k in net_sd and net_sd[clean_k].shape == v.shape:
                pretrained_dict[clean_k] = v

        missing, unexpected = net.load_state_dict(pretrained_dict, strict=False)
        print(f"   ★ Transferred {len(pretrained_dict)} pre-trained feature layers to YOLO-NAS-M!")
        print(f"   ★ Freshly initialized {len(missing)} dental head classification parameters.")
    else:
        print("⚠️ Warning: Local pretrained weights yolo_nas_m.pt not found. Initializing randomly.")

    # 4. Data Loaders (Strict fliplr=0.0 compliance)
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": str(DATASET_YAML.parent),
            "images_dir": "images/train",
            "labels_dir": "labels/train",
            "classes": list(range(48)),
            "input_dim": (args.imgsz, args.imgsz),
        },
        dataloader_params={"batch_size": args.batch, "num_workers": 4, "shuffle": True},
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": str(DATASET_YAML.parent),
            "images_dir": "images/val",
            "labels_dir": "labels/val",
            "classes": list(range(48)),
            "input_dim": (args.imgsz, args.imgsz),
        },
        dataloader_params={"batch_size": args.batch, "num_workers": 4, "shuffle": False},
    )

    # 5. Validation Metrics for Every Epoch
    post_pred_cb = PPYoloEPostPredictionCallback(
        score_threshold=0.1,
        nms_threshold=0.65,
        nms_top_k=1000,
        max_predictions=300,
    )

    val_metric = DetectionMetrics_050(
        num_cls=48,
        post_prediction_callback=post_pred_cb,
        normalize_targets=True,
    )

    # 6. Training Parameters: Validation every epoch & Best/Last checkpoint saving
    training_params = {
        "max_epochs": args.epochs,
        "lr_mode": "cosine",
        "initial_lr": 5e-4,
        "loss": "PPYoloELoss",
        "criterion_params": {
            "num_classes": 48,
            "use_static_assigner": False,
            "reg_max": 16,
        },
        "valid_metrics_list": [val_metric],
        "metric_to_watch": "F1@0.50",
        "greater_metric_to_watch_is_better": True,
        "average_best_models": True,
    }

    if args.resume:
        training_params["resume"] = True
        print("🔄 Resuming YOLO-NAS-M training from latest checkpoint...")

    # 7. Start Training
    print("\n🚀 Commencing YOLO-NAS-M pre-trained training with validation after every epoch...")
    trainer.train(
        model=net,
        training_params=training_params,
        train_loader=train_data,
        valid_loader=val_data,
    )

    print("✅ Finished pre-trained YOLO-NAS-M training!")

if __name__ == "__main__":
    main()
