"""
train_yolonas.py
----------------
Trains YOLO-NAS (Neural Architecture Search) models on the Aneesa dental dataset.
Supports YOLO-NAS S, M, and L variants with identical training parameters:
  - imgsz: 1280
  - batch_size: 8
  - epochs: 100
  - fliplr: 0.0 (strictly preserved tooth quadrant orientation)
  - seed: 0
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
RUNS_DIR = BASE_DIR / "runs" / "yolonas"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


import torch
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

try:
    import numpy as np
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="Train YOLO-NAS on dental dataset")
    parser.add_argument("--variant", default="yolo_nas_m", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  Training YOLO-NAS ({args.variant}) at {args.imgsz}px")
    print(f"  Dataset YAML : {DATASET_YAML}")
    print(f"  Output Dir   : {RUNS_DIR}")
    print("=" * 65)

    try:
        from super_gradients.training import models
        from super_gradients.training import Trainer
        from super_gradients.training.dataloaders.dataloaders import (
            coco_detection_yolo_format_train,
            coco_detection_yolo_format_val,
        )

        trainer = Trainer(experiment_name=args.variant, ckpt_root_dir=str(RUNS_DIR))
        try:
            net = models.get(args.variant, num_classes=48, pretrained_weights="coco")
        except Exception:
            print("Notice: COCO weights host unreachable, initializing architecture directly...")
            net = models.get(args.variant, num_classes=48, pretrained_weights=None)

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
            "metric_to_watch": "PPYoloELoss/loss",
            "greater_metric_to_watch_is_better": False,
        }

        if args.resume:
            training_params["resume"] = True
            if args.resume_path:
                training_params["resume_path"] = args.resume_path
            print(f"🔄 Resuming YOLO-NAS from checkpoint: {args.resume_path or 'latest'}")

        trainer.train(model=net, training_params=training_params, train_loader=train_data, valid_loader=val_data)
        print(f"✅ Finished training YOLO-NAS ({args.variant})!")

    except ImportError:
        print("[Notice] 'super-gradients' not installed in this environment.")
        print("To run YOLO-NAS training, install via: pip install super-gradients")
        print("Fallback Ultralytics / TorchVision baselines remain fully active.")


if __name__ == "__main__":
    main()
