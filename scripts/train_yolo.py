"""
train_yolo.py
-------------
Unified training script for modern YOLO architectures (YOLOv8, YOLOv10, YOLO11, YOLO12, etc.)
and modern PyTorch checkpoints on the Aneesa dataset.

Usage:
    # Train latest YOLO11 / YOLO12 / YOLOv8 / YOLOv10:
    python scripts/train_yolo.py --model yolo11x --epochs 150 --imgsz 640
    python scripts/train_yolo.py --model yolo12m --epochs 150 --imgsz 1280
    python scripts/train_yolo.py --model yolov8m --epochs 150 --imgsz 640
    python scripts/train_yolo.py --model yolov10m --epochs 150 --imgsz 640

    # Custom weights or resumed training:
    python scripts/train_yolo.py --model /path/to/weights.pt --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
RUNS_DIR = BASE_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train modern YOLO models on Aneesa dataset")
    parser.add_argument(
        "--model",
        default="yolo11x",
        help="Model architecture or weights file (e.g. yolo11n/s/m/l/x, yolo12m, yolov8m, yolov10m, or path to .pt)",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto-batch)")
    parser.add_argument("--device", default="0", help="CUDA device ID (e.g. 0) or 'cpu'")
    parser.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--name", default=None, help="Custom run name")
    parser.add_argument(
        "--no-fliplr",
        action="store_true",
        help="Disable horizontal flips (essential if left/right orientation matters, e.g. tooth numbering)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine model filename / path
    model_spec = args.model
    if not model_spec.endswith(".pt") and not Path(model_spec).exists():
        model_weights = f"{model_spec}.pt"
    else:
        model_weights = model_spec

    model_clean_name = Path(model_weights).stem
    run_name = args.name or f"{model_clean_name}_ep{args.epochs}_img{args.imgsz}"

    print("=" * 65)
    print(f"  Architecture/Weights : {model_weights}")
    print(f"  Epochs               : {args.epochs}")
    print(f"  Image Size           : {args.imgsz}")
    print(f"  Batch Size           : {args.batch}")
    print(f"  Device               : {args.device}")
    print(f"  Dataset Config       : {DATASET_YAML}")
    print(f"  Run Output Name      : {run_name}")
    print("=" * 65)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found at: {DATASET_YAML}")

    # ── Initialize model ──────────────────────────────────────────────────────
    if args.resume:
        print(f"\n▶ Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\n▶ Loading model / weights: {model_weights}")
        model = YOLO(model_weights)

    # ── Train ─────────────────────────────────────────────────────────────────
    fliplr_val = 0.0 if args.no_fliplr else 0.5

    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR / "train"),
        name=run_name,
        # Optimizer & Schedule
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        cos_lr=True,
        # Regularization
        weight_decay=0.0005,
        dropout=0.0,
        # Augmentations
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=fliplr_val,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Training controls
        patience=30,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    best_pt = RUNS_DIR / "train" / run_name / "weights" / "best.pt"
    print(f"   Best model weights : {best_pt}")
    print(f"   Training metrics   : {RUNS_DIR}/train/{run_name}/results.csv")

    # ── Validate best model ───────────────────────────────────────────────────
    if best_pt.exists():
        print("\n▶ Running validation on best model weights …")
        val_model = YOLO(str(best_pt))
        metrics = val_model.val(data=str(DATASET_YAML), device=args.device)
        print(f"\n  Validation Results ({model_clean_name}):")
        print(f"    mAP50    : {metrics.box.map50:.4f}")
        print(f"    mAP50-95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
