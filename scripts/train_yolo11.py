"""
train_yolo11.py
---------------
Train YOLOv11 on the Aneesa dataset (48 classes, 574 images).

GPU   : Quadro RTX 8000 (49 GB)
Models: yolo11n / yolo11s / yolo11m / yolo11l / yolo11x
        Set MODEL_SIZE below to choose.

Usage:
    conda run -n yolov8 python scripts/train_yolo11.py
    conda run -n yolov8 python scripts/train_yolo11.py --model yolo11l --epochs 150
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]   # …/yolov11_training/
DATASET_YAML = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
RUNS_DIR     = BASE_DIR / "runs"

def parse_args():
    ap = argparse.ArgumentParser(description="Train YOLOv11 on Aneesa dataset")
    ap.add_argument("--model",   default="yolo11x",  help="Model size: yolo11n/s/m/l/x")
    ap.add_argument("--epochs",  type=int, default=150,  help="Number of epochs")
    ap.add_argument("--imgsz",   type=int, default=640,  help="Input image size")
    ap.add_argument("--batch",   type=int, default=16,   help="Batch size (-1 = auto)")
    ap.add_argument("--resume",  default=None,           help="Path to checkpoint to resume from")
    ap.add_argument("--name",    default=None,           help="Run name (default: auto)")
    return ap.parse_args()


def main():
    args = parse_args()

    run_name = args.name or f"{args.model}_ep{args.epochs}_img{args.imgsz}"

    print("=" * 60)
    print(f"  Model   : {args.model}.pt")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"  Run name: {run_name}")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n▶ Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\n▶ Loading pretrained weights: {args.model}.pt")
        model = YOLO(f"{args.model}.pt")

    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(
        data        = str(DATASET_YAML),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = 0,                      # GPU 0 (Quadro RTX 8000)
        project     = str(RUNS_DIR / "train"),
        name        = run_name,

        # ── Optimizer & LR ───────────────────────────────────────────────────
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        warmup_epochs = 5,
        cos_lr      = True,

        # ── Regularisation ───────────────────────────────────────────────────
        weight_decay = 0.0005,
        dropout      = 0.0,

        # ── Augmentation ─────────────────────────────────────────────────────
        mosaic      = 1.0,
        mixup       = 0.1,
        copy_paste  = 0.1,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.0,
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,

        # ── Misc ─────────────────────────────────────────────────────────────
        patience    = 30,                     # early stopping
        save_period = 10,                     # save checkpoint every N epochs
        plots       = True,
        verbose     = True,
    )

    print("\n✅  Training complete!")
    print(f"   Best model : {RUNS_DIR}/train/{run_name}/weights/best.pt")
    print(f"   Results    : {RUNS_DIR}/train/{run_name}/results.csv")

    # ── Validate best model ───────────────────────────────────────────────────
    print("\n▶ Validating best model …")
    best_pt = RUNS_DIR / "train" / run_name / "weights" / "best.pt"
    val_model = YOLO(str(best_pt))
    metrics = val_model.val(data=str(DATASET_YAML), device=0)
    print(f"\n  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
