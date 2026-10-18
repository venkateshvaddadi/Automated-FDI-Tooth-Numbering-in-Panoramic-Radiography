"""
master_benchmark_queue.py
-------------------------
Unified Master Benchmark Queue:
Continuously checks for completed models and trains all remaining architectures sequentially:
1. Modern YOLO / Transformer Suite (via Ultralytics):
   - bench_yolov9c (Already Completed)
   - bench_yolov9e (Currently Running)
   - bench_rtdetr_x
   - bench_yolo11x
2. Classic Baselines (via PyTorch / Torchvision):
   - Faster R-CNN (ResNet-50-FPN-v2 at 1280px)
   - RetinaNet (ResNet-50-FPN-v2 at 1280px)

All models adhere to:
  - imgsz: 1280
  - epochs: 100
  - seed: 0
  - fliplr: 0.0
"""

import subprocess
import sys
import time
from pathlib import Path
from ultralytics import RTDETR, YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
RUNS_DIR = BASE_DIR / "runs"

MODELS_ULTRA = [
    ("bench_yolov9c", "yolov9c.pt", "YOLO"),
    ("bench_yolov9e", "yolov9e.pt", "YOLO"),
    ("bench_rtdetr_x", "rtdetr-x.pt", "RTDETR"),
    ("bench_yolo11x", "yolo11x.pt", "YOLO"),
]


def run_ultra():
    for name, weights, kind in MODELS_ULTRA:
        best_pt = RUNS_DIR / "train" / name / "weights" / "best.pt"
        if best_pt.exists():
            print(f"[SKIP] {name}: already trained ({best_pt})", flush=True)
            continue

        print(f"\n=======================================================", flush=True)
        print(f" ▶ Starting Training: {name} ({weights})", flush=True)
        print(f"=======================================================", flush=True)
        try:
            model_cls = RTDETR if kind == "RTDETR" else YOLO
            model = model_cls(weights)
            model.train(
                data=str(DATASET_YAML),
                epochs=100,
                imgsz=1280,
                batch=8,
                device=0,
                fliplr=0.0,
                seed=0,
                project=str(RUNS_DIR / "train"),
                name=name,
                exist_ok=True,
                patience=30,
            )
            print(f"✅ Successfully finished {name}!\n", flush=True)
        except Exception as e:
            print(f"❌ Error training {name}: {e}", flush=True)


def run_classic():
    classic_script = BASE_DIR / "scripts" / "train_classic_detectors.py"
    for m in ["Faster-RCNN", "RetinaNet"]:
        ckpt = RUNS_DIR / "classic" / m / "best.pt"
        if ckpt.exists():
            print(f"[SKIP] Classic {m}: already trained ({ckpt})", flush=True)
            continue
        print(f"\n▶ Launching Classic Detector: {m} ...", flush=True)
        cmd = [sys.executable, str(classic_script), m]
        subprocess.run(cmd, check=False)


def main():
    print("=" * 65)
    print("  Master Benchmark Pipeline Active")
    print("=" * 65)
    run_ultra()
    run_classic()
    print("\n🎉 ALL requested benchmark models in the queue are completed!")


if __name__ == "__main__":
    main()
