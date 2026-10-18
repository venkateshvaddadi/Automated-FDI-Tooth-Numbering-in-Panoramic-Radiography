"""
benchmark_extended.py
---------------------
Third benchmark queue: Trains additional models from YOLO and Transformer families
with the exact identical dataset, resolution, hyperparameters, and seed:
  - imgsz   : 1280
  - batch   : 8
  - epochs  : 100
  - fliplr  : 0.0 (prevents tooth left/right swapping)
  - seed    : 0
  - patience: 30

Models included:
  1. YOLOv9c / YOLOv9e (Programmable Gradient Information architecture)
  2. RT-DETR-X (Extra-large real-time transformer detector)
  3. YOLO11x (Extra-large YOLO11 variant)
"""

import traceback
from pathlib import Path
from ultralytics import RTDETR, YOLO

HERE = Path(__file__).resolve().parents[1]
DATASET_YAML = HERE / "datasets" / "aneesa" / "dataset.yaml"
RUNS_DIR = HERE / "runs"

MODELS = [
    # (run_name, weights_name, model_type)
    ("bench_yolov9c", "yolov9c.pt", "YOLO"),
    ("bench_yolov9e", "yolov9e.pt", "YOLO"),
    ("bench_rtdetr_x", "rtdetr-x.pt", "RTDETR"),
    ("bench_yolo11x", "yolo11x.pt", "YOLO"),
]


def main():
    print("=" * 65)
    print("  Starting Extended Benchmark Suite (1280px)")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"  Runs Dir: {RUNS_DIR}")
    print("=" * 65)

    for name, weights, kind in MODELS:
        best_pt = RUNS_DIR / "train" / name / "weights" / "best.pt"
        if best_pt.exists():
            print(f"skip {name}: already trained ({best_pt})", flush=True)
            continue

        print(f"\n▶ Starting Training: {name} ({weights})", flush=True)
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
        except Exception:
            print(f"❌ Failed to train {name}", flush=True)
            traceback.print_exc()

    print("\nAll extended Ultralytics benchmark models processed!", flush=True)

    # ── 1. Classic Fast Baselines ──
    import subprocess
    import sys
    classic_script = HERE / "scripts" / "train_classic_detectors.py"
    yolonas_script = HERE / "scripts" / "train_yolonas.py"
    dfine_script = HERE / "scripts" / "train_dfine.py"
    cascade_script = HERE / "scripts" / "train_cascade_rcnn.py"

    for m in ["Faster-RCNN", "RetinaNet"]:
        ckpt = RUNS_DIR / "classic" / m / "best.pt"
        if ckpt.exists():
            print(f"[SKIP] Classic {m}: already trained ({ckpt})", flush=True)
            continue
        print(f"\n=======================================================", flush=True)
        print(f" ▶ Auto-Launching Classic Detector: {m} (1280px) ...", flush=True)
        print(f"=======================================================", flush=True)
        subprocess.run([sys.executable, str(classic_script), m], check=False)

    # ── 2. Neural Architecture Search: YOLO-NAS ──
    yolonas_ckpt = RUNS_DIR / "yolonas" / "yolo_nas_m" / "ckpt_best.pth"
    if not yolonas_ckpt.exists() and yolonas_script.exists():
        print(f"\n=======================================================", flush=True)
        print(f" ▶ Auto-Launching YOLO-NAS-M (1280px) ...", flush=True)
        print(f"=======================================================", flush=True)
        subprocess.run([sys.executable, str(yolonas_script), "--variant", "yolo_nas_m"], check=False)

    # ── 3. Next-Gen Real-Time DETR: D-FINE / RT-DETRv2 ──
    dfine_cfg = RUNS_DIR / "dfine" / "dfine_teeth_1280.yml"
    if dfine_script.exists():
        print(f"\n=======================================================", flush=True)
        print(f" ▶ Setting up D-FINE / RT-DETRv2 Configuration ...", flush=True)
        print(f"=======================================================", flush=True)
        subprocess.run([sys.executable, str(dfine_script)], check=False)

    # ── 4. Multi-Stage Refinement: Cascade R-CNN (Placed Last) ──
    cascade_ckpt = RUNS_DIR / "classic" / "Cascade-RCNN" / "best.pt"
    if not cascade_ckpt.exists():
        print(f"\n=======================================================", flush=True)
        print(f" ▶ Auto-Launching Cascade R-CNN (1280px) ...", flush=True)
        print(f"=======================================================", flush=True)
        subprocess.run([sys.executable, str(cascade_script)], check=False)

    print("\n🎉 ALL Benchmark Models in Queue Successfully Completed!", flush=True)



if __name__ == "__main__":
    main()

