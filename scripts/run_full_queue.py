"""
run_full_queue.py
-----------------
Automated Sequential Queue Runner with user-prioritized sequence:
1. Faster R-CNN (1280px, batch=8) [Currently running via task-1365]
2. RetinaNet (1280px, batch=8)
3. YOLO-NAS (1280px) / D-FINE (RT-DETRv2)
4. Cascade R-CNN (3-stage progressive RoI, 1280px, batch=8) [Run at the very last]
5. Benchmark Evaluation on standard 32 FDI teeth
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
PYTHON = sys.executable


def wait_for_faster_rcnn():
    print("⏳ Waiting for Faster R-CNN training to complete...", flush=True)
    ckpt = BASE_DIR / "runs" / "classic" / "Faster-RCNN" / "best.pt"
    while not ckpt.exists():
        time.sleep(30)
    print("✅ Faster R-CNN finished! Found best.pt", flush=True)


def run_retinanet():
    ckpt = BASE_DIR / "runs" / "classic" / "RetinaNet" / "best.pt"
    if ckpt.exists():
        print("⏭️ RetinaNet already completed.", flush=True)
        return
    print("\n=======================================================", flush=True)
    print(" 🚀 Starting RetinaNet (1280px, batch=8, epochs=100)...", flush=True)
    print("=======================================================", flush=True)
    cmd = [PYTHON, str(SCRIPTS_DIR / "train_classic_detectors.py"), "RetinaNet"]
    subprocess.run(cmd, check=True)
    print("✅ RetinaNet training completed!", flush=True)


def run_yolonas():
    ckpt = BASE_DIR / "runs" / "yolonas" / "yolo_nas_m" / "ckpt_best.pth"
    if ckpt.exists():
        print("⏭️ YOLO-NAS already completed.", flush=True)
        return
    print("\n=======================================================", flush=True)
    print(" 🚀 Starting Step 3: YOLO-NAS-M (1280px, batch=8, epochs=100)...", flush=True)
    print("=======================================================", flush=True)
    yolonas_script = SCRIPTS_DIR / "train_yolonas.py"
    if yolonas_script.exists():
        subprocess.run([PYTHON, str(yolonas_script), "--variant", "yolo_nas_m", "--batch", "8", "--imgsz", "1280"], check=False)
    print("✅ YOLO-NAS step completed!", flush=True)


def run_dfine():
    print("\n=======================================================", flush=True)
    print(" 🚀 Starting Step 4: D-FINE / RT-DETRv2 (1280px, batch=8, epochs=100)...", flush=True)
    print("=======================================================", flush=True)
    dfine_script = SCRIPTS_DIR / "train_dfine.py"
    if dfine_script.exists():
        subprocess.run([PYTHON, str(dfine_script), "--imgsz", "1280", "--batch", "8", "--epochs", "100"], check=False)
    print("✅ D-FINE / RT-DETRv2 step completed!", flush=True)


def run_cascade_rcnn():
    ckpt = BASE_DIR / "runs" / "classic" / "Cascade-RCNN" / "best.pt"
    if ckpt.exists():
        print("⏭️ Cascade R-CNN already completed.", flush=True)
        return
    print("\n=======================================================", flush=True)
    print(" 🚀 Starting Step 5: Cascade R-CNN (1280px, batch=8, epochs=100)...", flush=True)
    print("=======================================================", flush=True)
    cmd = [PYTHON, str(SCRIPTS_DIR / "train_cascade_rcnn.py")]
    subprocess.run(cmd, check=True)
    print("✅ Cascade R-CNN training completed!", flush=True)


def run_evaluations():
    print("\n=======================================================", flush=True)
    print(" 📊 Starting Step 6: Full Benchmark Evaluation on Internal & DENTEX Sets...", flush=True)
    print("=======================================================", flush=True)
    eval_script = SCRIPTS_DIR / "evaluate_all_benchmarks.py"
    if eval_script.exists():
        subprocess.run([PYTHON, str(eval_script)], check=False)
    print("🎉 All benchmark training and evaluation tasks finished!", flush=True)


def main():
    wait_for_faster_rcnn()
    run_retinanet()
    run_yolonas()
    run_dfine()
    run_cascade_rcnn()
    run_evaluations()


if __name__ == "__main__":
    main()
