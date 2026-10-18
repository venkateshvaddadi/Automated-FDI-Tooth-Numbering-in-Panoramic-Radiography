"""
train_faster_rcnn_after_dfine.py
---------------------------------
Watches for D-FINE-M training completion, then automatically launches
Faster R-CNN (ResNet-50-FPN) retraining on the full Aneesa Dental Dataset.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DFINE_LOG = BASE_DIR / "runs" / "dfine_m_1280" / "log.txt"

def wait_for_dfine():
    print("Waiting for D-FINE-M (Epoch 100) to conclude cleanly...")
    while True:
        if DFINE_LOG.exists():
            try:
                lines = DFINE_LOG.read_text().splitlines()
                # Check if epoch 99/100 is logged
                for line in reversed(lines[-20:]):
                    if '"epoch": 99' in line or '"epoch": 100' in line:
                        print("D-FINE-M training reached epoch 100! Proceeding to Faster R-CNN...")
                        time.sleep(10)
                        return
            except Exception:
                pass
        
        # Check if D-FINE process is still running
        try:
            res = subprocess.run(["pgrep", "-f", "dfine_repo/train.py"], stdout=subprocess.PIPE)
            if not res.stdout.strip():
                print("D-FINE-M process has terminated. Proceeding to Faster R-CNN...")
                time.sleep(5)
                return
        except Exception:
            pass

        time.sleep(15)

def main():
    wait_for_dfine()
    
    # Ensure GPU cache is clean
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("  STARTING FASTER R-CNN RETRAINING (ResNet-50-FPN, 1280px, 100 Epochs)")
    print("=" * 70)

    from scripts.train_classic_detectors import train_model
    train_model("Faster-RCNN", epochs=100, batch_size=8, lr=0.001)

if __name__ == "__main__":
    main()
