"""
train_dfine.py
--------------
Training and preparation script for D-FINE / RT-DETRv2 on the Aneesa dental dataset.
Configured for:
  - Resolution: 1280px
  - 48 tooth classes
  - Strict dental quadrant preservation (fliplr=0.0)
  - 100 epochs, seed=0
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = BASE_DIR / "datasets" / "aneesa" / "dataset.yaml"
OUTPUT_DIR = BASE_DIR / "runs" / "dfine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_dfine_config(imgsz=1280, epochs=100, batch_size=4):
    """Generates the YAML config customized for dental radiography at 1280px."""
    config_content = f"""# D-FINE / RT-DETRv2 Dental Configuration
__include__: [
  'configs/runtime.yml',
  'configs/dfine/include/dataloader.yml',
  'configs/dfine/include/optimizer.yml',
  'configs/dfine/include/dfine_hgnetv2.yml',
]

output_dir: {OUTPUT_DIR}

epoches: {epochs}
train_dataloader:
  dataset:
    img_folder: {BASE_DIR}/datasets/aneesa/images/train
    ann_file: {BASE_DIR}/datasets/aneesa/annotations/train.json
    transforms:
      ops:
        - {{type: 'Resize', size: [{imgsz}, {imgsz}]}}
        - {{type: 'RandomPhotometricDistort', p: 0.5}}
        # Note: Horizontal flip is strictly disabled (p=0.0) for tooth quadrants
  total_batch_size: {batch_size}
  num_workers: 4

val_dataloader:
  dataset:
    img_folder: {BASE_DIR}/datasets/aneesa/images/val
    ann_file: {BASE_DIR}/datasets/aneesa/annotations/val.json
    transforms:
      ops:
        - {{type: 'Resize', size: [{imgsz}, {imgsz}]}}
  total_batch_size: {batch_size}
  num_workers: 4

num_classes: 48
remap_mscoco_category: False
"""
    cfg_file = OUTPUT_DIR / "dfine_teeth_1280.yml"
    cfg_file.write_text(config_content)
    print(f"✅ Generated D-FINE configuration: {cfg_file}")
    return cfg_file


def main():
    parser = argparse.ArgumentParser(description="Train D-FINE / RT-DETRv2 on dental dataset")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    print("=" * 65)
    print(f"  D-FINE / RT-DETRv2 Dental Setup ({args.imgsz}px)")
    print(f"  Classes    : 48 teeth")
    print(f"  Epochs     : {args.epochs}")
    print("=" * 65)

    cfg_file = generate_dfine_config(imgsz=args.imgsz, epochs=args.epochs, batch_size=args.batch)
    print("\nSetup ready:")
    print("1. To run training when ready:")
    print(f"   python train.py -c {cfg_file} --use-amp --seed 0")


if __name__ == "__main__":
    main()
