#!/bin/bash
set -e

echo "=== Pipeline step 1: Waiting for YOLO-NAS-M to complete ==="
while kill -0 3323 2>/dev/null; do
    sleep 30
done

# Check if YOLO-NAS finished or if any other python process is running train_yolonas
while pgrep -f "train_yolonas.py" > /dev/null; do
    sleep 20
done

echo "=== YOLO-NAS-M training completed! ==="
echo "=== Pipeline step 2: Resuming Cascade R-CNN (Epoch 51 -> 100) ==="
/home/venkatesh/anaconda3/envs/yolov8/bin/python scripts/train_cascade_rcnn.py

echo "=== Cascade R-CNN completed! ==="
