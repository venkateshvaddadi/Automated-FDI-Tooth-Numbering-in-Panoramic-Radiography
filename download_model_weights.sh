#!/usr/bin/env bash
# ==============================================================================
# Model Checkpoint Downloader for Automated FDI Tooth Numbering
# Downloads pre-trained models (YOLO11m, RT-DETR-L, D-FINE-M) to weights/ directory
# ==============================================================================

set -e
mkdir -p weights

echo "======================================================================"
echo "  Downloading Evaluated Model Checkpoints"
echo "======================================================================"

# GitHub Release Tag URL (Update when new release assets are attached)
RELEASE_URL="https://github.com/venkateshvaddadi/Automated-FDI-Tooth-Numbering-in-Panoramic-Radiography/releases/download/v1.0"

download_file() {
    local filename="$1"
    local dest="weights/${filename}"
    if [ -f "$dest" ]; then
        echo "✔ ${dest} already exists, skipping."
    else
        echo "⬇ Downloading ${filename}..."
        # If GitHub release asset is available, use curl/wget:
        curl -L "${RELEASE_URL}/${filename}" -o "$dest" || echo "[Note] Asset will be available upon publishing GitHub Release v1.0."
    fi
}

download_file "yolo11m_teeth_1280.pt"
download_file "rtdetr_l_teeth_1280.pt"
download_file "dfine_m_teeth_1280.pth"

echo "======================================================================"
echo "Checkpoints directory: ./weights/"
ls -la weights/
echo "======================================================================"
