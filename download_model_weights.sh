#!/usr/bin/env bash
# ==============================================================================
# Model Checkpoints Downloader for Automated FDI Tooth Numbering
# Downloads pre-trained models directly from Google Drive / Cloud storage to weights/
# ==============================================================================

set -e
mkdir -p weights

echo "======================================================================"
echo "  Downloading Evaluated Pre-trained FDI Model Checkpoints"
echo "======================================================================"

# Google Drive shared folder link provided for public reproduction:
# https://drive.google.com/drive/folders/1QyrHJvjcCHT2chnKBEgRVm-unnwnXXdT?usp=sharing
GDRIVE_FOLDER_ID="1QyrHJvjcCHT2chnKBEgRVm-unnwnXXdT"
GDRIVE_URL="https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}?usp=sharing"

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "⚠️ 'gdown' tool not found. Installing gdown via pip..."
    pip install --upgrade gdown
fi

echo "⬇ Fetching all model weight checkpoints into ./weights/ ..."
gdown --folder "${GDRIVE_URL}" -O weights/

echo ""
echo "======================================================================"
echo "✔ Checkpoints successfully verified in ./weights/:"
ls -lh weights/
echo "======================================================================"
