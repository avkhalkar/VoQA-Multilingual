#!/bin/bash
# ==============================================================================
# Script: scripts/data_prep/02_run_rendering.sh
# Purpose: Cluster-friendly execution of the Render/Watermark Overlay logic
# ==============================================================================

source ~/.bashrc 2>/dev/null
conda activate internvl 2>/dev/null

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "Initiating Stage 0.2: Native OpenCV/Pillow Watermarking..."
python $BASE_DIR/src/fine-tuning/python_modules/data_prep/render_universal.py
