#!/bin/bash
# Multilingual VoQA - Watermark Rendering Script
# Runs the robust stroke-text renderer over the translated labels

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)" 2>/dev/null

echo "Activating internvl conda environment..."
conda activate internvl

# Ensure we are executing from the root of the project
cd "$(dirname "$0")/.."

echo "Starting Multilingual Watermark Render pipeline..."
python src/data_prep/render_watermarks.py
