#!/bin/bash
# Multilingual VoQA - Zero-Shot Inference Script
# Bound explicitly to InternVL2_5-1B execution wrapper

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)" 2>/dev/null

echo "Activating internvl conda environment..."
conda activate internvl

# Shift scope to root since this script is now nested two directories deep 
cd "$(dirname "$0")/../.."

echo "Starting Multilingual InternVL Inference pipeline..."
python src/run_zero_shot_internvl.py
