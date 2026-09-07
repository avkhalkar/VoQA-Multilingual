#!/bin/bash
# ==============================================================================
# Script: scripts/data_prep/01_run_translation.sh
# Purpose: Cluster-friendly execution of the Universal SeamlessM4T Translation
# ==============================================================================

# Ensure environment activation maps to SLURM structure 
source ~/.bashrc 2>/dev/null
conda activate internvl 2>/dev/null

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "Initiating Stage 0.1: Native SeamlessM4T Translation..."
python $BASE_DIR/src/fine-tuning/python_modules/data_prep/translate_universal.py
