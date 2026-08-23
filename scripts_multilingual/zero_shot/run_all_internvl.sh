#!/bin/bash
# Multilingual VoQA - Full Pipeline Orchestrator InternVL Tracker
# This orchestrator will sequentially trigger every phase of the pipeline natively mapped for InternVL

set -e 

# Shift scope to root since this script is now nested two directories deep 
cd "$(dirname "$0")/../.."

echo "========================================================"
echo "          Multilingual VoQA Pipeline (InternVL)         "
echo "========================================================"

echo ""
echo ">>> [1/4] Starting Translation Phase (SeamlessM4T)..."
bash scripts_multilingual/01_run_translation.sh

echo ""
echo ">>> [2/4] Starting Watermark Rendering Phase..."
bash scripts_multilingual/02_run_rendering.sh

echo ""
echo ">>> [3/4] Starting Zero-Shot Inference Phase (InternVL)..."
bash scripts_multilingual/zero_shot/03_run_zero_shot_internvl.sh

echo ""
echo ">>> [4/4] Starting Evaluation & Scoring Phase..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)" 2>/dev/null

conda activate internvl
python src/eval/evaluate_multilingual.py

echo ""
echo "========================================================"
echo "          Pipeline Completed Successfully!              "
echo "    Check experiments/evaluation_results.json           "
echo "========================================================"
