#!/bin/bash
# Multilingual VoQA - Translation Execution Script
# Ensures WSL Conda environment is active before running

# Attempt to source conda (covers both miniconda and anaconda standard install paths)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)" 2>/dev/null

echo "Activating internvl conda environment..."
conda activate internvl

# Ensure we are executing from the root of the project
cd "$(dirname "$0")/.."

echo "Starting translation pipeline..."
python src/data_prep/translate_m4t.py
