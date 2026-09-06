#!/bin/bash
# ==============================================================================
# Script: core/environment.sh
# Purpose: Validates execution context and sets unified global variables.
# ==============================================================================

export MODEL_NAME="internvl-1B"
# Dynamically resolve base directory based on execution context (Compatible with WSL and SLURM Scratch Space)
export BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export SUBSET_DIR="$BASE_DIR/src/fine-tuning/data/subsets"
export OUTPUT_ROOT="$BASE_DIR/output_tuning/$MODEL_NAME"

export DATASETS=("gqa" "pope" "scienceqa" "textvqa" "vqav2")
export STRATEGIES=("baseline_sft" "qa_sft" "qra_sft" "r_qra_sft" "qa_only_sft" "voqa_baseline" "r_qa" "rqra")
export TARGET_LANGS=("ita" "fin" "nld" "spa")

validate_environment() {
    if [[ "$CONDA_DEFAULT_ENV" != "internvl" ]]; then
        echo "[!] CRITICAL ERROR: 'internvl' conda environment is not active."
        echo "    Execution blocked to prevent dependency contamination."
        exit 1
    fi
}
