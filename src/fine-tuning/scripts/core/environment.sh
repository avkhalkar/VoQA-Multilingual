#!/bin/bash

export MODEL_NAME="internvl2_5-1B"

export BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

export SUBSET_DIR="$BASE_DIR/src/fine-tuning/data/subsets"
export OUTPUT_ROOT="$BASE_DIR/output_tuning/$MODEL_NAME"

export DATASETS=(
    "coco"
    "gqa"
    "ocr_vqa"
    "textvqa"
    "vg_100k"
    "vg_100k_2"
)

export STRATEGIES=(
    "voqa_baseline"
    "baseline_sft"
)

export TARGET_LANGS=("ita" "fin" "nld" "spa")

validate_environment() {
    if [[ "$CONDA_DEFAULT_ENV" != "internvl" ]]; then
        echo "[!] CRITICAL ERROR: 'internvl' conda environment is not active."
        echo "    Execution blocked to prevent dependency contamination."
        exit 1
    fi
}
