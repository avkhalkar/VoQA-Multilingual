#!/bin/bash
#SBATCH --job-name=voqa_multilingual
#SBATCH --output=logs/voqa_%j.out
#SBATCH --error=logs/voqa_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# -------------------------------------------------------------------------
# Multilingual VoQA — Full-Grid SLURM Orchestrator
# Usage: sbatch run_pipeline.sh                    (all 5 languages)
#        sbatch run_pipeline.sh --lang ita          (single language)
# -------------------------------------------------------------------------
MODEL="internvl_1b"
DATASET="gqa"
TARGET_LANG="all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --lang) TARGET_LANG="$2"; shift ;;
    esac
    shift
done

echo "=================================================="
echo "Starting VoQA Multilingual Pipeline"
echo "Model: $MODEL | Dataset: $DATASET | Lang: $TARGET_LANG"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate internvl

mkdir -p logs

# Decide which languages to process
if [ "$TARGET_LANG" == "all" ]; then
    LANGS=("eng" "ita" "fin" "nld" "spa")
else
    LANGS=("$TARGET_LANG")
fi

# =========================================================================
# [1/4] DATA PREPARATION
# =========================================================================
echo "[1/4] Data Preparation"

# python src/data_prep/select_subset.py

# for LANG in "${LANGS[@]}"; do
#     if [ "$LANG" != "eng" ]; then
#         echo "  -> Translation: $LANG"
#         python src/data_prep/translate_m4t.py --lang $LANG
#     fi
# done

# echo "  -> Rendering Watermarks"
# python src/data_prep/render_watermarks.py --dataset $DATASET
# echo "  -> Rendering Concat-Resizing"
# python src/data_prep/render_concatenation_with_resizing.py --dataset $DATASET
# echo "  -> Rendering Concat-Padding"
# python src/data_prep/render_concatenation_without_resizing.py --dataset $DATASET

# =========================================================================
# [2/4] RESOLVE IMAGE DIRECTORIES
# =========================================================================
WATERMARK_DIR="data_multilingual/rendered_images_watermark_${DATASET}"
CONCAT_RESIZE_DIR="data_multilingual/rendered_concat_resizing_${DATASET}"
CONCAT_PAD_DIR="data_multilingual/rendered_concat_padding_${DATASET}"

# =========================================================================
# [3/4] INFERENCE — All langs × prompts × renders × shots (with resume)
# =========================================================================
for LANG in "${LANGS[@]}"; do
    echo ""
    echo "========== INFERENCE: $LANG =========="

    # --- A. Baseline (raw images, text question, 0-shot) ---
    echo "  -> baseline / 0-shot"
    python src/model/run_inference.py --model $MODEL --dataset $DATASET --lang $LANG \
        --prompt_type baseline --shots 0

    # --- B. Unstructured Prompts (rendered, 0-shot) ---
    for RENDER_MODE in watermark concat_resize concat_pad; do
        if [ "$RENDER_MODE" == "watermark" ]; then IMG_DIR="${WATERMARK_DIR}/${LANG}"; fi
        if [ "$RENDER_MODE" == "concat_resize" ]; then IMG_DIR="${CONCAT_RESIZE_DIR}/${LANG}"; fi
        if [ "$RENDER_MODE" == "concat_pad" ]; then IMG_DIR="${CONCAT_PAD_DIR}/${LANG}"; fi

        echo "  -> no_prompt / $RENDER_MODE / 0-shot"
        python src/model/run_inference.py --model $MODEL --dataset $DATASET --lang $LANG \
            --prompt_type no_prompt --shots 0 --image_dir $IMG_DIR

        echo "  -> light / $RENDER_MODE / 0-shot"
        python src/model/run_inference.py --model $MODEL --dataset $DATASET --lang $LANG \
            --prompt_type light --shots 0 --image_dir $IMG_DIR
    done

    # --- C. Structured Workflows (rendered, 0/1/2/4/8-shot + OCR) ---
    for RENDER_MODE in watermark concat_resize concat_pad; do
        if [ "$RENDER_MODE" == "watermark" ]; then IMG_DIR="${WATERMARK_DIR}/${LANG}"; fi
        if [ "$RENDER_MODE" == "concat_resize" ]; then IMG_DIR="${CONCAT_RESIZE_DIR}/${LANG}"; fi
        if [ "$RENDER_MODE" == "concat_pad" ]; then IMG_DIR="${CONCAT_PAD_DIR}/${LANG}"; fi

        for WORKFLOW in short_workflow long_workflow; do
            for SHOTS in 0 1 2 4 8; do
                echo "  -> $WORKFLOW / $RENDER_MODE / ${SHOTS}-shot"
                python src/model/run_inference.py --model $MODEL --dataset $DATASET --lang $LANG \
                    --prompt_type $WORKFLOW --shots $SHOTS --image_dir $IMG_DIR
            done

            echo "  -> $WORKFLOW / $RENDER_MODE / 0-shot+OCR"
            python src/model/run_inference.py --model $MODEL --dataset $DATASET --lang $LANG \
                --prompt_type $WORKFLOW --shots 0 --image_dir $IMG_DIR --use_ocr
        done
    done
done

# =========================================================================
# [4/4] EVALUATION
# =========================================================================
for LANG in "${LANGS[@]}"; do
    echo ""
    echo "========== EVALUATION: $LANG =========="
    python src/eval/evaluate_model.py --model $MODEL --dataset $DATASET --lang $LANG
done

echo "=================================================="
echo "PIPELINE COMPLETE"
echo "=================================================="
