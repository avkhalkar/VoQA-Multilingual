#!/bin/bash
# ==============================================================================
# Script: run_stage1_english_watermarked.sh
# Purpose: Executes STAGE 1 (English Task Adapters) but RESTRICTED to ONLY:
#          1. voqa_baseline (Zero-shot VoQA)
#          2. baseline_sft (Standard VQA)
#          AND forces both of them to use watermark rendered images ONLY.
# ==============================================================================

# Ensure environment variables are loaded
source "$(dirname "$0")/core/environment.sh"

echo "======================================================================"
echo "  [VoQA Pipeline] Custom Execution: English Stage 1 (Watermarked Only) "
echo "======================================================================"

validate_environment

source "$(dirname "$0")/core/engine.sh"

echo "[*] Triggering Stage 1 Pipeline (English Task Adapters) for $MODEL_NAME"
# Absolute Path Default (The Biggie!)
data_jsonl="${DATA_JSONL:-/scratch/users/anandk03/VoQA_train/VoQA_train_3_3M.json}"
if [[ -z "$IMAGE_FOLDER" ]]; then
    if [[ -d "/mnt/gpu_users/Aditya/rendered_images/eng" ]]; then
        img_folder="/mnt/gpu_users/Aditya/rendered_images/eng"
    elif [[ -d "/scratch/users/anandk03/VoQA_train/watermark_rendering_image" ]]; then
        img_folder="/scratch/users/anandk03/VoQA_train/watermark_rendering_image"
    elif [[ -d "/scratch/users/anandk03/VoQA_train/multilingual/rendered_images/eng" ]]; then
        img_folder="/scratch/users/anandk03/VoQA_train/multilingual/rendered_images/eng"
    else
        img_folder="/mnt/gpu_users/Aditya/rendered_images/eng"
    fi
else
    img_folder="$IMAGE_FOLDER"
fi 

# Parse command line flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --subset)
            echo ">>> [WARNING] SUBSET FLAG DETECTED. Locking to 600-sample tiny JSON."
            data_jsonl="/scratch/users/anandk03/VoQA_train/VoQA_train_subset_600.json"
            ;;
    esac
    shift
done

output_base="$OUTPUT_ROOT/S1_English_Task_Adapters"

# EXACTLY ONLY the first two tasks: voqa_baseline and baseline_sft
CUSTOM_STRATEGIES=("voqa_baseline" "baseline_sft")

for STRAT in "${CUSTOM_STRATEGIES[@]}"; do
    out_dir="$output_base/universal_benchmark/$STRAT"
    mkdir -p "$out_dir"
    
    echo ">>> Executing Universal Task Adapter: Strategy=[$STRAT]"
    
    # -------------------------------------------------------------
    # We assign the direct user-supplied img_folder here to avoid path crashing.
    # -------------------------------------------------------------
    target_img_folder="$img_folder"
    
    # Map the correct conversational schema
    case "$STRAT" in
        "baseline_sft")  conv_target="phi_baseline" ;;
        "voqa_baseline") conv_target="phi_baseline" ;;
        *)               conv_target="phi_qa" ;;
    esac
    
    echo "    - Target Schema:    $conv_target"
    echo "    - Target Images:    $target_img_folder (WATERMARKED)"
    echo "    - Target Output:    $out_dir"
    
    # Execute centralized deepspeed engine for Stage 1 (Rank 8, No Base Adapter) passing mapped conv_version
    run_deepspeed_finetune "$data_jsonl" "$target_img_folder" "$out_dir" 8 "" "$conv_target"
done

echo "[*] Customized Stage 1 Execution Complete."
