#!/bin/bash
# ==============================================================================
# Script: run_stage3_multilingual_lora.sh
# Purpose: Executes STAGE 3 (Language Adapters) iteratively linking Stage 1 Tasks.
#          Combines: Re-Rendered Sub-Images + Translated JSONL + Base English LoRA
# ==============================================================================

#SBATCH --job-name=voqa_s3
#SBATCH --partition=mig45g,mig90g
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=s3_train_%j.log
#SBATCH --error=s3_train_%j.err

source "$(dirname "$0")/core/environment.sh"

echo "======================================================================"
echo "  [VoQA Pipeline] Execution: Stage 3 Multilingual Language Adapters   "
echo "======================================================================"

validate_environment
source "$(dirname "$0")/core/engine.sh"

# Default configuration parameters
CURRENT_LANGUAGE="ita"
CUSTOM_STRATEGIES=("voqa_baseline" "baseline_sft")

# Parse command line flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --language) CURRENT_LANGUAGE="$2"; shift ;;
    esac
    shift
done

echo "[*] Triggering Stage 3 Modular Pipeline for $MODEL_NAME"
echo "    - Target Language   : $CURRENT_LANGUAGE"

# Dataset mappings based on Language target logic
data_jsonl="/scratch/users/anandk03/VoQA_train/multilingual/${CURRENT_LANGUAGE}_VoQA_train_3_3M.json"
img_folder="/mnt/gpu_users/Aditya/rendered_images/${CURRENT_LANGUAGE}"

# Verify critical dataset payloads explicitly
if [[ ! -f "$data_jsonl" ]]; then
    echo "[!] FATAL: Translated JSONL data missing at $data_jsonl"
    exit 1
fi
if [[ ! -d "$img_folder" ]]; then
    echo "[!] FATAL: High-capacity rendered image payload missing at $img_folder"
    exit 1
fi

output_base="$OUTPUT_ROOT/S3_Multilingual_Language_Adapters"
s1_base="$OUTPUT_ROOT/S1_English_Task_Adapters"

for STRAT in "${CUSTOM_STRATEGIES[@]}"; do
    out_dir="$output_base/universal_benchmark/$STRAT/$CURRENT_LANGUAGE"
    mkdir -p "$out_dir"
    
    # Resolving physical link back to the exact Stage 1 Task Adapter
    base_task_adapter_path="$s1_base/universal_benchmark/$STRAT"
    
    echo ">>> Executing Modular Language Adapter: Strategy=[$STRAT] -> Lang=[$CURRENT_LANGUAGE]"
    
    # Map conversational formatting identically to Phase 1
    case "$STRAT" in
        "baseline_sft")  conv_target="phi_baseline" ;;
        "voqa_baseline") conv_target="phi_baseline" ;;
        *)               conv_target="phi_qa" ;;
    esac
    
    echo "    - Base Task LoRA  : $base_task_adapter_path"
    echo "    - Format Schema   : $conv_target"
    echo "    - Target Output   : $out_dir"
    
    # Execute deepspeed with Stage 3 specific constraint (Rank 4, explicit multi-lora stacking)
    run_deepspeed_finetune "$data_jsonl" "$img_folder" "$out_dir" 4 "$base_task_adapter_path" "$conv_target"
done

echo "[*] Stage 3 Deployment Concluded."
