#!/bin/bash
# ==============================================================================
# Module: modules/task_adapter.sh
# Purpose: Orchestrates Stage 1 - Base English Task Math (Zero-Shot)
# ==============================================================================

source "$(dirname "$0")/../core/environment.sh"
source "$(dirname "$0")/../core/engine.sh"

run_stage1() {
    echo "[*] Triggering Stage 1 Pipeline (English Task Adapters) for $MODEL_NAME"
    local output_base="$OUTPUT_ROOT/S1_English_Task_Adapters"

    # Stage 1 now explicitly uses the 500-sample unified Dataset to build True Instruction-Tuned anchors!
    local data_jsonl="$SUBSET_DIR/universal_cross_dataset_500.jsonl"
    local img_folder="$BASE_DIR/voqa_test_benchmark/test" 

    for STRAT in "${STRATEGIES[@]}"; do
        local out_dir="$output_base/universal_benchmark/$STRAT"
        mkdir -p "$out_dir"
        
        echo ">>> Executing Universal Task Adapter: Strategy=[$STRAT]"
        
        # Execute centralized deepspeed engine for Stage 1 (Rank 8, No Base Adapter)
        run_deepspeed_finetune "$data_jsonl" "$img_folder" "$out_dir" 8 ""
    done
}
