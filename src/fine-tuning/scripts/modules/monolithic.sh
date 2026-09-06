#!/bin/bash
# ==============================================================================
# Module: modules/monolithic.sh
# Purpose: Orchestrates Stage 2 - The Naive Monolithic Trap
# ==============================================================================

source "$(dirname "$0")/../core/environment.sh"
source "$(dirname "$0")/../core/engine.sh"

run_stage2() {
    echo "[*] Triggering Stage 2 Pipeline (Monolithic Multi) for $MODEL_NAME"
    local output_base="$OUTPUT_ROOT/S2_Monolithic_Multi_Adapters"

    for LANG in "${TARGET_LANGS[@]}"; do
        # Stage 2 Monolithic explicitly uses the universal text dataset to prevent overfitting
        local data_jsonl="$SUBSET_DIR/${LANG}/universal_cross_dataset_500.jsonl"
        local img_folder="$BASE_DIR/voqa_test_benchmark/test/$LANG/cross_dataset_rerendered"
        
        for STRAT in "${STRATEGIES[@]}"; do
            local out_dir="$output_base/$LANG/universal_benchmark/$STRAT"
            mkdir -p "$out_dir"
            
            echo ">>> Executing Monolithic Stage 2: Lang=[$LANG] | Strat=[$STRAT]"
            
            # Execute unified deepspeed engine (Rank 8, No Base Adapter)
            run_deepspeed_finetune "$data_jsonl" "$img_folder" "$out_dir" 8 ""
        done
    done
}
