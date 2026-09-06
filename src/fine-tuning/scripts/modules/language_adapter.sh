#!/bin/bash
# ==============================================================================
# Module: modules/language_adapter.sh
# Purpose: Orchestrates Stage 3 (Pure Base) & Stage 4 (Exhaustive Target) Modularity
# ==============================================================================

source "$(dirname "$0")/../core/environment.sh"
source "$(dirname "$0")/../core/engine.sh"

run_stage3() {
    echo "[*] Triggering Stage 3 Pipeline (4+8 Pure Modularity) for $MODEL_NAME"
    local output_base="$OUTPUT_ROOT/S3_Pure_Language_Adapters"
    local task_adapter_base="$OUTPUT_ROOT/S1_English_Task_Adapters/universal_benchmark/baseline_sft" # Root anchor

    for LANG in "${TARGET_LANGS[@]}"; do
        local data_jsonl="$SUBSET_DIR/${LANG}/universal_cross_dataset_500.jsonl"
        local img_folder="$BASE_DIR/voqa_test_benchmark/test/$LANG/cross_dataset_rerendered"
        
        local out_dir="$output_base/$LANG/universal_lang_sft"
        mkdir -p "$out_dir"
        
        echo ">>> Executing Universal Language Adapter: Lang=[$LANG]"
        
        # Engine called with Rank 4 and Base Adapter passed in for stacking
        run_deepspeed_finetune "$data_jsonl" "$img_folder" "$out_dir" 4 "$task_adapter_base"
    done
}

run_stage4() {
    echo "[*] Triggering Stage 4 Pipeline (32+8 Exhaustive Modularity) for $MODEL_NAME"
    local output_base="$OUTPUT_ROOT/S4_Exhaustive_Language_Adapters"
    local task_adapter_base="$OUTPUT_ROOT/S1_English_Task_Adapters"

    for LANG in "${TARGET_LANGS[@]}"; do
        # Language learning uses the universal mix to prevent overfitting to specific task vocabulary
        local data_jsonl="$SUBSET_DIR/${LANG}/universal_cross_dataset_500.jsonl"
        local img_folder="$BASE_DIR/voqa_test_benchmark/test/$LANG/cross_dataset_rerendered"
        
        for STRAT in "${STRATEGIES[@]}"; do
            # Anchor the logic dynamically to the universal baseline for consistency
            local target_s1_adapter="$task_adapter_base/universal_benchmark/$STRAT"
            local out_dir="$output_base/$LANG/universal_benchmark/$STRAT"
            mkdir -p "$out_dir"
            
            echo ">>> Executing Exhaustive Strategy Adapter: Lang=[$LANG] | Strat=[$STRAT]"
            
            # Engine called with Rank 4 and strict strategy-mapped Base Adapter
            run_deepspeed_finetune "$data_jsonl" "$img_folder" "$out_dir" 4 "$target_s1_adapter"
        done
    done
}
