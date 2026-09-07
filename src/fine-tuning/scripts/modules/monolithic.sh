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
        local raw_img_folder="$BASE_DIR/voqa_test_benchmark/test/$LANG"
        local render_img_folder="$raw_img_folder/cross_dataset_rerendered"
        
        for STRAT in "${STRATEGIES[@]}"; do
            local out_dir="$output_base/$LANG/universal_benchmark/$STRAT"
            mkdir -p "$out_dir"
            
            echo ">>> Executing Monolithic Stage 2: Lang=[$LANG] | Strat=[$STRAT]"
            
            # --- UNIFIED STRATEGY & IO ROUTER ---
            local target_img_folder="$render_img_folder"
            local conv_target="phi_qa"
            
            case "$STRAT" in
                "baseline_sft") 
                    conv_target="phi_baseline" 
                    target_img_folder="$raw_img_folder" # Pure Control VQA uses RAW images!
                    ;;
                "qa_sft") conv_target="phi_qa" ;;
                "qra_sft") conv_target="phi_stage3" ;;
                "r_qra_sft") conv_target="phi_r_qra" ;;
                "qa_only_sft") conv_target="phi_qa_only" ;;
                "voqa_baseline") conv_target="phi_baseline" ;;
                *) conv_target="phi_qa" ;;
            esac
            # ------------------------------------
            
            # Execute unified deepspeed engine (Rank 8, No Base Adapter) passing the Mapped Strategy
            run_deepspeed_finetune "$data_jsonl" "$target_img_folder" "$out_dir" 8 "" "$conv_target"
        done
    done
}
