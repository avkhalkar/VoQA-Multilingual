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
    local task_adapter_base="$OUTPUT_ROOT/S1_English_Task_Adapters/universal_benchmark/voqa_baseline" # Root anchor for VoQA

    for LANG in "${TARGET_LANGS[@]}"; do
        local data_jsonl="$SUBSET_DIR/${LANG}/universal_cross_dataset_500.jsonl"
        local img_folder="$BASE_DIR/voqa_test_benchmark/test/$LANG/cross_dataset_rerendered"
        
        local out_dir="$output_base/$LANG/universal_lang_sft"
        mkdir -p "$out_dir"
        
        echo ">>> Executing Universal Language Adapter: Lang=[$LANG]"
        
        # Engine called with Rank 4, Base Adapter (baseline_sft), and Baseline PyTorch string explicitly
        run_deepspeed_finetune "$data_jsonl" "$img_folder" "$out_dir" 4 "$task_adapter_base" "phi_baseline"
    done
}

run_stage4() {
    echo "[*] Triggering Stage 4 Pipeline (32+8 Exhaustive Modularity) for $MODEL_NAME"
    local output_base="$OUTPUT_ROOT/S4_Exhaustive_Language_Adapters"
    local task_adapter_base="$OUTPUT_ROOT/S1_English_Task_Adapters"

    for LANG in "${TARGET_LANGS[@]}"; do
        # Language learning uses the universal mix to prevent overfitting to specific task vocabulary
        local data_jsonl="$SUBSET_DIR/${LANG}/universal_cross_dataset_500.jsonl"
        local raw_img_folder="$BASE_DIR/voqa_test_benchmark/test/$LANG"
        local render_img_folder="$raw_img_folder/cross_dataset_rerendered"
        
        for STRAT in "${STRATEGIES[@]}"; do
            # Anchor the logic dynamically to the universal baseline for consistency
            local target_s1_adapter="$task_adapter_base/universal_benchmark/$STRAT"
            local out_dir="$output_base/$LANG/universal_benchmark/$STRAT"
            mkdir -p "$out_dir"
            
            echo ">>> Executing Exhaustive Strategy Adapter: Lang=[$LANG] | Strat=[$STRAT]"
            
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
            
            # Engine called with Rank 4 and strict strategy-mapped Base Adapter
            run_deepspeed_finetune "$data_jsonl" "$target_img_folder" "$out_dir" 4 "$target_s1_adapter" "$conv_target"
        done
    done
}
