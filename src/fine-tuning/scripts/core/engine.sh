#!/bin/bash
# ==============================================================================
# Script: core/engine.sh
# Purpose: Core DeepSpeed execution function abstracting the PyTorch trainer.
# ==============================================================================

# Expects parameters: (data_path, image_folder, output_dir, lora_rank, base_task_adapter, strategy_type)
run_deepspeed_finetune() {
    local data_path="$1"
    local image_folder="$2"
    local output_dir="$3"
    local lora_rank="${4:-8}"
    local base_adapter="${5:-}"
    local strategy_type="${6:-internvl}" # Default to pure InternVL mapping if strategy absent

    local cmd=(
        deepspeed --num_gpus 1 $BASE_DIR/src/fine-tuning/python_modules/train_orchestrator.py
        --deepspeed $BASE_DIR/train/scripts/zero3.json
        --model_name_or_path "OpenGVLab/InternVL2-1B"
        --version "internvl"
        --conv_version "$strategy_type"
        --data_path "$data_path"
        --image_folder "$image_folder"
        --vision_tower "OpenGVLab/InternVL2-1B-Vision"
        --output_dir "$output_dir"
        --lora_rank "$lora_rank"
        --num_train_epochs 1
        --per_device_train_batch_size 8
        --per_device_eval_batch_size 4
        --learning_rate 1e-5
        --weight_decay 0.
        --warmup_ratio 0.1
        --lr_scheduler_type "cosine"
        --tf32 True
        --model_max_length 2048
        --gradient_checkpointing True
        --report_to none
    )

    if [[ -n "$base_adapter" ]]; then
        cmd+=("--base_task_adapter" "$base_adapter")
    fi

    # Execute
    "${cmd[@]}"
}
