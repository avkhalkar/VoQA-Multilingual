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

    local base_adapter_arg=""
    if [[ -n "$base_adapter" ]]; then
        base_adapter_arg="--base_task_adapter '$base_adapter'"
        echo "Base adapter correctly injected: $base_adapter"
    fi

    local cmd=(
        env ENROOT_RUNTIME_PATH=/scratch/users/$USER/enroot ENROOT_CACHE_PATH=/scratch/users/$USER/enroot_cache ENROOT_DATA_PATH=/scratch/users/$USER/enroot_data \
        enroot start --env NVIDIA_DISABLE_REQUIRE=1 --rw --mount /home:/home --mount /scratch:/scratch --mount /mnt:/mnt b200_sandbox bash -c "
        env TRUST_REMOTE_CODE=True HF_TRUST_REMOTE_CODE=1 PYTHONPATH=$BASE_DIR/src/fine-tuning/scripts/core:\$PYTHONPATH python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29500 \
        $BASE_DIR/src/fine-tuning/python_modules/train_orchestrator.py \
        --deepspeed $BASE_DIR/train/scripts/zero3.json \
        --model_name_or_path '/scratch/users/anandk03/pretrained/InternVL2_5-1B' \
        --conv_version '$strategy_type' \
        --data_path '$data_path' \
        --image_folder '$image_folder' \
        --vision_tower '/scratch/users/anandk03/pretrained/InternVL2_5-1B' \
        --output_dir '$output_dir' \
        --lora_rank '$lora_rank' \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --lr_scheduler_type 'cosine' \
        --bf16 True \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --report_to none \
        $base_adapter_arg
        "
    )

    # Execute
    "${cmd[@]}"
}
